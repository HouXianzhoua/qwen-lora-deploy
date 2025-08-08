# inference/model_service.py
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from api.model_registry import registry

from config import (
    MODEL_PATH,
    LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_DO_SAMPLE,
    MAX_WORKERS_THREAD_POOL,
)

# ===== 批量参数（可用环境变量覆盖）=====
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "32"))       # 每批最多合并多少请求
BATCH_INTERVAL_MS = int(os.getenv("BATCH_INTERVAL_MS", "40")) # 收集窗口（毫秒）

logger = logging.getLogger("model_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# 线程池（目前未直接使用，保留以便未来把阻塞计算丢到线程）
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD_POOL)

# --- 模块级单例 ---
_MODEL: Optional[AutoModelForCausalLM] = None
_TOKENIZER: Optional[AutoTokenizer] = None
_IS_LOADED: bool = False

def _format_prompt(instruction: str) -> str:
    """
    与训练保持一致：
    训练: f"问题：{inst}\\n回答：{resp}"
    推理: f"问题：{inst}\\n回答："
    """
    return f"问题：{instruction}\n回答："

def _ensure_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    """
    确保 tokenizer/model 有可用的 pad/eos 标记，避免 generate 行为异常。
    若新增了 token，需 resize_token_embeddings。
    """
    changed = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("pad_token 不存在，使用 eos_token 作为 pad。")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("pad_token/eos_token 都不存在，新增 [PAD] 作为 pad。")
            changed = True
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
        logger.info("eos_token 不存在，使用 pad_token 作为 eos。")
        changed = True
    if changed:
        model.resize_token_embeddings(len(tokenizer))

def _normalize_paths(base_model_path: Optional[str], lora_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    允许 lora_path 传 /app/finetune/output 或 /app/finetune/output/final_model
    统一归一到最终可加载的目录。
    """
    bmp = base_model_path or MODEL_PATH
    lrp = lora_path or LORA_PATH
    if lrp and os.path.isdir(lrp) and os.path.exists(os.path.join(lrp, "final_model")):
        lrp = os.path.join(lrp, "final_model")
    return bmp, lrp

def load_model(
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    device_map: Optional[str] = "auto",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基座模型与 LoRA（可选），返回 (model, tokenizer)，并缓存为单例。
    幂等：若已加载则直接返回。
    """
    global _MODEL, _TOKENIZER, _IS_LOADED
    if _IS_LOADED and _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    base_model_path, lora_path = _normalize_paths(base_model_path, lora_path)

    if not os.path.isdir(base_model_path):
        raise FileNotFoundError(f"Base model path not found: {base_model_path}")
    if lora_path and not os.path.isdir(lora_path):
        logger.warning(f"LoRA path not found, fallback to base model only: {lora_path}")
        lora_path = None

    # dtype 自动选择：GPU 用 float16，CPU 用 None（默认）
    if dtype is None and torch.cuda.is_available():
        dtype = torch.float16

    logger.info(f"加载基座模型: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
    )

    _ensure_tokens(tokenizer, model)

    if lora_path:
        logger.info(f"加载并合并 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # 失败忽略
        except Exception as e:
            logger.info(f"torch.compile 不可用/失败，忽略: {e}")

    model.eval()
    _MODEL, _TOKENIZER, _IS_LOADED = model, tokenizer, True
    logger.info("模型加载完成。")
    return _MODEL, _TOKENIZER

def generate_text(
    instruction: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
) -> str:
    """
    同步生成：返回仅补全部分（不包含 prompt）。
    """
    if not instruction or not instruction.strip():
        raise ValueError("instruction 不能为空。")

    if not _IS_LOADED or _MODEL is None or _TOKENIZER is None:
        load_model()  # 懒加载

    prompt = _format_prompt(instruction.strip())
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}
    input_len = int(inputs["attention_mask"].sum().item())  # 真实长度

    with torch.no_grad():
        outputs = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=_TOKENIZER.pad_token_id,
            eos_token_id=_TOKENIZER.eos_token_id,
        )

    gen_ids = outputs[0, input_len:]
    text = _TOKENIZER.decode(gen_ids, skip_special_tokens=True).strip()
    return text

# ===== 批量队列与任务 =====
class _BatchItem:
    __slots__ = ("prompt", "params", "future")
    def __init__(self, prompt: str, params: dict, loop: asyncio.AbstractEventLoop):
        self.prompt = prompt
        self.params = params
        self.future = loop.create_future()

_batch_queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
_batch_worker_task: Optional[asyncio.Task] = None

async def _batch_worker():
    """
    从队列取请求，按窗口/上限合批，一次性 generate，然后把结果分发回各自 Future。
    """
    model = registry.get_model()
    tokenizer = registry.get_tokenizer()
    assert model is not None and tokenizer is not None, "Model/tokenizer not initialized."

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    interval = BATCH_INTERVAL_MS / 1000.0

    while True:
        batch: List[_BatchItem] = []
        try:
            # 1) 至少拿到一个
            first: _BatchItem = await _batch_queue.get()
            batch.append(first)

            # 2) 在 interval 窗口内尽量多收集（不超过上限）
            t0 = asyncio.get_running_loop().time()
            while len(batch) < BATCH_MAX_SIZE:
                timeout = max(0.0, t0 + interval - asyncio.get_running_loop().time())
                try:
                    nxt = await asyncio.wait_for(_batch_queue.get(), timeout=timeout)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            # 3) 构造 batch 输入
            prompts = [ _format_prompt(it.prompt.strip()) for it in batch ]
            ref = batch[0].params
            max_new_tokens = ref.get("max_new_tokens", 128)
            temperature    = ref.get("temperature", 0.7)
            top_p          = ref.get("top_p", 0.9)
            do_sample      = ref.get("do_sample", True)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # 每条样本真实输入长度
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            texts: List[str] = []
            for i in range(outputs.size(0)):
                start = int(input_lens[i])
                gen_ids = outputs[i, start:]
                texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

            # 4) 回填结果
            for it, txt in zip(batch, texts):
                if not it.future.done():
                    it.future.set_result(txt)

        except Exception as e:
            logger.error(f"Batch worker error: {e}", exc_info=True)
            for it in batch:
                if not it.future.done():
                    it.future.set_exception(e)
            # 继续循环，避免任务崩溃后不再处理请求

async def async_generate_text(
    instruction: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """
    批量推理入口：将请求入队，等待后台 batch worker 合并后统一生成。
    """
    assert registry.is_loaded, "Model not initialized. Call init_and_register_for_api() first."

    loop = asyncio.get_running_loop()
    item = _BatchItem(
        prompt=instruction,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        },
        loop=loop,
    )
    await _batch_queue.put(item)
    return await item.future

def init_and_register_for_api(
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
):
    """
    FastAPI 启动时调用：加载模型并注册到 registry（幂等）。
    还会确保 batch worker 已启动（若挂掉会重启）。
    """
    # 幂等：如果已经在 registry 里了，就不再重复加载
    if registry.is_loaded and registry.get_model() is not None and registry.get_tokenizer() is not None:
        _ensure_batch_worker_running()
        return

    model, tokenizer = load_model(base_model_path, lora_path)
    registry.set_model(model, tokenizer)
    _ensure_batch_worker_running()

def _ensure_batch_worker_running():
    global _batch_worker_task
    loop = asyncio.get_event_loop()
    if _batch_worker_task is None or _batch_worker_task.done():
        _batch_worker_task = loop.create_task(_batch_worker())

