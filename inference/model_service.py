# inference/model_service.py
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Tuple
from api.model_registry import registry
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

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
BATCH_MAX_SIZE = int(os.getenv("BATCH_MAX_SIZE", "32"))     # 每批最多合并多少请求
BATCH_INTERVAL_MS = int(os.getenv("BATCH_INTERVAL_MS", "40"))  # 收集窗口（毫秒）

logger = logging.getLogger("model_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# 线程池：把阻塞的生成放到线程中执行，适配 FastAPI 异步场景
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD_POOL)

# --- 模块级单例 ---
_MODEL = None
_TOKENIZER = None
_IS_LOADED = False


def _format_prompt(instruction: str) -> str:
    """
    保持与训练时一致：
    训练拼接: f"问题：{inst}\\n回答：{resp}"
    推理拼接: f"问题：{inst}\\n回答："
    """
    return f"问题：{instruction}\n回答："


def _ensure_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
    """
    确保 tokenizer/model 有可用的 pad/eos 标记，避免 generate 报错或行为异常。
    - 若 tokenizer 无 pad_token：尽量用 eos_token 充当；若也无 eos，就增补一个 [PAD]
    - 若新增了 token，需 resize_token_embeddings
    """
    changed = False

    # 1) pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("pad_token 不存在，使用 eos_token 作为 pad。")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("pad_token/eos_token 都不存在，新增 [PAD] 作为 pad。")
            changed = True

    # 2) eos_token 至少要有一个 ID
    if tokenizer.eos_token_id is None:
        # 兜底：如果仍然没有 eos，就让 eos=pad
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

    # 确保 pad/eos 可用（必要时会扩词表并 resize）
    _ensure_tokens(tokenizer, model)

    # 合并 LoRA（若提供）
    if lora_path:
        logger.info(f"加载并合并 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    # 编译（有则用，失败忽略）
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
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
    同步生成：返回**仅补全部分**（不包含 prompt）。
    """
    if not instruction:
        raise ValueError("instruction 不能为空。")

    if not _IS_LOADED or _MODEL is None or _TOKENIZER is None:
        load_model()  # 懒加载

    prompt = _format_prompt(instruction)
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

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

    # 只取补全（去掉提示词）
    gen_ids = outputs[:, input_length:]
    text = _TOKENIZER.decode(gen_ids[0], skip_special_tokens=True).strip()
    return text

# ===== 批量队列与任务 =====
class _BatchItem:
    __slots__ = ("prompt", "params", "future")
    def __init__(self, prompt: str, params: dict, loop: asyncio.AbstractEventLoop):
        self.prompt = prompt
        self.params = params
        self.future = loop.create_future()

_batch_queue: asyncio.Queue[_BatchItem] = asyncio.Queue()
_batch_worker_task: asyncio.Task | None = None


async def _batch_worker():
    """
    周期性从队列取请求，按窗口/上限合批，一次性 generate，然后把结果分发回各自 Future。
    """
    model = registry.get_model()
    tokenizer = registry.get_tokenizer()
    assert model is not None and tokenizer is not None, "Model/tokenizer not initialized."

    # pad/eos 兜底
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    interval = BATCH_INTERVAL_MS / 1000.0

    while True:
        try:
            # 1) 先拿一个（阻塞直到有）
            first: _BatchItem = await _batch_queue.get()
            batch = [first]

            # 2) 在 interval 窗口内尽量多收集一些（不超过上限）
            t0 = asyncio.get_event_loop().time()
            while len(batch) < BATCH_MAX_SIZE:
                timeout = max(0.0, t0 + interval - asyncio.get_event_loop().time())
                try:
                    nxt = await asyncio.wait_for(_batch_queue.get(), timeout=timeout)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break  # 窗口截止

            # 3) 组 batch 输入
            prompts = [it.prompt for it in batch]
            # 取第一条的采样参数作为全批参数（也可做更细粒度分桶）
            ref = batch[0].params
            max_new_tokens = ref.get("max_new_tokens", 128)
            temperature    = ref.get("temperature", 0.7)
            top_p          = ref.get("top_p", 0.9)
            do_sample      = ref.get("do_sample", True)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

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

            # 对齐每条样本的起始位置，逐条 decode
            input_lens = inputs["input_ids"].shape[1]
            texts = []
            for i in range(outputs.size(0)):
                gen_ids = outputs[i, input_lens:]  # 简化：同一 padding 长度
                texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

            # 4) 回填结果
            for it, txt in zip(batch, texts):
                if not it.future.done():
                    it.future.set_result(txt)

        except Exception as e:
            # 将本批次所有 future 置错，避免 await 永久悬挂
            for it in batch if "batch" in locals() else []:
                if not it.future.done():
                    it.future.set_exception(e)


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
    # 防呆：若尚未初始化，直接报错（也可以在这里懒初始化）
    assert registry.is_loaded, "Model not initialized. Call init_and_register_for_api() first."

    loop = asyncio.get_event_loop()
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
    FastAPI 启动时调用：加载模型并注册到 registry，供 API 直接取用（可选）。
    """
    from api.model_registry import registry  # 延迟导入避免循环依赖
    model, tokenizer = load_model(base_model_path, lora_path)
    registry.set_model(model, tokenizer)
    
    # 启动 batch worker（只启动一次）
    global _batch_worker_task
    if _batch_worker_task is None:
        loop = asyncio.get_event_loop()
        _batch_worker_task = loop.create_task(_batch_worker())
