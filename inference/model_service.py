# inference/model_service.py
import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Tuple

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


async def async_generate_text(instruction: Optional[str] = None, **kwargs) -> str:
    """
    异步生成：兼容传入 prompt=... 的写法，避免 500 错。
    """
    if instruction is None:
        instruction = kwargs.pop("prompt", None)
    if instruction is None:
        raise ValueError("instruction（或 prompt）是必填参数。")

    loop = asyncio.get_event_loop()
    func = partial(generate_text, instruction, **kwargs)
    return await loop.run_in_executor(_executor, func)


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

