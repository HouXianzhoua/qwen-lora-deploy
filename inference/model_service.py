# inference/model_service.py
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# 统一从 config 读取默认参数与路径
from config import (
    MODEL_PATH,
    LORA_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_DO_SAMPLE,
    MAX_WORKERS_THREAD_POOL,
)

# 线程池用于把阻塞的生成放到线程中执行（API 场景需要）
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD_POOL)

# --- 内部状态（单例） ---
_MODEL = None
_TOKENIZER = None
_IS_LOADED = False


def _format_prompt(instruction: str) -> str:
    """
    保持与训练时一致的提示模板。
    训练脚本里使用了： f"问题：{inst}\n回答：{resp}"
    推理时应当构造： f"问题：{inst}\n回答："
    """
    return f"问题：{instruction}\n回答："


def load_model(
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    device_map: Optional[str] = "auto",
    dtype: Optional[torch.dtype] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基座模型与 LoRA（可选），返回 (model, tokenizer)。
    也会把模型缓存到模块级单例，便于多处复用。
    """
    global _MODEL, _TOKENIZER, _IS_LOADED

    if _IS_LOADED and _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    base_model_path = base_model_path or MODEL_PATH
    lora_path = lora_path or LORA_PATH

    # 自动 dtype：若未指定，GPU 用 float16，CPU 保持默认
    if dtype is None and torch.cuda.is_available():
        dtype = torch.float16

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # 与训练保持一致：训练时可能补了 [PAD]
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 若 tokenizer 扩展了 vocab，需要对齐
    model.resize_token_embeddings(len(tokenizer))

    # 加载并合并 LoRA（可选）
    if lora_path and os.path.exists(lora_path):
        # 既支持传 final_model 目录，也支持传到 finetune/output 上层
        # 优先尝试 final_model
        lora_dir = lora_path
        if os.path.isdir(lora_path) and os.path.exists(os.path.join(lora_path, "final_model")):
            lora_dir = os.path.join(lora_path, "final_model")

        model = PeftModel.from_pretrained(model, lora_dir)
        # 合并权重，推理更快更省显存
        model = model.merge_and_unload()

    # 编译（可用则编译）
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass  # 某些环境/驱动不支持没关系

    model.eval()

    _MODEL, _TOKENIZER, _IS_LOADED = model, tokenizer, True
    return _MODEL, _TOKENIZER


def generate_text(
    instruction: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    do_sample: bool = DEFAULT_DO_SAMPLE,
) -> str:
    """
    同步生成接口：从单例模型生成文本。
    """
    if not _IS_LOADED or _MODEL is None or _TOKENIZER is None:
        # 懒加载（使用默认路径）
        load_model()

    prompt = _format_prompt(instruction)
    inputs = _TOKENIZER(prompt, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}

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
    return _TOKENIZER.decode(outputs[0], skip_special_tokens=True)


async def async_generate_text(
    instruction: str,
    **kwargs,
) -> str:
    """
    异步生成接口：把同步的 generate_text 丢到线程池，适配 FastAPI。
    """
    loop = asyncio.get_event_loop()
    func = partial(generate_text, instruction, **kwargs)
    return await loop.run_in_executor(_executor, func)


def init_and_register_for_api(
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
):
    """
    可选：用于 FastAPI 启动时加载并注册到 api.model_registry（如果项目使用该注册器）。
    这样无需在 API 层重复维护模型逻辑。
    """
    from api.model_registry import registry  # 延迟导入避免循环依赖
    model, tokenizer = load_model(base_model_path, lora_path)
    registry.set_model(model, tokenizer)

