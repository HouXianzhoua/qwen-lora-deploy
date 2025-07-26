# api/infer.py
import os
import torch
import asyncio
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional
from config import (
    MODEL_PATH,
    LORA_PATH,
    MAX_WORKERS_THREAD_POOL,
    DEFAULT_MAX_NEW_TOKENS,  # ✅ 新增
    DEFAULT_TEMPERATURE,     # ✅ 新增
    DEFAULT_TOP_P,           # ✅ 新增
    DEFAULT_DO_SAMPLE        # ✅ 新增
)
from api.model_registry import registry
from api.logger import logger  # <-- 新增

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS_THREAD_POOL)

def init_model(model_path: str = None, lora_path: str = None):
    model_path = model_path or MODEL_PATH
    lora_path = lora_path or LORA_PATH
    try:
        # ✅ 替换 print
        logger.info(f"🔧 正在加载基础模型：{model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        if lora_path and os.path.exists(lora_path):
            logger.info(f"📎 加载 LoRA 权重：{lora_path}")
            model = PeftModel.from_pretrained(base_model, lora_path)
            logger.info("🔄 正在合并 LoRA 权重...")
            model = model.merge_and_unload()
            logger.info("✅ LoRA 权重已合并")
        else:
            logger.warning("⚠️ 未提供 LoRA，使用原始模型")
            model = base_model
        model.eval()
        
        device_info = str(model.device)
        memory_mb = torch.cuda.memory_allocated()/1024**2
        # ✅ 替换 print
        logger.info(f"🎉 模型加载成功！运行设备：{device_info}")
        logger.info(f"   架构：{model.config.architectures}")
        logger.info(f"   显存占用：{memory_mb:.2f} MB")
        
        registry.set_model(model, tokenizer)
    except Exception as e:
        # ✅ 记录错误
        logger.error(f"❌ 模型加载失败：{type(e).__name__}: {e}", exc_info=True)
        raise

def safe_generate_response(prompt: str, **kwargs):
    model = registry.get_model()
    tokenizer = registry.get_tokenizer()

    max_new_tokens = kwargs.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS)
    temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
    top_p = kwargs.get('top_p', DEFAULT_TOP_P)
    do_sample = kwargs.get('do_sample', DEFAULT_DO_SAMPLE)

    logger.debug(f"开始生成: prompt='{prompt[:50]}...', max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")

    # ✅ 正确方式：先获取 inputs，再移动到设备
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ✅ 逐个移动 tensor

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.debug(f"生成完成，输出长度: {len(result)}")
    return result

async def async_generate_response(prompt: str, **kwargs):
    loop = asyncio.get_event_loop()
    func = partial(safe_generate_response, prompt, **kwargs)
    return await loop.run_in_executor(executor, func)
