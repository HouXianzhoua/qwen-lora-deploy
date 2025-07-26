# api/main.py
import torch
import os
import threading  # ←←← 新增：用于加锁
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import DEBUG, WORKERS, API_PORT, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from api.infer import async_generate_response
from api.model_registry import registry

# ✅ 替换 print，使用 logger
from api.logger import logger  # <-- 新增

# 🔒 新增：模块级锁，确保 init_model 只执行一次
_init_lock = threading.Lock()

app = FastAPI(title="Qwen-LoRA API", debug=DEBUG)


@app.on_event("startup")
async def startup_event():
    from api.infer import init_model

    # ✅ 第一层检查：registry 是否已加载
    if registry.is_loaded:
        logger.info("✅ 模型已由 registry 加载，跳过初始化。")
        return

    # 🔒 加锁，防止多个 worker 同时初始化
    with _init_lock:
        # ✅ 第二层检查：拿到锁后再次确认是否已加载（防止竞争）
        if registry.is_loaded:
            logger.info("✅ 模型已在其他 worker 中加载，当前 worker 跳过。")
            return

        # ✅ 此时只有 1 个 worker 能进入
        logger.info("🚀 应用启动中，开始加载模型...")
        init_model()
        logger.info("✅ 模型加载完成")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    try:
        gpu_available = torch.cuda.is_available()
        model = registry.get_model()
        tokenizer = registry.get_tokenizer()

        if model is not None and tokenizer is not None:
            device_info = str(model.device)
            model_loaded = True
        else:
            device_info = "model or tokenizer is None"
            model_loaded = False

        return {
            "status": "healthy",
            "gpu": gpu_available,
            "device": device_info,
            "model_loaded": model_loaded,
            "is_loaded_via_registry": registry.is_loaded
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "error", "message": f"Health check failed: {str(e)}"}


class InferenceRequest(BaseModel):
    prompt: str
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    do_sample: bool = True


@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        logger.debug(f"收到推理请求: prompt='{request.prompt}', max_new_tokens={request.max_new_tokens}, temperature={request.temperature}, top_p={request.top_p}")

        output = await async_generate_response(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        logger.info(f"推理成功，输出长度: {len(output)}")
        return {"output": output}
    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")
