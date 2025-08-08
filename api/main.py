# api/main.py
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from config import (
    DEBUG,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)
from api.model_registry import registry
from inference.model_service import async_generate_text, init_and_register_for_api
from api.logger import logger

app = FastAPI(title="Qwen-LoRA API", debug=DEBUG)

@app.on_event("startup")
async def startup_event():
    # 交给 model_service 做幂等，避免多次初始化
    try:
        init_and_register_for_api()
        logger.info("✅ 模型加载完成（幂等保护）。")
    except Exception as e:
        logger.error(f"🚨 启动加载模型失败: {e}", exc_info=True)
        raise

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
        model_loaded = model is not None and tokenizer is not None

        # 更稳的设备信息获取
        device_attr = getattr(model, "device", None) if model is not None else None
        device_info = getattr(device_attr, "type", str(device_attr)) if device_attr is not None else "uninitialized"

        return {
            "status": "healthy" if model_loaded else "initializing",
            "gpu": gpu_available,
            "device": device_info,
            "model_loaded": model_loaded,
            "is_loaded_via_registry": registry.is_loaded,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Health check failed: {str(e)}"}

class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="输入提示，不能为空")
    max_new_tokens: int = Field(DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(DEFAULT_TOP_P, gt=0.0, le=1.0)
    do_sample: bool = True

@app.post("/predict")
async def predict(request: InferenceRequest):
    try:
        logger.debug(
            f"收到推理请求: max_new_tokens={request.max_new_tokens}, "
            f"temperature={request.temperature}, top_p={request.top_p}, do_sample={request.do_sample}"
        )

        if not registry.is_loaded:
            raise HTTPException(status_code=503, detail="模型尚未初始化，请稍后重试。")

        output = await async_generate_text(
            request.prompt.strip(),
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )

        logger.info(f"推理成功，输出长度: {len(output)}")
        return {"output": output}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

