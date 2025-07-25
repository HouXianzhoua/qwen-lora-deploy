# api/main.py
import torch
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import DEBUG, WORKERS, API_PORT, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from api.infer import async_generate_response
from api.model_registry import registry

# ✅ 替换 print，使用 logger
from api.logger import logger  # <-- 新增

app = FastAPI(title="Qwen-LoRA API", debug=DEBUG)

@app.on_event("startup")
async def startup_event():
    from api.infer import init_model
    if not registry.is_loaded:
        # ✅ 替换 print
        logger.info("🚀 应用启动中，开始加载模型...")
        init_model()
        logger.info("✅ 模型加载完成")
    else:
        logger.info("✅ 模型已由 registry 加载，跳过初始化。")

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
        # ✅ 替换 print
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
        # ✅ 记录请求信息 (DEBUG 级别)
        logger.debug(f"收到推理请求: prompt='{request.prompt}', max_new_tokens={request.max_new_tokens}, temperature={request.temperature}, top_p={request.top_p}")
        
        output = await async_generate_response(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=request.do_sample
        )
        
        # ✅ 记录成功响应
        logger.info(f"推理成功，输出长度: {len(output)}")
        return {"output": output}
    except Exception as e:
        # ✅ 记录错误信息
        logger.error(f"生成失败: {str(e)}", exc_info=True) # exc_info=True 会记录完整的 traceback
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")
