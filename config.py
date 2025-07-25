# config.py
import os
from pathlib import Path

# === 路径配置 ===
# 基础模型路径（挂载卷）
MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/models/Qwen2-0.5B-Instruct")

# LoRA 微调权重路径
LORA_PATH = os.getenv("LORA_PATH", "/app/finetune/output/final_model")

# 日志目录（可选）
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(exist_ok=True)

# === 推理参数默认值 ===
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", 128))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", 0.9))
DEFAULT_DO_SAMPLE = os.getenv("DEFAULT_DO_SAMPLE", "true").lower() == "true"

# === 服务配置 ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))  # Gunicorn workers
MAX_WORKERS_THREAD_POOL = int(os.getenv("MAX_WORKERS_THREAD_POOL", 2))  # 线程池大小

# === 其他 ===
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
