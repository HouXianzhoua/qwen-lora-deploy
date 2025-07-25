#!/bin/bash

# 检查模型目录是否存在
if [ ! -d "${MODEL_PATH}/Qwen2-0.5B-Instruct" ]; then
    echo "Error: Model not found at ${MODEL_PATH}"
    exit 1
fi

# 启动API服务（根据实际需求选择）
cd /app && \
python api/main.py \
    --model_path ${MODEL_PATH}/Qwen2-0.5B-Instruct \
    --lora_path /app/finetune/output  # 如果有LoRA权重
