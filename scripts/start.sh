#!/bin/bash
# scripts/start.sh

# 获取环境变量 WORKERS，如果未设置则默认为 1
WORKERS=${WORKERS:-1}

echo "🚀 启动 Gunicorn，使用 ${WORKERS} 个 workers..."
echo "💡 使用 --preload 模式以共享模型内存..."

# 启动服务：关键是添加 --preload
exec gunicorn -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --workers ${WORKERS} \
    --worker-connections 1000 \
    --timeout 600 \
    --keep-alive 5 \
    api.main:app
