# 使用 NVIDIA 官方 CUDA 镜像
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 换源（加速国内构建）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 更新包列表（忽略 NVIDIA 仓库错误）
RUN apt-get update || echo "APT update failed, continuing..."

# 安装依赖
RUN apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip3 install --upgrade pip

# =============================
# 🔥 关键：合并安装命令，使用镜像源和重试
# =============================
# 先复制 requirements.txt 以便利用 Docker 缓存
COPY requirements.txt .

# 合并安装 PyTorch + 项目依赖，使用清华源加速，并增加重试次数和超时时间
RUN pip install --no-cache-dir --retries 5 --timeout 300 \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1 \
   --index-url https://download.pytorch.org/whl/cu128   \
   --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple   \
   -r requirements.txt

# 复制项目文件
COPY . .

# 环境变量
ENV PYTHONPATH=/app \
    MODEL_PATH=/mnt/models/Qwen2-0.5B-Instruct \
    LORA_PATH=/app/finetune/output/final_model

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--workers=1", "api.main:app"]
