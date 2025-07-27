# 🚀 Qwen-LoRA 微调与推理服务封装项目

> 基于 Qwen2-0.5B-Instruct 的轻量级指令微调（LoRA）与 FastAPI 推理服务封装，支持一键训练、推理与 Docker 部署，适用于小样本定制化 AI 场景。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.0-orange)](https://huggingface.co/transformers)

---

## 📌 项目简介

本项目实现了对 **Qwen2-0.5B-Instruct** 模型的 **LoRA 指令微调**（r=16），并封装了完整的 **训练 → 推理 → 服务化 → 部署** 流程。适用于企业私有知识问答、轻量化 AI 接入、小样本指令学习等场景。

- ✅ **小样本微调**：使用 138 条自定义指令数据，训练仅需 **72 秒**。
- ✅ **低延迟推理**：本地推理延迟约 **1.5 秒**。
- ✅ **高性能服务**：基于 FastAPI + Uvicorn + Gunicorn 构建异步 API，支持并发请求。
- ✅ **Docker 一键部署**：支持本地与云端（如阿里云 GPU 实例）一键启动。
- ✅ **开源可扩展**：代码结构清晰，易于二次开发与集成。

---

## 🧰 技术栈

| 组件 | 说明 |
|------|------|
| **Qwen2-0.5B-Instruct** | 阿里通义千问轻量级大模型 |
| **LoRA 微调** | 使用 PEFT 框架进行低秩适配，r=16 |
| **Transformers** | Hugging Face 模型加载与推理 |
| **FastAPI** | 高性能异步 API 接口 |
| **Gunicorn + Uvicorn** | 多 Worker 生产级部署 |
| **Docker + NVIDIA GPU** | 容器化部署，支持 CUDA 加速 |
| **ThreadPoolExecutor** | 线程池控制推理并发 |
---

## 🔧 快速开始

### 1. 环境准备

```bash
# 推荐使用 conda 或 venv
conda create -n qwen python=3.10
conda activate qwen
# 安装依赖
pip install -r requirements.txt
```

### 2. 📥 下载基础模型（关键步骤！）

本项目依赖 `Qwen2-0.5B-Instruct` 作为基础模型。**该模型文件较大且需遵守许可协议，不会包含在本仓库中**。请按以下步骤自行下载：

#### 方法一：使用 Hugging Face (国际站)

```bash
# 安装 Git LFS
git lfs install

# 创建模型目录
mkdir -p models

# 克隆模型（需登录 Hugging Face 并同意协议）
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct models/Qwen2-0.5B-Instruct
```

> 🔗 访问 [https://huggingface.co/Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 并点击 "Agree and access repository"。

#### 方法二：使用 ModelScope (魔搭，国内推荐)

```bash
# 安装 modelscope
pip install modelscope

# 使用 Python 脚本下载
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('qwen/Qwen2-0.5B-Instruct', cache_dir='models')
```

> 🔗 访问 [https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct)

#### 方法三：使用 Hugging Face 镜像站

```bash
# 使用国内镜像加速
git clone https://hf-mirror.com/Qwen/Qwen2-0.5B-Instruct models/Qwen2-0.5B-Instruct
```

> 🔗 [https://hf-mirror.com](https://hf-mirror.com)

✅ **验证目录结构**：下载后，确保路径为 `models/Qwen2-0.5B-Instruct/`，且包含 `config.json`, `model.safetensors` 等文件。

---

### 3. 数据准备与LoRA 微调

---

## 📚 数据格式说明

本项目使用 JSONL (JSON Lines) 格式进行模型微调。`data/train.jsonl` 文件中，**每行是一个独立的 JSON 对象**，包含以下字段：

- `instruction`: str，任务指令（如“请总结以下文章”）
- `response`: str，期望模型输出的答案

### 示例（脱敏后）：

```jsonl
{"instruction": "转博前需要具备哪些科研成果?", "response": "通常需要一篇一作论文（B类或以上），因为课题组内部和外部申请竞争激烈，成果是决定性因素。"}
```

📁 **注意**：
- `data/` 目录已被 `.gitignore` 忽略，请自行创建并放入你的私有数据。
- 你可以通过修改 `lora_train.py` 中的 `dataset` 加载路径来使用自定义数据集。

---

## 📚 微调
```bash
# 进入微调目录
cd finetune

# 执行训练（默认使用 data/train.jsonl）
python lora_train.py
```


---

### 4. 启动推理服务

#### 本地启动 (调试)

```bash
# 返回根目录
cd ..

# 直接运行（单 worker）
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 生产模式 (Gunicorn)

```bash
# 使用 Gunicorn 启动（多 worker）
gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers 2 api.main:app
```

#### Docker 一键部署

```bash
# 构建镜像
docker-compose build

# 启动服务（自动挂载模型与 LoRA 权重）
docker-compose up -d
```

服务将运行在 `http://localhost:8000`。

> 💡 支持 NVIDIA GPU 加速（需安装 `nvidia-docker`）。

---

## 🌐 API 接口说明

### 🔹 `GET /health` - 健康检查

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "gpu": true,
  "device": "cuda:0",
  "model_loaded": true
}
```

### 🔹 `POST /predict` - 文本生成

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "请写一首关于春天的诗。",
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

```json
{
  "output": "春风拂面花自开，柳绿桃红映山川..."
}
```

---

## ⚙️ 配置说明

通过 `.env` 文件或环境变量配置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `MODEL_PATH` | `/mnt/models/Qwen2-0.5B-Instruct` | 基础模型路径 |
| `LORA_PATH` | `/app/finetune/output/final_model` | LoRA 微调权重路径 |
| `API_PORT` | `8000` | 服务端口 |
| `DEFAULT_MAX_NEW_TOKENS` | `128` | 默认生成长度 |
| `DEFAULT_TEMPERATURE` | `0.7` | 温度参数 |
| `DEFAULT_TOP_P` | `0.9` | Top-p 采样 |
| `WORKERS` | `2` | Gunicorn Worker 数 |
| `DEBUG` | `false` | 是否开启调试模式 |

> 📌 **创建 `.env` 文件**：复制 `env.example` 并根据需要修改。

---

## 📈 性能表现

| 场景 | 指标 |
|------|------|
| **本地推理延迟** | ~1.5 秒（prompt + 生成） |
| **训练耗时** | ~72 秒（138 条数据，r=16） |
| **10 QPS 负载测试** | 平均响应 16 秒，P95 33 秒 |
| **显存占用** | ~1.8 GB（合并后模型） |

> 测试环境：NVIDIA 4070S GPU，32GB RAM，Ubuntu 22.04

---

## 🧪 测试与验证

```bash
# 单元测试
pytest tests/

# 集成测试
python test_api_v1.py
python test_api_v2.py

# 压力测试（Locust）
locust -f locustfile.py --headless -u 10 -r 2 -t 5m
```

---

## 📚 部署到阿里云 GPU 实例

1. 购买阿里云 GPU 云服务器（如 ecs.gn6i-c4g1.xlarge）
2. 安装 Docker + NVIDIA Container Toolkit
3. 克隆本项目
4. 下载基础模型到 `models/` 目录
5. 执行 `docker-compose up -d`
6. 开放 8000 端口，即可通过公网访问 API

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE)，欢迎 Fork、Star 与贡献！

---

## 🙌 贡献指南

欢迎提交 Issue 或 Pull Request，包括：
- 新功能建议
- Bug 修复
- 性能优化
- 文档改进

请遵守 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 📬 联系方式

如有问题，欢迎通过 GitHub Issue 联系，或发送邮件至：`2683180773@qq.com`

---

## 🌟 致谢

- [Hugging Face](https://huggingface.co) - 提供强大的模型生态
- [Qwen Team](https://qwenlm.github.io/) - 开源 Qwen 系列模型
- [PEFT](https://github.com/huggingface/peft) - LoRA 实现框架

> ✨ 用小模型，做大事。
```

