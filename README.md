# 🚀 Qwen-LoRA 微调与高性能推理服务

> 基于 Qwen2-0.5B-Instruct 的轻量级 LoRA 指令微调与高并发推理服务封装，支持一键训练、推理与 Docker 部署，适用于小样本定制化 AI 场景。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.0-orange)](https://huggingface.co/transformers)

---

## 📌 项目亮点

- **分钟级 LoRA 微调**  
  使用 138 条指令数据（r=16），72 秒完成微调并生成可用模型。
- **高并发推理优化**  
  结合 FastAPI 异步框架、线程池、请求批量合并（Batching），支持高吞吐推理。
- **吞吐提升显著**  
  Locust 压测下，**100 并发用户 / 2s 请求间隔**，QPS 提升至 **25+**，P99 延迟 **3.1 秒**。
- **容器化 GPU 部署**  
  基于 Docker + NVIDIA Container Toolkit，一键部署到本地/云端 GPU 环境。
- **模块化与可扩展性**  
  推理参数、路径、批量配置等可通过 `.env` 动态调整，无需改代码。

---

## 🧰 技术栈

| 组件 | 说明 |
|------|------|
| **Qwen2-0.5B-Instruct** | 阿里通义千问轻量级大模型 |
| **LoRA 微调** | 使用 PEFT 框架进行低秩适配，r=16 |
| **Transformers** | Hugging Face 模型加载与推理 |
| **FastAPI** | 高性能异步 API 接口 |
| **Gunicorn + Uvicorn** | 多 Worker 生产级部署 |
| **Batching + ThreadPoolExecutor** | 批量合并请求 + 线程池推理 |
| **Docker + NVIDIA GPU** | 容器化部署，支持 CUDA 加速 |

---

## 🔧 快速开始

### 1. 环境准备
```bash
conda create -n qwen python=3.10
conda activate qwen
pip install -r requirements.txt
````

### 2. 下载基础模型

本项目不包含模型，请手动下载 Qwen2-0.5B-Instruct 到 `models/`：

```bash
git lfs install
mkdir -p models
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct models/Qwen2-0.5B-Instruct
```

或参考 ModelScope / Hugging Face 镜像站下载方式。

---

### 3. 数据准备与 LoRA 微调

训练数据格式（JSONL，每行一条）：

```json
{"instruction": "转博前需要具备哪些科研成果?", "response": "通常需要一篇一作论文..."}
```

运行微调：

```bash
cd finetune
python lora_train.py
```

---

### 4. 启动推理服务

**本地调试（单 worker）**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**生产模式（推荐）**

```bash
docker-compose build
docker-compose up -d
```

服务地址：`http://localhost:8000`

---

## 🌐 API 说明

**健康检查**

```bash
curl http://localhost:8000/health
```

**文本生成**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt":"请写一首关于春天的诗。"}'
```

---

## 📊 性能指标

| 场景                   | QPS    | P99 延迟 |
| -------------------- | ------ | ------ |
| **100 用户 / 2s 请求间隔** | 25+    | 3.1s   |
| **本地单条推理延迟**         | \~1.2s | -      |

> 测试环境：NVIDIA 4070S GPU，32GB RAM，Ubuntu 22.04
> 优化策略：**线程池 + 批量合并（BATCH\_MAX\_SIZE=32）**

---

## 🧪 测试

单元测试：

```bash
pytest tests/
```

压测（Web UI）：

```bash
locust -f locustfile.py
```

---

## 📄 协议

MIT License

---

## 🌟 致谢

* [Qwen Team](https://qwenlm.github.io/)
* [Hugging Face](https://huggingface.co)
* [PEFT](https://github.com/huggingface/peft)

