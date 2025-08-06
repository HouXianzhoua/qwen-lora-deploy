# 🚀 Qwen-LoRA 微调与推理服务封装项目

> 基于 Qwen2-0.5B-Instruct 的轻量级指令微调（LoRA）与 FastAPI 推理服务封装，支持一键训练、推理与 Docker 部署，适用于小样本定制化 AI 场景。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/Transformers-4.40.0-orange)](https://huggingface.co/transformers)

---

## 📌 项目简介

本项目实现了对 **Qwen2-0.5B-Instruct** 模型的 **LoRA 指令微调**（r=16），并封装了完整的 **训练 → 推理 → 服务化 → 部署** 流程。适用于企业私有知识问答、轻量化 AI 接入、小样本指令学习等场景。

* ✅ **小样本微调**：使用 138 条自定义指令数据，训练仅需 **72 秒**。
* ✅ **低延迟推理**：本地单请求推理延迟约 **1.5 秒**。
* ✅ **高性能服务**：异步推理 + 请求批量合并，Locust 压测在 100 并发用户、2 秒请求间隔下达到 **25+ QPS**，P99 延迟 **3.1 秒**。
* ✅ **Docker 一键部署**：支持本地与云端（如阿里云 GPU 实例）一键启动。
* ✅ **开源可扩展**：代码结构清晰，易于二次开发与集成。

---

## 🧰 技术栈

| 组件                      | 说明                    |
| ----------------------- | --------------------- |
| **Qwen2-0.5B-Instruct** | 阿里通义千问轻量级大模型          |
| **LoRA 微调**             | 使用 PEFT 框架进行低秩适配，r=16 |
| **Transformers**        | Hugging Face 模型加载与推理  |
| **FastAPI**             | 高性能异步 API 接口          |
| **Gunicorn + Uvicorn**  | 多 Worker 生产级部署        |
| **Docker + NVIDIA GPU** | 容器化部署，支持 CUDA 加速      |
| **批量推理优化**              | 线程池 + 请求批量合并，显著提升吞吐   |
| **Locust**              | 接口性能压测                |

---

## 🔧 快速开始

### 1. 环境准备

```bash
conda create -n qwen python=3.10
conda activate qwen
pip install -r requirements.txt
```

### 2. 📥 下载基础模型

> **模型文件较大且需遵守许可协议，不会包含在本仓库中**

```bash
mkdir -p models
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct models/Qwen2-0.5B-Instruct
```

或使用 [ModelScope](https://modelscope.cn/models/qwen/Qwen2-0.5B-Instruct) / Hugging Face 镜像。

---

## ⚙️ 配置说明

所有部署参数可通过 `.env` 文件或环境变量修改，无需改代码：

| 环境变量                     | 默认值                                | 说明                               |
| ------------------------ | ---------------------------------- | -------------------------------- |
| `MODEL_PATH`             | `/mnt/models/Qwen2-0.5B-Instruct`  | 基础模型路径                           |
| `LORA_PATH`              | `/app/finetune/output/final_model` | LoRA 微调权重路径                      |
| `API_PORT`               | `8000`                             | 服务端口                             |
| `DEFAULT_MAX_NEW_TOKENS` | `128`                              | 默认生成长度                           |
| `DEFAULT_TEMPERATURE`    | `0.7`                              | 温度参数                             |
| `DEFAULT_TOP_P`          | `0.9`                              | Top-p 采样                         |
| `WORKERS`                | `2`                                | Gunicorn Worker 数（GPU 场景通常 1 即可） |
| `DEBUG`                  | `false`                            | 是否开启调试模式                         |
| `BATCH_MAX_SIZE`         | `32`                               | 批量推理最大合并请求数                      |
| `BATCH_INTERVAL_MS`      | `40`                               | 批量推理收集窗口（毫秒），越大合并率越高但延迟可能增加      |

> 复制 `.env.example` 并修改：

```bash
cp .env.example .env
```

---
## 🎯 LoRA 微调步骤

本项目使用 LoRA（Low-Rank Adaptation）对 **Qwen2-0.5B-Instruct** 进行轻量化指令微调，支持在极少数据和极短时间内完成模型定制。

### 1️⃣ 数据准备

- 数据格式为 **JSONL**（每行一个 JSON 对象），字段说明：
  - `instruction`: str，任务指令（如“请总结以下文章”）
  - `response`: str，期望的输出内容
- 训练数据放在 `data/train.jsonl`（该目录已被 `.gitignore` 忽略，可自由替换为自己的数据集）。

**示例（脱敏）**：
```jsonl
{"instruction": "转博前需要具备哪些科研成果?", "response": "通常需要一篇一作论文（B类或以上），因为课题组内部和外部申请竞争激烈，成果是决定性因素。"}
````

### 2️⃣ 启动微调

进入微调目录：

```bash
cd finetune
```

执行训练（默认使用 `data/train.jsonl`）：

```bash
python lora_train.py
```

> 💡 **训练耗时参考**：138 条数据，NVIDIA 4070S，约 **72 秒** 完成。

### 3️⃣ 微调结果

* 最终 LoRA 权重保存在：

```
finetune/output/final_model/
```
---

## 🚀 启动推理服务

### 本地启动（调试）

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 生产模式（Gunicorn）

```bash
WORKERS=1 gunicorn -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 api.main:app
```

### Docker 一键部署

```bash
docker-compose build
docker-compose up -d
```

---

## 🌐 API 接口

* **GET /health** - 健康检查
* **POST /predict** - 文本生成

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt":"请写一首关于春天的诗。"}'
```

---

## 📊 性能指标

| 指标                         | 数值                   |
| -------------------------- | -------------------- |
| 本地单请求推理延迟                  | \~1.5s               |
| Locust 压测（100 并发用户, 2s 间隔） | QPS 25+, P99 延迟 3.1s |
| 显存占用                       | \~1.8 GB（合并后模型）      |

测试环境：NVIDIA 4070S GPU, 32GB RAM, Ubuntu 22.04

---

## 🧪 测试与验证

```bash
pytest tests/
locust -f locustfile.py
```

---

## 📄 开源协议

本项目使用 [MIT License](LICENSE)


