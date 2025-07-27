import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, default_data_collator)

# 设置环境变量减少显存碎片（重要！）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# 1. 加载模型与 Tokenizer
# -----------------------------
model_name = "../models/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果 tokenizer 缺少 eos_token 或 pad_token，手动添加
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # 确保 token 数量一致

# -----------------------------
# 2. 配置 LoRA 微调
# -----------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

model.enable_input_require_grads()
model.print_trainable_parameters()

# 设置模型配置
model.config.use_cache = False
model.config.pretraining_tp = 1  # 防止某些模型结构出错

# -----------------------------
# 3. 加载并处理训练数据
# -----------------------------
dataset = load_dataset("json", data_files={"train": "../data/train.jsonl"})["train"]

print("First raw sample:")
print(dataset[0])


def tokenize_function(examples):
    instructions = examples["instruction"]
    responses = examples["response"]
    texts = []
    labels_list = []

    for inst, resp in zip(instructions, responses):
        # 构建完整文本
        text = f"问题：{inst}\n回答：{resp}"
        texts.append(text)

    # ✅ 关键：只 tokenize 一次，并保留 attention_mask
    tokenized = tokenizer(
        texts,
        padding=False,  # 先不 padding，后面再处理
        truncation=True,
        max_length=512,
        return_tensors=None,
        add_special_tokens=True,  # 确保添加 BOS/EOS
        return_attention_mask=True,
    )

    # 构建 labels：遍历每条数据，将“问题”部分设为 -100
    labels_list = []
    for i, (inst, resp) in enumerate(zip(instructions, responses)):
        # ✅ 在完整 tokenized 结果中，计算“问题”部分的长度
        prompt_text = f"问题：{inst}\n回答："
        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=True,  # 与上面保持一致
            truncation=True,
            max_length=512,
            return_tensors=None,
        )
        input_len = len(prompt_tokens["input_ids"])  # 这是包含 BOS 的长度

        # 获取完整 input_ids
        full_input_ids = tokenized["input_ids"][i]

        # 构建 label：前面 -100，后面保留
        label = [-100] * input_len + full_input_ids[input_len:]
        labels_list.append(label)

    # ✅ 现在统一 padding
    max_length = max(len(x) for x in tokenized["input_ids"])
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for i in range(len(tokenized["input_ids"])):
        ids = tokenized["input_ids"][i]
        mask = tokenized["attention_mask"][i]
        label = labels_list[i]

        # Padding 到 max_length
        pad_len = max_length - len(ids)
        padded_input_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        padded_attention_mask.append(mask + [0] * pad_len)
        padded_labels.append(label + [-100] * pad_len)

    tokenized["input_ids"] = padded_input_ids
    tokenized["attention_mask"] = padded_attention_mask
    tokenized["labels"] = padded_labels
    return tokenized


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["instruction", "response"]
)

print("\nTokenized sample keys:", tokenized_datasets[0].keys())
print("First tokenized sample (raw):")
print(tokenized_datasets[0])

for i in range(min(3, len(tokenized_datasets))):
    print(f"\n--- Sample {i} ---")
    decoded_text = tokenizer.decode(tokenized_datasets[i]["input_ids"])
    print("Decoded text:\n", decoded_text)
    print("Length of input_ids:", len(tokenized_datasets[i]["input_ids"]))

# -----------------------------
# 4. 设置 Data Collator
# -----------------------------
data_collator = default_data_collator

# -----------------------------
# 5. 训练参数配置
# -----------------------------
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    save_steps=10,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    report_to="none",
    remove_unused_columns=False,
    ignore_data_skip=True,
    disable_tqdm=False,
    label_names=["labels"],
)

# -----------------------------
# 6. 初始化 Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# -----------------------------
# [调试] 检查 loss 是否能反向传播
# -----------------------------
print("\n✅ [DEBUG] Checking model forward pass and loss computation...")
print("Model device:", next(model.parameters()).device)
sample = tokenized_datasets[0]
input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(model.device)
attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long).to(
    model.device
)
labels = torch.tensor([sample["labels"]], dtype=torch.long).to(model.device)
print("Labels:", labels)
print("Unique labels:", torch.unique(labels))
model.train()

# 不用 autocast，直接前向
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
print(f"✅ [DEBUG] Loss = {loss.item()}")

loss.backward()

for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"✅ [DEBUG] Gradient computed for: {name}")
        break
else:
    print("❌ [DEBUG] No gradient was computed! Check input / loss.")

# -----------------------------
# 7. 开始训练
# -----------------------------
trainer.train()
# -----------------------------
# 8. 保存模型和 tokenizer
# -----------------------------
model.save_pretrained("./output/final_model")

# ✅ 关键：从 base model 路径重新加载干净的 tokenizer 再保存
clean_tokenizer = AutoTokenizer.from_pretrained(
    "../models/Qwen2-0.5B-Instruct", trust_remote_code=True
)
# 如果你在训练时加了 pad_token，这里也要加
if clean_tokenizer.pad_token is None:
    clean_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
clean_tokenizer.save_pretrained("./output/final_model")
