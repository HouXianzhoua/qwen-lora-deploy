from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import default_data_collator
import torch
import os

# 设置环境变量减少显存碎片（重要！）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# 1. 加载模型与 Tokenizer
# -----------------------------
model_name = "../models/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果 tokenizer 缺少 eos_token 或 pad_token，手动添加
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # 确保 token 数量一致

# -----------------------------
# 2. 配置 LoRA 微调
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
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
    texts = [f"{inst}\n{resp}" for inst, resp in zip(examples["instruction"], examples["response"])]

    tokenized = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors=None
    )

    # 添加 -100 mask 忽略 padding 部分
    labels = []
    for input_ids, attn_mask in zip(tokenized["input_ids"], tokenized["attention_mask"]):
        label = [(token_id if mask == 1 else -100) for token_id, mask in zip(input_ids, attn_mask)]
        labels.append(label)
    tokenized["labels"] = labels

    return tokenized

tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["instruction", "response"]
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
    num_train_epochs=3,
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
attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long).to(model.device)
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
# 8. 保存模型
# -----------------------------
model.save_pretrained("./output/final_model")
tokenizer.save_pretrained("./output/final_model")
