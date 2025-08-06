import os
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# 让 CUDA 显存分配更平滑（尤其是小显存）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    BASE_DIR = Path(__file__).resolve().parent         # finetune/
    ROOT_DIR = BASE_DIR.parent                         # 项目根目录

    parser = argparse.ArgumentParser("Qwen LoRA finetune")
    # 路径
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=str((ROOT_DIR / "models" / "Qwen2-0.5B-Instruct").resolve()),
        help="基座模型本地目录（包含 config.json / model.safetensors 等）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str((ROOT_DIR / "data" / "train.jsonl").resolve()),
        help="训练数据 jsonl 路径（需要包含 instruction / response 字段）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str((Path(__file__).resolve().parent / "output" / "final_model").resolve()),
        help="LoRA 适配器与 tokenizer 输出目录",
    )

    # 数据与长度
    parser.add_argument("--max_length", type=int, default=512, help="每条样本的最大 token 长度")

    # LoRA 超参（默认保持与你原脚本一致）
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="逗号分隔的模块名列表",
    )

    # 训练超参（默认与你原脚本一致）
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--logging_steps", type=int, default=10)

    # 精度/设备
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--use_gpu", action="store_true", default=torch.cuda.is_available())

    # 调试开关
    parser.add_argument("--debug_forward", action="store_true", default=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    BASE_MODEL_PATH = Path(args.base_model_path)
    DATA_PATH = Path(args.data_path)
    OUTPUT_DIR = Path(args.output_dir)

    assert BASE_MODEL_PATH.exists(), f"Base model not found: {BASE_MODEL_PATH}"
    assert DATA_PATH.exists(), f"Training data not found: {DATA_PATH}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Paths]\n  base_model_path = {BASE_MODEL_PATH}\n  data_path       = {DATA_PATH}\n  output_dir      = {OUTPUT_DIR}")

    # -----------------------------
    # 1) 加载模型与 Tokenizer
    # -----------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH))

    # 若 tokenizer 缺少 pad_token，补齐
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 设备/精度设置
    torch_dtype = None
    if args.fp16:
        torch_dtype = torch.float16
    elif args.bf16:
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        torch_dtype=torch_dtype,
        device_map="auto" if args.use_gpu else None,
    )

    # 若新增了 pad_token，需要调整 embedding 大小
    model.resize_token_embeddings(len(tokenizer))

    # -----------------------------
    # 2) 配置 LoRA
    # -----------------------------
    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # 一些模型兼容性设置
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # -----------------------------
    # 3) 加载与处理训练数据
    # -----------------------------
    dataset = load_dataset("json", data_files={"train": str(DATA_PATH)})["train"]
    print("First raw sample:")
    print(dataset[0])

    max_length = int(args.max_length)

    def tokenize_function(examples):
        instructions = examples["instruction"]
        responses = examples["response"]
        texts = []

        for inst, resp in zip(instructions, responses):
            text = f"问题：{inst}\n回答：{resp}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
            add_special_tokens=True,
            return_attention_mask=True,
        )

        # 构建 labels：将“问题”部分 mask 为 -100
        labels_list = []
        for i, (inst, _) in enumerate(zip(instructions, responses)):
            prompt_text = f"问题：{inst}\n回答："
            prompt_tokens = tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )
            input_len = len(prompt_tokens["input_ids"])

            full_input_ids = tokenized["input_ids"][i]
            label = [-100] * input_len + full_input_ids[input_len:]
            labels_list.append(label)

        # 手动 pad 到该 batch 的最大长度
        cur_max_len = max(len(x) for x in tokenized["input_ids"])
        padded_input_ids, padded_attention_mask, padded_labels = [], [], []
        for i in range(len(tokenized["input_ids"])):
            ids = tokenized["input_ids"][i]
            mask = tokenized["attention_mask"][i]
            label = labels_list[i]
            pad_len = cur_max_len - len(ids)

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
    # 4) Data Collator
    # -----------------------------
    data_collator = default_data_collator

    # -----------------------------
    # 5) 训练参数
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR.parent),  # 保存日志/检查点到 finetune/output/
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        logging_dir=str(OUTPUT_DIR.parent / "logs"),
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
        ignore_data_skip=True,
        disable_tqdm=False,
        label_names=["labels"],
    )

    # -----------------------------
    # 6) Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # -----------------------------
    # 7) 调试前向（可选）
    # -----------------------------
    if args.debug_forward:
        print("\n✅ [DEBUG] Checking model forward pass and loss computation...")
        device_str = next(model.parameters()).device
        print("Model device:", device_str)
        sample = tokenized_datasets[0]
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device_str)
        attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long).to(device_str)
        labels = torch.tensor([sample["labels"]], dtype=torch.long).to(device_str)
        print("Labels unique values:", torch.unique(labels))
        model.train()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"✅ [DEBUG] Loss = {loss.item()}")
        loss.backward()
        grad_ok = any(
            (p.requires_grad and p.grad is not None)
            for _, p in model.named_parameters()
        )
        print("✅ [DEBUG] Gradient computed." if grad_ok else "❌ [DEBUG] No gradient was computed!")

    # -----------------------------
    # 8) 训练
    # -----------------------------
    trainer.train()

    # -----------------------------
    # 9) 保存 LoRA 适配器与 tokenizer
    # -----------------------------
    model.save_pretrained(str(OUTPUT_DIR))

    # 用“干净”的 tokenizer 保存（与基座一致，再补 pad_token）
    clean_tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_PATH), trust_remote_code=True)
    if clean_tokenizer.pad_token is None:
        clean_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    clean_tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"\n✅ Done. LoRA adapter and tokenizer saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

