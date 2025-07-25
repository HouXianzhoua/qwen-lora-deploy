import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_path = "/root/qwen-lora-deploy/models/Qwen2-0.5B-Instruct"

    print(f"🔍 正在从 {model_path} 加载模型与分词器...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.float16
    )

    print("✅ 模型与分词器加载成功！")

    # 简单测试生成
    prompt = "请简要介绍一下量子力学。"
    inputs = tokenizer(prompt, return_tensors="pt").to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(inputs.input_ids.device)

    print("🧠 模型生成中...")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("📢 模型输出：")
    print(response)


if __name__ == "__main__":
    main()
