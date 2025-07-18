# ========================
# 1. 导入依赖
# ========================
import os
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ========================
# 2. 配置日志
# ========================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/infer.log"),
        logging.StreamHandler()
    ]
)

# ========================
# 3. 加载模型和 LoRA 权重
# ========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_MODEL_PATH = os.path.join(PROJECT_ROOT, "finetune", "output", "final_model")
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen2-0.5B-Instruct")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH, trust_remote_code=True)

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)

# 合并权重
model = model.merge_and_unload()

# 编译模型（如果支持）
if hasattr(torch, 'compile'):
    model = torch.compile(model)

# 设置为评估模式
model.eval()

# ========================
# 4. 推理函数
# ========================
def generate_response(instruction, max_new_tokens=128, temperature=0.7, top_p=0.9, do_sample=False):
    prompt = f"{instruction}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        end_time = time.time()

    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[:, input_length:]
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    latency = end_time - start_time

    logging.info(f"Instruction: {instruction}")
    logging.info(f"Response: {response}")
    logging.info(f"Time taken: {latency:.2f}s")

    return response, latency

def safe_generate_response(instruction, **kwargs):
    try:
        return generate_response(instruction, **kwargs)
    except Exception as e:
        logging.error(f"Error generating response for '{instruction}': {e}", exc_info=True)
        return "抱歉，生成过程中出现错误。", 0.0
# ========================
# 5. 测试入口
# ========================
if __name__ == "__main__":
    test_instructions = [
        "为什么创建《中国科学技术大学软件学院冒险者指南》?",
        "科软研究生有哪些常见学习与交流难题?",
        "什么是科软冒险者指南的核心理念?"
    ]

    results = []  # ✅ 在这里定义 results
    total_latency = 0
    num_runs = len(test_instructions)

    for instruction in test_instructions:
        response, latency = safe_generate_response(instruction)
        total_latency += latency

        # 添加结果到列表中
        results.append({
            "instruction": instruction,
            "response": response,
            "latency": f"{latency:.2f}s",
            "status": "success" if "抱歉" not in response else "failed"
        })


    avg_latency = total_latency / num_runs
    logging.info(f"✅ 平均推理延迟: {avg_latency:.2f} 秒")
    print(f"✅ 平均推理延迟: {avg_latency:.2f} 秒")

    # ✅ 保存为 JSON 文件
    import json
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
