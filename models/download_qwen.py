from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-0.5B-Instruct"

# 下载并加载 tokenizer 和 model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 保存到本地（可选）
save_path = "./Qwen2-0.5B-Instruct-local"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
