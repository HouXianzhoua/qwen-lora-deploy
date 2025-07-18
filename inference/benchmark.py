import time
from infer import build_prompt, generate_response

def benchmark():
    test_prompts = [
        "为什么创建《中国科学技术大学软件学院冒险者指南》?",
        "科软研究生有哪些常见学习与交流难题?",
        "什么是科软冒险者指南的核心理念?"
    ]
    for prompt in test_prompts:
        full_prompt = build_prompt(prompt)
        start_time = time.time()
        response = generate_response(full_prompt)
        end_time = time.time()
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Time taken: {end_time - start_time:.2f}s\n")

if __name__ == "__main__":
    benchmark()

