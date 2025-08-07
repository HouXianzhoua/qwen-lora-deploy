import time
import random
import requests

API_URL = "http://localhost:8000/predict"

PROMPTS = [
    "问题：Python 中列表和元组的区别是什么？\n回答：",
    "问题：帮我规划一个三天的北京旅行行程。\n回答：",
    "问题：请写一首关于春天的诗。\n回答：",
]

def send_request(prompt):
    payload = {
        "prompt": prompt,
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }
    start = time.time()
    response = requests.post(API_URL, json=payload)
    end = time.time()
    latency = (end - start) * 1000  # 毫秒
    return latency, response.json().get("output", "")

def main():
    results = []
    selected_prompts = random.sample(PROMPTS, 3)

    print("🌐 发送 3 个推理请求...\n")

    for i, prompt in enumerate(selected_prompts):
        latency, output = send_request(prompt)
        print(f"📝 请求 {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"延迟: {latency:.2f}ms")
        print(f"输出: {output[:60]}...\n")
        results.append((prompt, latency))
        print(f"输出长度: {len(output)}")
        print(f"完整输出内容: '{output}'\n")


if __name__ == "__main__":
    main()

