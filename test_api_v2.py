# test_api.py
import requests
import time
import json

API_URL = "http://localhost:8000/predict"
HEADERS = {"Content-Type": "application/json"}

def test_predict(prompt: str):
    print("🚀 正在发送请求...")
    print(f"📝 Prompt: {prompt}")
    print("-" * 60)

    data = {
        "prompt": prompt,
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }

    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()["output"]
            print(f"✅ 请求成功！")
            print(f"⏱️  耗时: {end_time - start_time:.2f} 秒")
            print(f"💬 回复:\n{result}")
        else:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 请求异常: {e}")

if __name__ == "__main__":
    # 测试用例 1：关于《冒险者指南》
    test_predict("为什么创建《中国科学技术大学软件学院冒险者指南》?")

    print("\n")  # 分隔符

    # 测试用例 2：通用问题（验证通用能力）
    test_predict("请简要介绍中国科学技术大学软件学院的特色。")

    print("\n")

    # 测试用例 3：指令遵循（验证 LoRA 可能的风格）
    test_predict("用轻松幽默的语气解释什么是机器学习。")
