#!/usr/bin/env python3
# test_api.py

import time

import requests

API_URL = "http://localhost:8000/predict"


def test_predict():
    payload = {
        "prompt": "为什么创建《中国科学技术大学软件学院冒险者指南》?",
        "max_new_tokens": 128,
    }

    print("🚀 正在发送请求...")
    print(f"📝 Prompt: {payload['prompt']}")
    print("-" * 60)

    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        end_time = time.time()
        duration = end_time - start_time

        print("✅ 请求成功！")
        print(f"⏱️  耗时: {duration:.2f} 秒")
        print("💬 回复:")
        print(f"\033[1;36m{result.get('output', '无输出')}\033[0m")  # 蓝色高亮

    except requests.exceptions.ConnectionError:
        print("❌ 连接失败，请检查服务是否运行：docker-compose ps")
    except requests.exceptions.Timeout:
        print("⏰ 请求超时，请检查模型加载是否卡住")
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败: {e}")


if __name__ == "__main__":
    test_predict()
