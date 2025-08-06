# locustfile.py
import random
from locust import HttpUser, task, between

PROMPTS = [
    "请写一首关于春天的诗。",
    "解释什么是机器学习。",
    "帮我规划一个三天的北京旅行行程。",
    "Python 中列表和元组的区别是什么？",
    "请用英文介绍你自己。",
]

class QwenUser(HttpUser):
    host = "http://localhost:8000"  # 直接写死本地 API 地址
    wait_time = between(1, 3)

    @task
    def predict(self):
        request_body = {
            "prompt": random.choice(PROMPTS),
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        with self.client.post("/predict", json=request_body, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"返回状态码 {resp.status_code}: {resp.text}")
            else:
                try:
                    result = resp.json()
                    if "output" not in result:
                        resp.failure("响应缺少 'output' 字段")
                except Exception as e:
                    resp.failure(f"解析 JSON 失败: {e}")

