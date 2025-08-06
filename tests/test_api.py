# tests/test_api.py
import requests

API_URL = "http://localhost:8000/predict"

def test_predict_endpoint():
    """要求本地服务已启动（docker compose up）"""
    payload = {
        "prompt": "测试一下模型能不能正常回复。",
        "max_new_tokens": 32
    }
    resp = requests.post(API_URL, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "output" in data
    assert isinstance(data["output"], str)
    assert len(data["output"]) > 0

