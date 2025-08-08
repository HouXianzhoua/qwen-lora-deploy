# api/model_registry.py
"""
模型注册器，确保进程内全局唯一实例
（注意：对 Gunicorn 多进程无效；如需全局唯一需进程间协调）
"""
from typing import Optional
import torch  # 可能在其他地方用到，这里保留无伤

class ModelRegistry:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def set_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # 只有在两者都就位时才标记 loaded
        self.is_loaded = (model is not None and tokenizer is not None)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

# 进程内全局单例
registry = ModelRegistry()

