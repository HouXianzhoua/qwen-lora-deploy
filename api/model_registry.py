# api/model_registry.py
"""
模型注册器，确保全局唯一实例
"""

import torch


class ModelRegistry:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def set_model(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.is_loaded = True

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer


# 全局单例
registry = ModelRegistry()
