# tests/test_model_registry.py
from api.model_registry import registry

def test_model_registry_singleton():
    """测试 ModelRegistry 是单例模式。"""
    # 获取 registry 实例
    reg1 = registry
    reg2 = registry  # 再次获取

    # 它们应该是同一个对象
    assert reg1 is reg2

def test_model_registry_set_and_get():
    """测试可以正确设置和获取模型与分词器。"""
    # 创建模拟对象
    mock_model = object()
    mock_tokenizer = object()

    # 设置
    registry.set_model(mock_model, mock_tokenizer)

    # 获取并验证
    assert registry.get_model() is mock_model
    assert registry.get_tokenizer() is mock_tokenizer
    assert registry.is_loaded is True
