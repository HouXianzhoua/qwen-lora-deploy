# tests/test_infer.py
from unittest.mock import MagicMock, patch
import pytest
import torch
from api.infer import safe_generate_response
from api.model_registry import registry
@pytest.fixture(autouse=True)
def setup_registry():
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    # 模拟 generate 输出
    mock_outputs = MagicMock()
    mock_outputs.__getitem__.return_value = torch.tensor([1, 2, 3, 4])
    mock_model.generate.return_value = mock_outputs
    mock_model.device = 'cuda'  # 模拟设备
    mock_tokenizer.decode.return_value = "模拟的生成结果"
    mock_inputs = {
        "input_ids": torch.tensor([[101, 200, 300]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }

    mock_tokenizer.return_value = mock_inputs  # 现在返回字典，但 infer.py 会用 .items()

    registry.set_model(mock_model, mock_tokenizer)
    yield
    registry.set_model(None, None)
def test_safe_generate_response_basic():
    prompt = "你好"
    result = safe_generate_response(prompt)
    assert result == "模拟的生成结果"

def test_safe_generate_response_with_kwargs():
    result = safe_generate_response(
        "世界",
        max_new_tokens=64,
        temperature=0.8,
        top_p=0.95
    )
    # 验证参数传递
    call_kwargs = registry.get_model().generate.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 64

def test_safe_generate_response_registry_not_loaded():
    registry.set_model(None, None)
    with pytest.raises(Exception):
        safe_generate_response("任何提示")
