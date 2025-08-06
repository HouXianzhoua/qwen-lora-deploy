# tests/test_model_registry.py
from api.model_registry import registry

def test_model_registry_set_and_get():
    mock_model = object()
    mock_tokenizer = object()

    registry.set_model(mock_model, mock_tokenizer)

    assert registry.get_model() is mock_model
    assert registry.get_tokenizer() is mock_tokenizer
    assert registry.is_loaded

