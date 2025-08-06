# tests/test_config.py
import os
import importlib
from unittest.mock import patch

def test_config_defaults():
    with patch.dict(os.environ, {}, clear=True):
        import config
        importlib.reload(config)
        assert config.DEFAULT_MAX_NEW_TOKENS == 128
        assert config.DEFAULT_TEMPERATURE == 0.7
        assert config.DEFAULT_TOP_P == 0.9
        assert config.DEFAULT_DO_SAMPLE is True

def test_config_from_env():
    with patch.dict(os.environ, {
        "DEFAULT_MAX_NEW_TOKENS": "256",
        "DEFAULT_TEMPERATURE": "0.5",
        "DEFAULT_TOP_P": "0.8",
        "DEFAULT_DO_SAMPLE": "false",
    }):
        import config
        importlib.reload(config)
        assert config.DEFAULT_MAX_NEW_TOKENS == 256
        assert config.DEFAULT_TEMPERATURE == 0.5
        assert config.DEFAULT_TOP_P == 0.8
        assert config.DEFAULT_DO_SAMPLE is False

