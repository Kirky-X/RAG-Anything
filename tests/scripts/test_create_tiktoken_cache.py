
import pytest
from unittest.mock import patch, MagicMock
import os
from scripts import create_tiktoken_cache

def test_tiktoken_cache_logic():
    # Since the script runs on import, it's hard to test directly without side effects
    # But we can check if the environment variable was set
    assert "TIKTOKEN_CACHE_DIR" in os.environ
    assert os.path.exists(os.environ["TIKTOKEN_CACHE_DIR"])
