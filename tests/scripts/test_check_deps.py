
import sys
import pytest
from unittest.mock import patch, MagicMock
from scripts.check_deps import pydub, funasr, AUDIO_DEPS_AVAILABLE

def test_pydub_import():
    assert pydub is not None
    assert hasattr(pydub, "__file__")

def test_funasr_import():
    assert funasr is not None
    assert hasattr(funasr, "__file__")

def test_audio_deps_available():
    assert AUDIO_DEPS_AVAILABLE is True
