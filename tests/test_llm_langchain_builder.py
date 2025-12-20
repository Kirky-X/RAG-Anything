import asyncio
import os
import sys
from pathlib import Path

import pytest
import toml
from pydantic import BaseModel

from raganything.llm import LLMProviderConfig, build_llm


# Load real config
def load_config():
    """Load configuration from config.toml"""
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        return toml.load(config_path)
    return {}


# Check if langchain libraries are available
def check_langchain_available():
    """Check if required langchain libraries are installed"""
    try:
        import langchain_openai
        import langchain_ollama
        import langchain_core.messages
        return True
    except ImportError:
        return False


# Skip tests if langchain libraries are not available and no API keys are configured
config = load_config()
LANGCHAIN_AVAILABLE = check_langchain_available()

# Check specific provider configurations
HAS_OPENAI_CONFIG = (
    config.get("OPENAI_API_KEY") and 
    config.get("OPENAI_API_KEY") != "your_openai_api_key_here"
)
HAS_OLLAMA_CONFIG = (
    config.get("OLLAMA_BASE_URL") and 
    config.get("OLLAMA_BASE_URL") != "http://localhost:11434"
)
HAS_OPENROUTER_CONFIG = HAS_OPENAI_CONFIG  # OpenRouter uses OpenAI key

# General skip condition
HAS_VALID_CONFIG = LANGCHAIN_AVAILABLE and (HAS_OPENAI_CONFIG or HAS_OLLAMA_CONFIG)

skip_reason = "LangChain libraries not available or no valid API configuration"


@pytest.mark.skipif(not HAS_OPENAI_CONFIG or not LANGCHAIN_AVAILABLE, reason="OpenAI API key not configured")
def test_build_openai_and_call_text():
    """Test building OpenAI LLM with real configuration"""
    cfg = LLMProviderConfig(
        provider="openai", 
        model=config.get("LLM_MODEL", "gpt-4o-mini"), 
        api_key=config.get("OPENAI_API_KEY")
    )
    
    with build_llm(cfg) as llm:
        # Use a simple test prompt
        out = asyncio.run(llm("Say 'test successful' in exactly 2 words"))
        assert len(out) > 0  # Basic check that we got a response
        print(f"OpenAI response: {out}")


@pytest.mark.skipif(not HAS_OPENROUTER_CONFIG or not LANGCHAIN_AVAILABLE, reason="OpenRouter configuration not available")
def test_build_openrouter_and_call_messages():
    """Test building OpenRouter LLM with messages format"""
    cfg = LLMProviderConfig(
        provider="openrouter",
        model=config.get("LLM_MODEL", "openrouter/model"),
        api_base="https://openrouter.ai/api/v1",
        api_key=config.get("OPENAI_API_KEY")  # Reuse OpenAI key for OpenRouter
    )
    
    with build_llm(cfg) as llm:
        msgs = [
            {"role": "system", "content": "You are a helpful assistant. Reply with exactly 3 words."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        
        out = asyncio.run(llm("", messages=msgs))
        assert len(out) > 0
        print(f"OpenRouter response: {out}")


@pytest.mark.skipif(not HAS_OLLAMA_CONFIG or not LANGCHAIN_AVAILABLE, reason="Ollama configuration not available")
def test_build_ollama_and_call_text():
    """Test building Ollama LLM with real configuration"""
    ollama_url = config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    cfg = LLMProviderConfig(
        provider="ollama", 
        model=config.get("LLM_MODEL", "qwen3:1.7b"), 
        api_base=ollama_url
    )
    
    with build_llm(cfg) as llm:
        out = asyncio.run(llm("Reply with exactly 2 words: test"))
        assert len(out) > 0
        print(f"Ollama response: {out}")


def test_config_loading():
    """Test that configuration loads correctly"""
    assert isinstance(config, dict)
    print(f"Loaded config keys: {list(config.keys())}")
    print(f"LLM_PROVIDER: {config.get('LLM_PROVIDER')}")
    print(f"LLM_MODEL: {config.get('LLM_MODEL')}")
    print(f"OLLAMA_BASE_URL: {config.get('OLLAMA_BASE_URL')}")
    print(f"OpenAI API key configured: {'yes' if config.get('OPENAI_API_KEY') and config.get('OPENAI_API_KEY') != 'your_openai_api_key_here' else 'no'}")


def test_langchain_availability():
    """Test if langchain libraries are available"""
    print(f"LangChain available: {LANGCHAIN_AVAILABLE}")
    print(f"Valid configuration: {HAS_VALID_CONFIG}")
    if not LANGCHAIN_AVAILABLE:
        print("Install with: pip install langchain-openai langchain-ollama langchain-core")
