import pickle
import asyncio
import pytest
import sys
from unittest.mock import patch, MagicMock

from raganything.llm.embedding import build_embedding_func, LazyLangChainEmbeddingWrapper
from raganything.llm.llm import build_llm, LLMProviderConfig

class MockEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    async def aembed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

class MockLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, messages):
        return "mock response"

def test_pickle_embedding_func():
    """Test picklability of embedding function."""
    # We use new=MockEmbeddings so that the class itself is replaced by our mock class,
    # rather than a MagicMock object. This ensures the class reference in LazyLangChainEmbeddingWrapper
    # is picklable.
    with patch("langchain_community.embeddings.OllamaEmbeddings", new=MockEmbeddings):
        embed_func_wrapper = build_embedding_func(
            provider="ollama",
            model="nomic-embed-text:latest",
            api_base="http://localhost:11434"
        )
        
        # Try pickling the wrapper
        pickled = pickle.dumps(embed_func_wrapper)
        unpickled = pickle.loads(pickled)
        
        assert unpickled is not None
        assert unpickled.embedding_dim == embed_func_wrapper.embedding_dim

@pytest.mark.asyncio
async def test_pickle_llm():
    """Test picklability of LLM function."""
    config = LLMProviderConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        api_base="http://localhost:11434"
    )
    
    # Mock dependencies that might be missing or problematic during test
    # We need to mock them BEFORE importing raganything.llm.ollama_client
    mock_lc_messages = MagicMock()
    mock_lc_messages.BaseMessage = MagicMock
    mock_lc_messages.AIMessage = MagicMock
    
    mock_lc_outputs = MagicMock()
    mock_lc_outputs.ChatResult = MagicMock
    
    mock_lc_ollama = MagicMock()
    mock_lc_ollama.ChatOllama = MagicMock

    with patch.dict(sys.modules, {
        "langchain_core.messages": mock_lc_messages,
        "langchain_core.outputs": mock_lc_outputs,
        "langchain_ollama": mock_lc_ollama
    }):
        # Now we can import the module that was failing
        import raganything.llm.ollama_client
        
        # Now patch the class in that module
        with patch("raganything.llm.ollama_client.RobustOllamaClient", new=MockLLM):
            llm_instance = build_llm(config)
            
            # Try pickling
            pickled = pickle.dumps(llm_instance)
            unpickled = pickle.loads(pickled)
            
            assert unpickled is not None
            # The unpickled object's client should be an instance of our MockLLM
            assert isinstance(unpickled.chat_model.client, MockLLM)
