from unittest.mock import patch

import pytest

from raganything.llm.embedding import (LazyLangChainEmbeddingWrapper,
                                       build_embedding_func)


class MockEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def aembed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]


@pytest.mark.asyncio
async def test_lazy_wrapper_initialization():
    """Test LazyLangChainEmbeddingWrapper initialization and call."""
    init_kwargs = {"model": "test-model", "base_url": "http://localhost:11434"}

    # Mock the underlying embedding class
    with patch(
        "langchain_community.embeddings.OllamaEmbeddings", side_effect=MockEmbeddings
    ) as MockClass:
        wrapper = LazyLangChainEmbeddingWrapper(MockClass, init_kwargs)

        # Test calling the wrapper
        texts = ["Hello world", "Test sentence"]
        vectors = await wrapper(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 1024

        # Verify initialization happened
        MockClass.assert_called_once()
        call_args = MockClass.call_args[1]
        assert call_args["model"] == "test-model"
        assert call_args["base_url"] == "http://localhost:11434"


@pytest.mark.asyncio
async def test_build_embedding_func_ollama():
    """Test build_embedding_func for Ollama provider."""
    # Use new=MockEmbeddings instead of side_effect so that initialization works correctly without calling real __init__
    # and the class itself is replaced, making it safe for instantiation in the wrapper.
    with patch("langchain_community.embeddings.OllamaEmbeddings", new=MockEmbeddings):
        # We also need to patch the one imported in raganything.llm.embedding if it exists,
        # or if it's imported inside the function.
        # Since the function uses `from langchain_community.embeddings import OllamaEmbeddings` locally,
        # patching `langchain_community.embeddings.OllamaEmbeddings` should be enough if sys.modules is patched.

        embedding_func = build_embedding_func(
            provider="ollama",
            model="bge-m3:567m",
            api_base="http://localhost:11434",
            embedding_dim=1024,
        )

        assert embedding_func is not None
        assert embedding_func.embedding_dim == 1024

        # Test functionality
        wrapper = embedding_func.func
        texts = ["Test"]
        vectors = await wrapper(texts)
        assert len(vectors) == 1
        assert len(vectors[0]) == 1024


@pytest.mark.asyncio
async def test_build_embedding_func_openai():
    """Test build_embedding_func for OpenAI provider."""
    # We need to mock OpenAIEmbeddings if we were testing that path
    # For now, just ensuring the switch works if we had mocked it
    pass
