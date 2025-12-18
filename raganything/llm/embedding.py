# Copyright (c) 2025 Kirky.X
# All rights reserved.

import os
from typing import Any, Dict, List, Optional
import numpy as np
from raganything.logger import logger

def build_embedding_func(
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_dim: int = 1536,
    max_token_size: int = 8192,
    extra: Optional[Dict[str, Any]] = None,
):
    """Build an embedding function for the specified provider and model.

    Args:
        provider (str): The embedding provider (openai, ollama, huggingface, etc.).
        model (str): The model name to use for embeddings.
        api_base (Optional[str]): Optional API base URL for the provider.
        api_key (Optional[str]): Optional API key for the provider.
        embedding_dim (int): The dimension of the embedding vectors. Defaults to 1536.
        max_token_size (int): Maximum token size for the model. Defaults to 8192.
        extra (Optional[Dict[str, Any]]): Extra parameters for the provider.

    Returns:
        EmbeddingFunc: An embedding function compatible with LightRAG.

    Raises:
        ValueError: If the provider is unsupported or dependencies are missing.
    """
    from lightrag.utils import EmbeddingFunc

    p = (provider or "").lower()
    extra = extra or {}

    # Force local embedding if the environment variable is set
    if os.environ.get("EMBEDDING_PROVIDER") == "local":
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=LocalEmbeddingWrapper(embedding_dim),
        )

    if p in ("openai", "openrouter", "azure-openai"):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            raise ValueError(f"LangChain OpenAI embeddings unavailable: {e}")

        # Check for dummy key to use local fallback (e.g. for testing)
        current_key = api_key or os.environ.get("OPENAI_API_KEY")
        if current_key == "DUMMY_KEY":
            from raganything.logger import logger
            logger.warning("Using LocalEmbeddingWrapper because OPENAI_API_KEY is DUMMY_KEY")
            return EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=max_token_size,
                func=LocalEmbeddingWrapper(embedding_dim),
            )

        # Fail fast if API key is missing to allow fallback logic to work
        if not current_key:
             raise ValueError("OpenAI API key is required but not provided in arguments or environment variables.")

        init_kwargs: Dict[str, Any] = {"model": model}
        if api_base:
            init_kwargs["base_url"] = api_base
        if api_key:
            init_kwargs["api_key"] = api_key
        init_kwargs.update(extra)

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=LazyLangChainEmbeddingWrapper(OpenAIEmbeddings, init_kwargs),
        )

    if p == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except Exception as e:
            raise ValueError(f"LangChain Ollama embeddings unavailable: {e}")

        init_kwargs: Dict[str, Any] = {"model": model}
        if api_base:
            init_kwargs["base_url"] = api_base
        init_kwargs.update(extra)

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=LazyLangChainEmbeddingWrapper(OllamaEmbeddings, init_kwargs),
        )

    if p == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception as e:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except Exception as e2:
                raise ValueError(f"LangChain HuggingFace embeddings unavailable: {e}, {e2}")

        init_kwargs: Dict[str, Any] = {"model_name": model}
        init_kwargs.update(extra)

        # IMPORTANT: Ensure HuggingFaceEmbeddings doesn't try to use multiprocessing which might fail in some environments
        # or when called from within an async loop
        if "encode_kwargs" not in init_kwargs:
            init_kwargs["encode_kwargs"] = {}
        # Force batch size to be reasonable to avoid OOM
        if "batch_size" not in init_kwargs.get("encode_kwargs", {}):
            init_kwargs["encode_kwargs"]["batch_size"] = 32

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=LazyLangChainEmbeddingWrapper(HuggingFaceEmbeddings, init_kwargs),
        )

    if p == "local":
        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=LocalEmbeddingWrapper(embedding_dim),
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


class LazyLangChainEmbeddingWrapper:
    def __init__(self, provider_cls, init_kwargs: Dict[str, Any]):
        self.provider_cls = provider_cls
        self.init_kwargs = init_kwargs
        self._client = None

    def __getstate__(self):
        """Support pickling by excluding the initialized client (which may have locks)"""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        """Restore state and reset client to None"""
        self.__dict__.update(state)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self.provider_cls(**self.init_kwargs)
        return self._client

    async def __call__(self, texts: List[str], **kwargs) -> List[List[float]]:
        # Run sync method in a thread to avoid blocking event loop
        import asyncio
        from raganything.logger import logger
        import sys
        
        # If texts is empty, return empty list
        if not texts:
            return np.array([])
            
        # logger = logging.getLogger("LazyLangChainEmbeddingWrapper")
        # msg = f"Embedding request for {len(texts)} texts. First text len: {len(texts[0]) if texts else 0}"
        # logger.debug(msg)
        # print(f"[EmbeddingWrapper] {msg}", file=sys.stderr, flush=True)
        
        try:
            # Check client initialization
            if self._client is None:
                # logger.debug(f"Initializing client with kwargs: {self.init_kwargs}")
                # print(f"[EmbeddingWrapper] Initializing client...", file=sys.stderr, flush=True)
                # Initialize client immediately (this is sync but fast usually)
                self._client = self.provider_cls(**self.init_kwargs)
            
            # Use asyncio.to_thread to run the sync embed_documents call in a separate thread
            # This is crucial for avoiding event loop blocking
            result = await asyncio.to_thread(self.client.embed_documents, texts)
            return np.array(result)

        except Exception as e:
            # Capture full traceback for better debugging
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Embedding generation failed: {e}\n{tb}")
            print(f"[EmbeddingWrapper] Embedding generation failed: {e}\n{tb}", file=sys.stderr, flush=True)
            raise


class LocalEmbeddingWrapper:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    async def __call__(self, texts: List[str], **kwargs) -> Any:
        import sys
        # Local mock implementation
        print(f"[LocalEmbeddingWrapper] Called with {len(texts)} texts (ASYNC)", file=sys.stderr, flush=True)
        import numpy as np
        
        # Deterministic but fast "embedding" based on SHA256
        import hashlib
        embeddings = []
        for t in texts:
            # Create a seed from text
            h = hashlib.sha256(t.encode()).hexdigest()
            # Generate deterministic pseudo-random vector
            np.random.seed(int(h[:8], 16))
            vec = np.random.rand(self.embedding_dim).tolist()
            embeddings.append(vec)
            
        print(f"[LocalEmbeddingWrapper] Generated {len(embeddings)} vectors", file=sys.stderr, flush=True)
        return np.array(embeddings)
