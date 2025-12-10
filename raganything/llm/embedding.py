import os
from typing import Any, Dict, List, Optional
import numpy as np

def build_embedding_func(
    provider: str,
    model: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_dim: int = 1536,
    max_token_size: int = 8192,
    extra: Optional[Dict[str, Any]] = None,
):
    from lightrag.utils import EmbeddingFunc

    p = (provider or "").lower()
    extra = extra or {}

    if p in ("openai", "openrouter", "azure-openai"):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            raise ValueError(f"LangChain OpenAI embeddings unavailable: {e}")

        # Fail fast if API key is missing to allow fallback logic to work
        if not api_key and not os.environ.get("OPENAI_API_KEY"):
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

    async def __call__(self, texts: List[str]) -> List[List[float]]:
        # Run sync method in a thread to avoid blocking event loop
        import asyncio
        import logging
        import sys
        logger = logging.getLogger("LazyLangChainEmbeddingWrapper")
        msg = f"Embedding request for {len(texts)} texts. First text len: {len(texts[0]) if texts else 0}"
        logger.debug(msg)
        print(f"[EmbeddingWrapper] {msg}", file=sys.stderr, flush=True)
        
        try:
            # Check client initialization
            if self._client is None:
                logger.debug(f"Initializing client with kwargs: {self.init_kwargs}")
                print(f"[EmbeddingWrapper] Initializing client...", file=sys.stderr, flush=True)
            
            # Use asyncio.wait_for to enforce a timeout on the thread execution
            # This protects against the sync call hanging indefinitely
            print(f"[EmbeddingWrapper] calling asyncio.to_thread with {len(texts)} texts...", file=sys.stderr, flush=True)
            
            # Heartbeat task to show we are alive
            async def _heartbeat():
                import asyncio
                import sys
                count = 0
                while True:
                    await asyncio.sleep(5)
                    count += 5
                    print(f"[EmbeddingWrapper] Still waiting for embedding... ({len(texts)} texts, {count}s elapsed)", file=sys.stderr, flush=True)

            heartbeat_task = asyncio.create_task(_heartbeat())
            
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.client.embed_documents, texts),
                    timeout=300 # Increased to 300 seconds (5 minutes) for large batches/slow models
                )
                print(f"[EmbeddingWrapper] Embedding request complete. Result len: {len(result)}", file=sys.stderr, flush=True)
                logger.debug("Embedding request complete.")
                return np.array(result)
            except asyncio.TimeoutError:
                err_msg = f"Embedding generation timed out after 300s for {len(texts)} texts"
                logger.error(err_msg)
                print(f"[EmbeddingWrapper] {err_msg}", file=sys.stderr, flush=True)
                raise
            except Exception as e:
                # Capture full traceback for better debugging
                import traceback
                tb = traceback.format_exc()
                logger.error(f"Embedding generation failed: {e}\n{tb}")
                print(f"[EmbeddingWrapper] Embedding generation failed: {e}\n{tb}", file=sys.stderr, flush=True)
                raise
            finally:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
        except Exception as e:
            # Fallback for outer try block if anything weird happens outside the inner try
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Embedding wrapper outer error: {e}\n{tb}")
            print(f"[EmbeddingWrapper] Embedding wrapper outer error: {e}\n{tb}", file=sys.stderr, flush=True)
            raise


class LocalEmbeddingWrapper:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    async def __call__(self, texts: List[str]) -> Any:
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
