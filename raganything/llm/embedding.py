from typing import Any, Dict, List, Optional

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

        init_kwargs: Dict[str, Any] = {"model": model}
        if api_base:
            init_kwargs["base_url"] = api_base
        if api_key:
            init_kwargs["api_key"] = api_key
        init_kwargs.update(extra)

        client = OpenAIEmbeddings(**init_kwargs)

        def sync_embed(texts: List[str]) -> List[List[float]]:
            return client.embed_documents(texts)

        return EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            func=sync_embed,
        )

    if p == "ollama":
        raise ValueError("Ollama embeddings via LangChain are not configured")

    raise ValueError(f"Unsupported embedding provider: {provider}")

