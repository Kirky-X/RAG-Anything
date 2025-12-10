
import asyncio
import os
import pytest
from langchain_community.embeddings import OllamaEmbeddings
from raganything.llm.embedding import LazyLangChainEmbeddingWrapper

@pytest.mark.asyncio
async def test_embedding():
    print("Initializing wrapper...")
    init_kwargs = {
        "model": "bge-m3:567m",
        "base_url": "http://172.24.160.1:11434"
    }
    wrapper = LazyLangChainEmbeddingWrapper(OllamaEmbeddings, init_kwargs)
    
    print("Calling wrapper...")
    try:
        vectors = await wrapper(["Hello world"])
        print(f"Vectors received: {len(vectors)}")
        print(f"Vector dim: {len(vectors[0])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding())
