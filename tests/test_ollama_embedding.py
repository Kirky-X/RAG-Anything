
import asyncio
import os
import sys
from raganything.llm.embedding import build_embedding_func

async def main():
    print("Testing Ollama embedding...", flush=True)
    
    # Configure similar to verify_video_parser.py
    api_base = "http://172.24.160.1:11434"
    model = "bge-m3:567m"
    
    print(f"Building embedding func for {api_base} model {model}", flush=True)
    try:
        embedding_func = build_embedding_func(
            provider="ollama",
            model=model,
            api_base=api_base,
            embedding_dim=1024
        )
        
        wrapper = embedding_func.func
        
        texts = ["This is a test sentence.", "Another test sentence."]
        
        print(f"Sending request to {api_base} with model {model}", flush=True)
        embeddings = await wrapper(texts)
        print(f"Received {len(embeddings)} embeddings", flush=True)
        print(f"Embedding dim: {len(embeddings[0])}", flush=True)
        
    except Exception as e:
        print(f"Embedding failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
