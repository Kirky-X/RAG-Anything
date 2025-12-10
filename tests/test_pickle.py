
import pickle
import sys
import os

# Add project root to path
sys.path.append("/home/project/RAG-Anything")

from raganything.llm.embedding import build_embedding_func

def test_pickle_embedding_func():
    print("Testing picklability of embedding function...")
    try:
        # Test Ollama
        print("Building Ollama embedding func...")
        embed_func_wrapper = build_embedding_func(
            provider="ollama",
            model="nomic-embed-text:latest",
            api_base="http://localhost:11434"
        )
        print(f"Embedding func wrapper type: {type(embed_func_wrapper)}")
        print(f"Embedding func inner func type: {type(embed_func_wrapper.func)}")
        
        # Try pickling the wrapper
        print("Attempting to pickle EmbeddingFunc wrapper...")
        pickled = pickle.dumps(embed_func_wrapper)
        print("Successfully pickled EmbeddingFunc wrapper!")
        
        unpickled = pickle.loads(pickled)
        print("Successfully unpickled EmbeddingFunc wrapper!")
        
    except Exception as e:
        print(f"Pickle failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pickle_embedding_func()
