
import pickle
import sys
import os
import asyncio
import pytest

# Add project root to path
sys.path.append("/home/project/RAG-Anything")

from raganything.llm.llm import build_llm, LLMProviderConfig

@pytest.mark.asyncio
async def test_pickle_llm():
    print("Testing picklability of LLM function...")
    try:
        # Test Ollama
        print("Building Ollama LLM...")
        config = LLMProviderConfig(
            provider="ollama",
            model="qwen2.5-coder:7b",
            api_base="http://localhost:11434"
        )
        llm_instance = build_llm(config)
        print(f"LLM instance type: {type(llm_instance)}")
        print(f"Chat model type: {type(llm_instance.chat_model)}")
        
        # Try pickling the LLM instance
        print("Attempting to pickle LLM instance...")
        pickled = pickle.dumps(llm_instance)
        print("Successfully pickled LLM instance!")
        
        unpickled = pickle.loads(pickled)
        print("Successfully unpickled LLM instance!")
        
    except Exception as e:
        print(f"Pickle failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pickle_llm())
