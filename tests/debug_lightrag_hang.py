
import os
import asyncio
import logging
import sys
from raganything.raganything import RAGAnything, RAGAnythingConfig
from raganything.llm.embedding import LazyLangChainEmbeddingWrapper

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_script")

async def main():
    work_dir = "./tests/resource/debug_lightrag_hang"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    config = RAGAnythingConfig()
    config.directory.working_dir = work_dir
    
    # Configure LLM
    config.llm.provider = "ollama"
    config.llm.model = "qwen2.5:3b"
    config.llm.api_base = "http://172.24.160.1:11434"
    
    # Configure Embedding
    config.embedding.provider = "ollama"
    config.embedding.model = "bge-m3:567m"
    config.embedding.api_base = "http://172.24.160.1:11434"
    config.embedding.dim = 1024
    
    # Force use of this config
    rag = RAGAnything(config=config)
    await rag._ensure_lightrag_initialized()
    
    logger.info("RAGAnything initialized.")
    logger.info(f"LightRAG instance: {rag.lightrag}")
    
    if hasattr(rag.lightrag, "chunks_vdb"):
        vdb = rag.lightrag.chunks_vdb
        logger.info(f"chunks_vdb type: {type(vdb)}")
        if hasattr(vdb, "embedding_func"):
            ef = vdb.embedding_func
            logger.info(f"embedding_func type: {type(ef)}")
            if hasattr(ef, "func"):
                logger.info(f"embedding_func.func type: {type(ef.func)}")
    
    # Create a dummy chunk
    chunks = {
        "chunk_test_1": {
            "content": "This is a test chunk to verify upsert.",
            "content_len": 30,
            "tokens": 10,
            "chunk_order_index": 0,
            "full_doc_id": "doc_test_1",
            "doc_id": "doc_test_1"
        }
    }
    
    logger.info("Attempting chunks_vdb.upsert...")
    try:
        # We simulate what processor.py does
        await asyncio.wait_for(rag.lightrag.chunks_vdb.upsert(chunks), timeout=30)
        logger.info("Upsert successful!")
    except asyncio.TimeoutError:
        logger.error("Upsert timed out!")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
