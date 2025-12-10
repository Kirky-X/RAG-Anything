import os
import json
import logging
import asyncio
import nest_asyncio
nest_asyncio.apply()
from pathlib import Path
from raganything.raganything import RAGAnything
from raganything.config import RAGAnythingConfig, DirectoryConfig, EmbeddingConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerificationScript")

# Set raganything logger to DEBUG
logging.getLogger("raganything").setLevel(logging.DEBUG)
logging.getLogger("LazyLangChainEmbeddingWrapper").setLevel(logging.DEBUG)
logging.getLogger("lightrag").setLevel(logging.DEBUG)

async def main():
    """
    Verification script for video processing via main pipeline.
    1. Load a video file
    2. Process through RAGAnything process_document_complete
    3. Ensure multimodal chunks and relations are stored and persisted
    4. Save a status report
    """
    
    # Config
    SOURCE_VIDEO = Path("/home/project/RAG-Anything/tests/resource/project_1.mp4")
    OUTPUT_DIR = Path("/home/project/RAG-Anything/tests/resource/verification_output_v2")
    # Force clean output directory for each run
    import shutil
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set working directory to output dir to ensure fresh LightRAG state
    os.environ["RAGANYTHING_WORKING_DIR"] = str(OUTPUT_DIR)
    
    if not SOURCE_VIDEO.exists():
        logger.error(f"Source video not found: {SOURCE_VIDEO}")
        return

    # 1. Use Full Video (No Slicing)
    logger.info("Step 1: Using full video for analysis...")
    target_path = SOURCE_VIDEO
    
    # 2. Run VideoParser
    logger.info("Step 2: Running RAGAnything main pipeline...")

    # Initialize RAG with explicit config
    config = RAGAnythingConfig(
        directory=DirectoryConfig(
            working_dir=str(OUTPUT_DIR),
            parser_output_dir=str(OUTPUT_DIR / "parser_output")
        ),
        embedding=EmbeddingConfig(
            dim=1024, # Set to 1024 for local fallback (bge-m3:567m) compatibility
            provider="ollama", # Prefer ollama if available, otherwise it will fallback
            model="bge-m3:567m",
            api_base="http://172.24.160.1:11434",
            api_key="",
            func_max_async=32,
            batch_num=16
        )
    )
    
    # Force update config fields after instantiation to avoid TOML interference
    # RAGAnythingConfig.__post_init__ might load config.toml and overwrite our values
    config.directory.working_dir = str(OUTPUT_DIR)
    config.directory.parser_output_dir = str(OUTPUT_DIR / "parser_output")
    
    # Force update embedding config to ensure correct values
    # Use 'ollama' as requested by user
    config.embedding.api_base = "http://172.24.160.1:11434"
    config.embedding.model = "bge-m3:567m"
    config.embedding.dim = 1024
    config.embedding.provider = "ollama"
    config.embedding.func_max_async = 4
    config.embedding.batch_num = 4

    # Configure LLM
    config.llm.provider = "ollama"
    config.llm.model = "qwen3:8b"
    config.llm.api_base = "http://172.24.160.1:11434"
    config.llm.api_key = "none"

    # Configure Vision
    config.vision.provider = "ollama"
    config.vision.model = "qwen3-vl:2b"
    config.vision.api_base = "http://172.24.160.1:11434"
    
    # Configure Vision (VLM) to use ollama
    # We inject this into the rag config so VlmParser picks it up
    if not hasattr(config, "vision"):
        # Create a dummy structure if needed, or just set it in the dict representation if we could
        # But RAGAnythingConfig is a dataclass. Let's see if we can hack it or if we need to pass it via config file.
        # Actually RAGAnythingConfig might not have a 'vision' field defined in the class if it's dynamic.
        # Let's check config.py.
        pass
    
    # We can also set environment variables that VlmParser might respect if we updated it?
    # No, VlmParser reads from toml. 
    # But wait, VlmParser is initialized inside RAGAnything -> Processor.
    # We can try to patch the config object inside RAGAnything if it stores it.
    
    # Set environment variable for Ollama client just in case
    os.environ["OLLAMA_HOST"] = "http://172.24.160.1:11434"

    rag = RAGAnything(config=config)
    
    # HACK: Manually override the VLM config in the processor's config if possible
    # Or better, create a temporary config.toml with the settings we want.
    
    # Let's create a temporary config.toml in the CURRENT WORKING DIRECTORY
    # because VlmParser looks for "config.toml" in CWD by default.
    config_toml_content = """
[raganything.vision]
provider = "ollama"
model = "qwen3-vl:2b"
timeout = 60
api_base = "http://172.24.160.1:11434"
"""
    # Use absolute path to CWD
    cwd_config_path = Path(os.getcwd()) / "config.toml"
    # Backup existing config if any
    config_backup_path = None
    if cwd_config_path.exists():
        config_backup_path = Path(os.getcwd()) / "config.toml.bak"
        import shutil
        shutil.copy2(cwd_config_path, config_backup_path)
        
    with open(cwd_config_path, "w") as f:
        f.write(config_toml_content)
        
    logger.info(f"Created temporary config.toml at {cwd_config_path} to force ollama VLM")

    try:
        # Initialize RAG with explicit config
        rag = RAGAnything(config=config)
        
        # Set environment variable for Ollama client just in case
        os.environ["OLLAMA_HOST"] = "http://172.24.160.1:11434"
        
        logger.info(f"RAG initialized with working_dir: {rag.config.directory.working_dir}")

        # Inspect LightRAG internals
        if hasattr(rag, "lightrag") and hasattr(rag.lightrag, "chunks_vdb"):
            logger.info(f"LightRAG chunks_vdb type: {type(rag.lightrag.chunks_vdb)}")
            if hasattr(rag.lightrag.chunks_vdb, "embedding_func"):
                ef = rag.lightrag.chunks_vdb.embedding_func
                logger.info(f"LightRAG chunks_vdb.embedding_func type: {type(ef)}")
                logger.info(f"LightRAG chunks_vdb.embedding_func repr: {repr(ef)}")
                if hasattr(ef, "func"):
                    logger.info(f"LightRAG chunks_vdb.embedding_func.func type: {type(ef.func)}")
                    logger.info(f"LightRAG chunks_vdb.embedding_func.func repr: {repr(ef.func)}")
        
        try:
            await rag.process_document_complete(
                file_path=str(target_path),
                output_dir=str(OUTPUT_DIR),
                parse_method="auto",
                display_stats=True,
                video_fps=0.5,
                cleanup_frames=False,  # Ensure frames are preserved for multimodal processing
                force_parse=True,  # Force re-parsing to avoid stale cache issues
            )
            info = rag.get_processor_info()
            result = {"mode": "pipeline", "info": info}
            
            result_json_path = OUTPUT_DIR / "pipeline_status.json"
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Verification complete (pipeline). Saved to {result_json_path}")
            
        except Exception as e:
            logger.error(f"Pipeline mode failed: {e}")
            raise
    finally:
        # Ensure cleanup of storages to prevent event loop errors
        if 'rag' in locals() and rag:
            await rag.finalize_storages()

        # Restore config.toml if it was backed up
        if config_backup_path and config_backup_path.exists():
            import shutil
            shutil.move(config_backup_path, cwd_config_path)
            logger.info("Restored original config.toml")
        elif cwd_config_path.exists():
            os.remove(cwd_config_path)
            logger.info("Removed temporary config.toml")

if __name__ == "__main__":
    asyncio.run(main())
