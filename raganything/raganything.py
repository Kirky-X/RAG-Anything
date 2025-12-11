"""
Complete document parsing + multimodal content insertion Pipeline

This script integrates:
1. Document parsing (using configurable parsers)
2. Pure text content LightRAG insertion
3. Specialized processing for multimodal content (using different processors)
"""

import os
from typing import Dict, Any, Optional, Callable
import sys
import asyncio
import atexit
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Add project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file BEFORE importing LightRAG
# This is critical for TIKTOKEN_CACHE_DIR to work properly in offline environments
# The OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

# Load tiktoken cache dir from TOML config early so LightRAG/tiktoken can see it
try:
    from raganything.config import RAGAnythingConfig

    _pre_cfg = RAGAnythingConfig()
    if getattr(_pre_cfg, "tiktoken", None) and _pre_cfg.tiktoken.cache_dir:
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", _pre_cfg.tiktoken.cache_dir)
except Exception:
    pass

from lightrag import LightRAG
from raganything.logger import logger, init_logger

# Import configuration and modules
from raganything.config import RAGAnythingConfig
from raganything.query import QueryMixin
from raganything.processor import ProcessorMixin
from raganything.batch import BatchMixin
from raganything.utils import get_processor_supports
from raganything.parser import MineruParser, DoclingParser
from raganything.llm import LLMProviderConfig, build_llm
from raganything.llm.embedding import build_embedding_func
from raganything.llm.validation import validate_provider, ensure_non_empty

# Import specialized processors
from raganything.modalprocessors import (
    ImageModalProcessor,
    TableModalProcessor,
    EquationModalProcessor,
    GenericModalProcessor,
    ContextExtractor,
    ContextConfig,
)


@dataclass
class RAGAnything(QueryMixin, ProcessorMixin, BatchMixin):
    """Multimodal Document Processing Pipeline - Complete document parsing and insertion pipeline"""

    # Core Components
    # ---
    lightrag: Optional[LightRAG] = field(default=None)
    """Optional pre-initialized LightRAG instance."""

    llm_model_func: Optional[Callable] = field(default=None)
    """LLM model function for text analysis."""

    vision_model_func: Optional[Callable] = field(default=None)
    """Vision model function for image analysis."""

    embedding_func: Optional[Callable] = field(default=None)
    """Embedding function for text vectorization."""

    config: Optional[RAGAnythingConfig] = field(default=None)
    """Configuration object, if None will create with environment variables."""

    # LightRAG Configuration
    # ---
    lightrag_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments for LightRAG initialization when lightrag is not provided.
    This allows passing all LightRAG configuration parameters like:
    - kv_storage, vector_storage, graph_storage, doc_status_storage
    - top_k, chunk_top_k, max_entity_tokens, max_relation_tokens, max_total_tokens
    - cosine_threshold, related_chunk_number
    - chunk_token_size, chunk_overlap_token_size, tokenizer, tiktoken_model_name
    - embedding_batch_num, embedding_func_max_async, embedding_cache_config
    - llm_model_name, llm_model_max_token_size, llm_model_max_async, llm_model_kwargs
    - rerank_model_func, vector_db_storage_cls_kwargs, enable_llm_cache
    - max_parallel_insert, max_graph_nodes, addon_params, etc.
    """

    # Internal State
    # ---
    modal_processors: Dict[str, Any] = field(default_factory=dict, init=False)
    """Dictionary of multimodal processors."""

    context_extractor: Optional[ContextExtractor] = field(default=None, init=False)
    """Context extractor for providing surrounding content to modal processors."""

    parse_cache: Optional[Any] = field(default=None, init=False)
    """Parse result cache storage using LightRAG KV storage."""

    _parser_installation_checked: bool = field(default=False, init=False)
    """Flag to track if parser installation has been checked."""

    _storages_finalized: bool = field(default=False, init=False)
    """Flag to track if storages have been finalized."""

    def __post_init__(self):
        """Post-initialization setup following LightRAG pattern"""
        # Initialize configuration if not provided
        if self.config is None:
            self.config = RAGAnythingConfig()

        # Set working directory
        self.working_dir = self.config.working_dir

        # Configure logger from config (env variables still override inside init_logger)
        try:
            rotation_arg = self.config.logging.rotation
            retention_arg = self.config.logging.retention
            if rotation_arg == "00:00" and self.config.logging.max_bytes > 0:
                mb = self.config.logging.max_bytes // (1024 * 1024)
                kb = self.config.logging.max_bytes // 1024
                if mb > 0 and self.config.logging.max_bytes % (1024 * 1024) == 0:
                    rotation_arg = f"{mb} MB"
                elif kb > 0 and self.config.logging.max_bytes % 1024 == 0:
                    rotation_arg = f"{kb} KB"
                else:
                    rotation_arg = f"{self.config.logging.max_bytes} B"
            if (not retention_arg or retention_arg.strip() == "" or retention_arg == "7 days") and self.config.logging.backup_count > 0:
                retention_arg = f"{self.config.logging.backup_count} files"
            init_logger(
                level=self.config.logging.level,
                log_dir=Path(self.config.logging.dir) if self.config.logging.dir else None,
                rotation=rotation_arg,
                retention=retention_arg,
            )
        except Exception:
            pass
        self.logger = logger

        # Set up document parser
        self.doc_parser = (
            DoclingParser() if self.config.parser == "docling" else MineruParser()
        )

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(f"Created working directory: {self.working_dir}")

        # Log configuration info
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(f"  Working directory: {self.config.working_dir}")
        self.logger.info(f"  Parser: {self.config.parser}")
        self.logger.info(f"  Parse method: {self.config.parse_method}")
        self.logger.info(
            f"  Multimodal processing - Image: {self.config.enable_image_processing}, "
            f"Table: {self.config.enable_table_processing}, "
            f"Equation: {self.config.enable_equation_processing}"
        )
        self.logger.info(f"  Max concurrent files: {self.config.max_concurrent_files}")

    def close(self):
        """Cleanup resources when object is destroyed"""
        try:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # If we're in an async context, schedule cleanup
                    # Use ensure_future to attach to the loop without awaiting if we can't await
                    asyncio.ensure_future(self.finalize_storages())
                else:
                    # Should not happen if get_running_loop succeeds
                    asyncio.run(self.finalize_storages())
            except RuntimeError:
                # No running event loop
                try:
                    asyncio.run(self.finalize_storages())
                except Exception as e:
                     logger.warning(f"Failed to run cleanup with new loop: {e}")

        except Exception as e:
            logger.warning(f"Warning: Failed to finalize RAGAnything storages: {e}")

    def _create_context_config(self) -> ContextConfig:
        """Create context configuration from RAGAnything config"""
        return ContextConfig(
            context_window=self.config.context_window,
            context_mode=self.config.context_mode,
            max_context_tokens=self.config.max_context_tokens,
            include_headers=self.config.include_headers,
            include_captions=self.config.include_captions,
            filter_content_types=self.config.context_filter_content_types,
        )

    def _create_context_extractor(self) -> ContextExtractor:
        """Create context extractor with tokenizer from LightRAG"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG must be initialized before creating context extractor"
            )

        context_config = self._create_context_config()
        return ContextExtractor(
            config=context_config, tokenizer=self.lightrag.tokenizer
        )

    def _initialize_processors(self):
        """Initialize multimodal processors with appropriate model functions"""
        if self.lightrag is None:
            raise ValueError(
                "LightRAG instance must be initialized before creating processors"
            )

        # Create context extractor
        self.context_extractor = self._create_context_extractor()

        # Create different multimodal processors based on configuration
        self.modal_processors = {}

        if self.config.enable_image_processing:
            self.modal_processors["image"] = ImageModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.vision_model_func or self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_table_processing:
            self.modal_processors["table"] = TableModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        if self.config.enable_equation_processing:
            self.modal_processors["equation"] = EquationModalProcessor(
                lightrag=self.lightrag,
                modal_caption_func=self.llm_model_func,
                context_extractor=self.context_extractor,
            )

        # Always include generic processor as fallback
        self.modal_processors["generic"] = GenericModalProcessor(
            lightrag=self.lightrag,
            modal_caption_func=self.llm_model_func,
            context_extractor=self.context_extractor,
        )

        self.logger.info("Multimodal processors initialized with context support")
        self.logger.info(f"Available processors: {list(self.modal_processors.keys())}")
        self.logger.info(f"Context configuration: {self._create_context_config()}")

    def _maybe_build_llm_functions(self):
        """Build LangChain-backed llm/vision functions when not provided, based on config."""
        # LLM text function
        if self.llm_model_func is None:
            if not validate_provider(self.config.llm_provider):
                self.logger.warning(f"Invalid llm provider: {self.config.llm_provider}")
                return
            cfg = LLMProviderConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_base=self.config.llm_api_base or None,
                api_key=self.config.llm_api_key or None,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            try:
                lc_llm = build_llm(cfg)
                self.llm_model_func = lc_llm
                self.logger.info(f"LLM function built via LangChain provider: {cfg.provider}")
            except Exception as e:
                self.logger.warning(f"Failed to build LangChain LLM: {e}")

        # Vision function
        if self.vision_model_func is None:
            if not validate_provider(self.config.vision_provider):
                self.logger.warning(f"Invalid vision provider: {self.config.vision_provider}")
                return
            vcfg = LLMProviderConfig(
                provider=self.config.vision_provider,
                model=self.config.vision_model,
                api_base=self.config.vision_api_base or None,
                api_key=self.config.vision_api_key or None,
                timeout=self.config.vision_timeout,
                max_retries=self.config.vision_max_retries,
            )
            try:
                lc_vlm = build_llm(vcfg)
                self.vision_model_func = lc_vlm
                self.logger.info(f"Vision function built via LangChain provider: {vcfg.provider}")
            except Exception as e:
                self.logger.warning(f"Failed to build LangChain Vision: {e}")

        # LLM text function fallback to Ollama when OpenAI not configured
        if self.llm_model_func is None:
            if not validate_provider(self.config.llm_provider):
                self.logger.warning(f"Invalid llm provider: {self.config.llm_provider}")
                return
            cfg = LLMProviderConfig(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
                api_base=self.config.llm_api_base or None,
                api_key=self.config.llm_api_key or None,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            try:
                lc_llm = build_llm(cfg)
                self.llm_model_func = lc_llm
                self.logger.info(f"LLM function built via LangChain provider: {cfg.provider}")
            except Exception as e:
                # Fallback to Ollama locally if configured via env (OLLAMA_HOST)
                try:
                    from raganything.llm import LLMProviderConfig as LPC, build_llm as build_llm_fallback
                    ocfg = LPC(
                        provider="ollama",
                        model="qwen3:8b",
                        api_base=self.config.vision_api_base or "http://172.24.160.1:11434",
                        timeout=self.config.llm_timeout,
                        max_retries=self.config.llm_max_retries,
                    )
                    self.llm_model_func = build_llm_fallback(ocfg)
                    self.logger.info("LLM function built via Ollama fallback (qwen3:8b)")
                except Exception as e2:
                    self.logger.warning(f"Failed to build LLM via fallback: {e2}")

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config parameter: {key}")

    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG instance is initialized, create if necessary"""
        try:
            # Check parser installation first, but do not block non-parsing operations
            if not self._parser_installation_checked:
                if not self.doc_parser.check_installation():
                    self.logger.warning(
                        f"Parser '{self.config.parser}' is not properly installed; continuing initialization for operations that do not require parsing"
                    )
                    self._parser_installation_checked = True
                else:
                    self._parser_installation_checked = True
                    self.logger.info(f"Parser '{self.config.parser}' installation verified")

            if self.lightrag is not None:
                self.logger.debug(f"Checking LightRAG instance: type={type(self.lightrag)}")
                
                # Ensure RLock is removed from pre-provided LightRAG instance too
                if hasattr(self.lightrag, "lock"):
                    self.logger.warning("Removing 'lock' attribute from pre-provided LightRAG instance")
                    try:
                        delattr(self.lightrag, "lock")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove 'lock' attribute: {e}")

                # LightRAG was pre-provided, but we need to ensure it's properly initialized
                try:
                    # Ensure LightRAG storages are initialized
                    if (
                        not hasattr(self.lightrag, "_storages_status")
                        or self.lightrag._storages_status.name != "INITIALIZED"
                    ):
                        self.logger.info(
                            "Initializing storages for pre-provided LightRAG instance"
                        )
                        await self.lightrag.initialize_storages()
                        from lightrag.kg.shared_storage import (
                            initialize_pipeline_status,
                        )

                        await initialize_pipeline_status()

                    # Initialize parse cache if not already done
                    if self.parse_cache is None:
                        self.logger.info(
                            "Initializing parse cache for pre-provided LightRAG instance"
                        )
                        self.parse_cache = (
                            self.lightrag.key_string_value_json_storage_cls(
                                namespace="parse_cache",
                                workspace=self.lightrag.workspace,
                                global_config=self.lightrag.__dict__,
                                embedding_func=self.embedding_func,
                            )
                        )
                        await self.parse_cache.initialize()

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self._initialize_processors()

                    return {"success": True}

                except Exception as e:
                    error_msg = (
                        f"Failed to initialize pre-provided LightRAG instance: {str(e)}"
                    )
                    self.logger.error(error_msg, exc_info=True)
                    return {"success": False, "error": error_msg}

            # Try build llm/vision functions from config if missing
            self._maybe_build_llm_functions()

            # Validate required functions for creating new LightRAG instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided or buildable from config when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                try:
                    self.logger.info(f"Attempting to build embedding function with provider: {self.config.embedding_provider}")
                    self.embedding_func = build_embedding_func(
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        api_base=self.config.embedding_api_base or None,
                        api_key=self.config.embedding_api_key or None,
                        embedding_dim=self.config.embedding_dim,
                        max_token_size=8192,
                    )
                    self.logger.info(
                        f"Embedding function built successfully via provider: {self.config.embedding_provider}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to build primary embedding function ({self.config.embedding_provider}): {e}")
                    # Fallback to Ollama embeddings locally
                    try:
                        self.logger.info("Attempting fallback to Ollama embeddings...")
                        self.embedding_func = build_embedding_func(
                            provider="ollama",
                            model="bge-m3:567m",
                            api_base=self.config.vision_api_base or "http://172.24.160.1:11434",
                            embedding_dim=1024,
                            max_token_size=8192,
                        )
                        self.logger.info("Embedding function built via Ollama fallback (bge-m3:567m)")
                        # Update config to reflect fallback embedding dimension
                        self.config.embedding_dim = 1024
                        self.logger.info("Updated config.embedding_dim to 1024 for fallback model")
                    except Exception as e2:
                        error_msg = f"embedding_func must be provided and building from config failed. Primary error: {e}. Fallback error: {e2}"
                        self.logger.error(error_msg)
                        return {"success": False, "error": error_msg}

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # Prepare LightRAG initialization parameters
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Merge user-provided lightrag_kwargs, which can override defaults
            lightrag_params.update(self.lightrag_kwargs)

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(f"Initializing LightRAG with parameters: {log_params}")

            try:
                # Create LightRAG instance with merged parameters
                # NOTE: LightRAG instance might create an RLock which is not picklable.
                # We need to make sure that when LightRAG is initialized, it doesn't create unpicklable state
                # that persists during serialization.
                self.logger.debug("Creating LightRAG instance...")
                self.lightrag = LightRAG(**lightrag_params)
                self.logger.debug("LightRAG instance created.")
                
                # Remove 'lock' attribute immediately after creation to prevent pickle issues
                # This must be done BEFORE any asyncio operations that might involve this instance
                if hasattr(self.lightrag, "lock"):
                    self.logger.warning("Removing 'lock' attribute from LightRAG instance to prevent pickle issues")
                    try:
                        delattr(self.lightrag, "lock")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove 'lock' attribute: {e}")
                
                # Ensure embedding_func does not have RLock if it was attached during initialization
                if hasattr(self.lightrag, "embedding_func") and hasattr(self.lightrag.embedding_func, "lock"):
                     self.logger.warning("Removing 'lock' attribute from LightRAG embedding_func to prevent pickle issues")
                     try:
                        delattr(self.lightrag.embedding_func, "lock")
                     except Exception as e:
                        self.logger.warning(f"Failed to remove 'lock' attribute from embedding_func: {e}")

                
                # Also check for 'embedding_func' and 'llm_model_func' attached to instance
                # If they are bound methods of objects with locks, it might cause issues.
                # But we have already wrapped them in Lazy/Clean wrappers in their respective modules.
                
                self.logger.debug("Initializing LightRAG storages...")
                await self.lightrag.initialize_storages()
                self.logger.debug("LightRAG storages initialized.")
                await initialize_pipeline_status()

                # Initialize parse cache storage using LightRAG's KV storage
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=self.embedding_func,
                )
                await self.parse_cache.initialize()

                # Ensure all vector DBs have the correct embedding_func wrapper
                # LightRAG initializes VDBs with the provided embedding_func,
                # which we ensured is a LazyLangChainEmbeddingWrapper instance.
                # However, LightRAG might wrap it again or attach it differently.
                # Let's verify and force update if needed to ensure async compatibility.
                for vdb_name in ["chunks_vdb", "entities_vdb", "relationships_vdb"]:
                    if hasattr(self.lightrag, vdb_name) and hasattr(getattr(self.lightrag, vdb_name), "embedding_func"):
                        vdb = getattr(self.lightrag, vdb_name)
                        current_func = vdb.embedding_func
                        
                        # If it's a LightRAG EmbeddingFunc, the actual callable is in .func
                        if hasattr(current_func, "func"):
                             # UNWRAP IT! LightRAG's wrapper might be causing deadlocks with async functions.
                             # Since our wrappers (LazyLangChainEmbeddingWrapper/LocalEmbeddingWrapper) 
                             # handle their own async/sync logic, we want chunks_vdb to call them directly.
                             self.logger.warning(f"Unwrapping {vdb_name}.embedding_func to bypass LightRAG concurrency wrapper")
                             vdb.embedding_func = current_func.func
                             current_func = current_func.func
                        
                        self.logger.debug(f"LightRAG {vdb_name} embedding_func type: {type(current_func)}")

                # Diagnostic check for chunks_vdb (representative)
                if hasattr(self.lightrag, "chunks_vdb") and hasattr(self.lightrag.chunks_vdb, "embedding_func"):
                    current_func = self.lightrag.chunks_vdb.embedding_func
                    
                    # IMPORTANT: Check if the embedding function is properly awaited
                    # LightRAG expects a sync-looking call that might return an awaitable.
                    # We inject a small diagnostic log here.
                    self.logger.debug(f"Diagnostics: embedding_func name: {getattr(current_func, '__name__', 'unknown')}")
                    
                    # Test if the embedding function returns a coroutine when called
                    try:
                        test_res = current_func(["test"])
                        is_coroutine = asyncio.iscoroutine(test_res)
                        self.logger.info(f"Diagnostics: chunks_vdb.embedding_func(['test']) returns coroutine: {is_coroutine}")
                        
                        # If we get a coroutine but LightRAG expects sync result (which it might if it doesn't await),
                        # or if we get a sync result but LightRAG awaits it...
                        # LightRAG usually handles both if configured correctly, but let's be sure.
                        if is_coroutine:
                            # Clean up the coroutine
                            test_res.close()
                    except Exception as e:
                        self.logger.warning(f"Diagnostics: Failed to test embedding_func call: {e}")

                # Initialize processors after LightRAG is ready
                self._initialize_processors()

                self.logger.info(
                    "LightRAG, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                error_msg = f"Failed to initialize LightRAG instance: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = f"Unexpected error during LightRAG initialization: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}

    async def finalize_storages(self):
        """Finalize all storages including parse cache and LightRAG storages
        
        This method should be called when shutting down to properly clean up resources
        and persist any cached data. It will finalize both the parse cache and LightRAG's
        internal storages.
        """
        if self._storages_finalized:
            # self.logger.debug("Storages already finalized, skipping")
            return

        try:
            tasks = []
            
            # Finalize parse cache if it exists
            if self.parse_cache is not None:
                tasks.append(self.parse_cache.finalize())
                # self.logger.debug("Scheduled parse cache finalization")
            
            # Finalize LightRAG storages if LightRAG is initialized
            if self.lightrag is not None:
                # Ensure LightRAG internal loops are handled
                tasks.append(self.lightrag.finalize_storages())
                # self.logger.debug("Scheduled LightRAG storages finalization")
            
            # Run all finalization tasks concurrently
            if tasks:
                # Use shield to prevent cancellation during shutdown if loop is closing
                await asyncio.gather(*tasks, return_exceptions=True)
                # self.logger.info("Successfully finalized all RAGAnything storages")
            else:
                pass
                # self.logger.debug("No storages to finalize")
            
            self._storages_finalized = True
                
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                pass
                # self.logger.warning("Event loop closed during storage finalization")
            else:
                self.logger.error(f"Runtime error during storage finalization: {e}")
        except Exception as e:
            self.logger.error(f"Error during storage finalization: {e}")

    def check_parser_installation(self) -> bool:
        """
        Check if the configured parser is properly installed

        Returns:
            bool: True if the configured parser is properly installed
        """
        return self.doc_parser.check_installation()

    def verify_parser_installation_once(self) -> bool:
        if not self._parser_installation_checked:
            if not self.doc_parser.check_installation():
                raise RuntimeError(
                    f"Parser '{self.config.parser}' is not properly installed. "
                    "Please install it using pip install or uv pip install."
                )
            self._parser_installation_checked = True
            self.logger.info(f"Parser '{self.config.parser}' installation verified")
        return True

    def get_config_info(self) -> Dict[str, Any]:
        """Get current configuration information"""
        config_info = {
            "directory": {
                "working_dir": self.config.working_dir,
                "parser_output_dir": self.config.parser_output_dir,
            },
            "parsing": {
                "parser": self.config.parser,
                "parse_method": self.config.parse_method,
                "display_content_stats": self.config.display_content_stats,
            },
            "multimodal_processing": {
                "enable_image_processing": self.config.enable_image_processing,
                "enable_table_processing": self.config.enable_table_processing,
                "enable_equation_processing": self.config.enable_equation_processing,
            },
            "context_extraction": {
                "context_window": self.config.context_window,
                "context_mode": self.config.context_mode,
                "max_context_tokens": self.config.max_context_tokens,
                "include_headers": self.config.include_headers,
                "include_captions": self.config.include_captions,
                "filter_content_types": self.config.context_filter_content_types,
            },
            "batch_processing": {
                "max_concurrent_files": self.config.max_concurrent_files,
                "supported_file_extensions": self.config.supported_file_extensions,
                "recursive_folder_processing": self.config.recursive_folder_processing,
            },
            "logging": {
                "note": "Logging fields have been removed - configure logging externally",
            },
        }

        # Add LightRAG configuration if available
        if self.lightrag_kwargs:
            # Filter out sensitive data and callable objects for display
            safe_kwargs = {
                k: v
                for k, v in self.lightrag_kwargs.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            config_info["lightrag_config"] = {
                "custom_parameters": safe_kwargs,
                "note": "LightRAG will be initialized with these additional parameters",
            }
        else:
            config_info["lightrag_config"] = {
                "custom_parameters": {},
                "note": "Using default LightRAG parameters",
            }

        return config_info

    def set_content_source_for_context(
        self, content_source, content_format: str = "auto"
    ):
        """Set content source for context extraction in all modal processors

        Args:
            content_source: Source content for context extraction (e.g., MinerU content list)
            content_format: Format of content source ("minerU", "text_chunks", "auto")
        """
        if not self.modal_processors:
            self.logger.warning(
                "Modal processors not initialized. Content source will be set when processors are created."
            )
            return

        for processor_name, processor in self.modal_processors.items():
            try:
                processor.set_content_source(content_source, content_format)
                self.logger.debug(f"Set content source for {processor_name} processor")
            except Exception as e:
                self.logger.error(
                    f"Failed to set content source for {processor_name}: {e}"
                )

        self.logger.info(
            f"Content source set for context extraction (format: {content_format})"
        )

    def update_context_config(self, **context_kwargs):
        """Update context extraction configuration

        Args:
            **context_kwargs: Context configuration parameters to update
                (context_window, context_mode, max_context_tokens, etc.)
        """
        # Update the main config
        for key, value in context_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Updated context config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown context config parameter: {key}")

        # Recreate context extractor with new config if processors are initialized
        if self.lightrag and self.modal_processors:
            try:
                self.context_extractor = self._create_context_extractor()
                # Update all processors with new context extractor
                for processor_name, processor in self.modal_processors.items():
                    processor.context_extractor = self.context_extractor

                self.logger.info(
                    "Context configuration updated and applied to all processors"
                )
                self.logger.info(
                    f"New context configuration: {self._create_context_config()}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update context configuration: {e}")

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "config": self.get_config_info(),
            "models": {
                "llm_model": "External function"
                if self.llm_model_func
                else "Not provided",
                "vision_model": "External function"
                if self.vision_model_func
                else "Not provided",
                "embedding_model": "External function"
                if self.embedding_func
                else "Not provided",
            },
        }

        if not self.modal_processors:
            base_info["status"] = "Not initialized"
            base_info["processors"] = {}
        else:
            base_info["status"] = "Initialized"
            base_info["processors"] = {}

            for proc_type, processor in self.modal_processors.items():
                base_info["processors"][proc_type] = {
                    "class": processor.__class__.__name__,
                    "supports": get_processor_supports(proc_type),
                    "enabled": True,
                }

        return base_info
