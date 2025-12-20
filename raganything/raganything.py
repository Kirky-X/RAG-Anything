"""
Complete document parsing + multimodal content insertion Pipeline

This script integrates:
1. Document parsing (using configurable parsers)
2. Pure text content LightRAG insertion
3. Specialized processing for multimodal content (using different processors)
"""

import asyncio
import atexit
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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

from lightrag.lightrag import LightRAG

from raganything.batch import BatchMixin
from raganything.config import RAGAnythingConfig
from raganything.llm import (LLMProviderConfig, build_embedding_func,
                             build_llm, ensure_non_empty, validate_provider)
from raganything.logger import init_logger, logger
from raganything.i18n_logger import get_i18n_logger
# Import specialized processors
from raganything.modalprocessors import (ContextConfig, ContextExtractor,
                                         EquationModalProcessor,
                                         GenericModalProcessor,
                                         ImageModalProcessor,
                                         TableModalProcessor)
from raganything.parser import DoclingParser, MineruParser
from raganything.processor import ProcessorMixin
from raganything.query import QueryMixin
from raganything.utils import get_processor_supports
from raganything.i18n import _


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
            if (
                not retention_arg
                or retention_arg.strip() == ""
                or retention_arg == "7 days"
            ) and self.config.logging.backup_count > 0:
                retention_arg = f"{self.config.logging.backup_count} files"
            init_logger(
                level=self.config.logging.level,
                log_dir=(
                    Path(self.config.logging.dir) if self.config.logging.dir else None
                ),
                rotation=rotation_arg,
                retention=retention_arg,
            )
        except Exception:
            pass
        self.logger = get_i18n_logger()

        # Set up document parser
        self.doc_parser = (
            DoclingParser() if self.config.parser == "docling" else MineruParser()
        )

        # Register close method for cleanup
        atexit.register(self.close)

        # Create working directory if needed
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
            self.logger.info(_("Created working directory: {}").format(self.working_dir))

        # Log configuration info
        self.logger.info("RAGAnything initialized with config:")
        self.logger.info(_("  Working directory: {}").format(self.config.working_dir))
        self.logger.info(_("  Parser: {}").format(self.config.parser))
        self.logger.info(_("  Parse method: {}").format(self.config.parse_method))
        self.logger.info(
            _("  Multimodal processing - Image: {}, "
            "Table: {}, "
            "Equation: {}").format(self.config.enable_image_processing, self.config.enable_table_processing, self.config.enable_equation_processing)
        )
        self.logger.info(_("  Max concurrent files: {}").format(self.config.max_concurrent_files))

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
                    logger.warning(_("Failed to run cleanup with new loop: {}").format(e))

        except Exception as e:
            logger.warning(_("Warning: Failed to finalize RAGAnything storages: {}").format(e))

    async def initialize(self):
        """Async initialization of LightRAG and related components"""
        # Ensure LLM functions are built
        self._maybe_build_llm_functions()

        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Initialize modal processors
        self._initialize_processors()

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
            if self.vision_model_func is None:
                self.logger.error(
                    "Vision model function is not available. Image processing will be disabled."
                )
                # Don't create image processor if vision model is not available
            else:
                self.modal_processors["image"] = ImageModalProcessor(
                    lightrag=self.lightrag,
                    modal_caption_func=self.vision_model_func,
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
        self.logger.info(_("Available processors: {}").format(list(self.modal_processors.keys())))
        self.logger.info(_("Context configuration: {}").format(self._create_context_config()))

    def _maybe_build_llm_functions(self):
        """Build LangChain-backed llm/vision functions when not provided, based on config."""
        # Import here to ensure it's available in this scope
        from raganything.llm import LLMProviderConfig, build_llm

        # LLM text function
        if self.llm_model_func is None:
            if not validate_provider(self.config.llm_provider):
                self.logger.warning(_("Invalid llm provider: {}").format(self.config.llm_provider))
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
                self.logger.info(
                    _("LLM function built via LangChain provider: {}").format(cfg.provider)
                )
            except Exception as e:
                self.logger.warning(_("Failed to build LangChain LLM: {}").format(e))

        # Vision function
        if self.vision_model_func is None:
            if not validate_provider(self.config.vision_provider):
                self.logger.warning(
                    _("Invalid vision provider: {}").format(self.config.vision_provider)
                )
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
                self.logger.info(
                    _("Vision function built via LangChain provider: {}").format(vcfg.provider)
                )
            except Exception as e:
                self.logger.error(_("Failed to build LangChain Vision model: {}").format(e))
                # Don't proceed if vision model fails to build - this prevents fallback to LLM
                raise RuntimeError(_("Vision model configuration failed: {}").format(e))

        # LLM text function fallback to Ollama when OpenAI not configured
        if self.llm_model_func is None:
            if not validate_provider(self.config.llm_provider):
                self.logger.warning(_("Invalid llm provider: {}").format(self.config.llm_provider))
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
                self.logger.info(
                    _("LLM function built via LangChain provider: {}").format(cfg.provider)
                )
            except Exception as e:
                # Fallback to Ollama locally if configured via env (OLLAMA_HOST)
                try:
                    ocfg = LLMProviderConfig(
                        provider="ollama",
                        model="qwen3:8b",
                        api_base=self.config.vision_api_base
                        or "http://172.24.160.1:11434",
                        timeout=self.config.llm_timeout,
                        max_retries=self.config.llm_max_retries,
                    )
                    self.llm_model_func = build_llm(ocfg)
                    self.logger.info(
                        "LLM function built via Ollama fallback (qwen3:8b)"
                    )
                except Exception as e2:
                    self.logger.warning(_("Failed to build LLM via fallback: {}").format(e2))

    def update_config(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(_("Updated config: {} = {}").format(key, value))
            else:
                self.logger.warning(_("Unknown config parameter: {}").format(key))

    async def _ensure_lightrag_initialized(self):
        """Ensure LightRAG instance is initialized, create if necessary"""
        try:
            self.logger.info("Starting _ensure_lightrag_initialized...")
            # Check parser installation first, but do not block non-parsing operations
            if not self._parser_installation_checked:
                self.logger.info("Checking parser installation...")
                if not self.doc_parser.check_installation():
                    self.logger.warning(
                        f"Parser '{self.config.parser}' is not properly installed; continuing initialization for operations that do not require parsing"
                    )
                    self._parser_installation_checked = True
                else:
                    self._parser_installation_checked = True
                    self.logger.info(
                        f"Parser '{self.config.parser}' installation verified"
                    )

            if self.lightrag is not None:
                self.logger.info(
                    "Pre-provided LightRAG instance found. Verifying initialization..."
                )
                self.logger.debug(
                    _("Checking LightRAG instance: type={}").format(type(self.lightrag))
                )

                # Ensure RLock is removed from pre-provided LightRAG instance too
                if hasattr(self.lightrag, "lock"):
                    self.logger.warning(
                        "Removing 'lock' attribute from pre-provided LightRAG instance"
                    )
                    try:
                        delattr(self.lightrag, "lock")
                    except Exception as e:
                        self.logger.warning(_("Failed to remove 'lock' attribute: {}").format(e))

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
                        self.logger.info(
                            "Storages initialized for pre-provided LightRAG instance."
                        )

                    # Ensure pipeline status is initialized (required for multimodal processing status tracking)
                    try:
                        from lightrag.kg.shared_storage import \
                            initialize_pipeline_status

                        await initialize_pipeline_status()
                        self.logger.info("Pipeline status initialized.")
                    except ImportError:
                        self.logger.warning(
                            "Could not import initialize_pipeline_status from lightrag.kg.shared_storage"
                        )
                    except Exception as e:
                        self.logger.warning(
                            _("Failed to initialize pipeline status: {}").format(e)
                        )

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
                        self.logger.info("Parse cache initialized.")

                    # Initialize processors if not already done
                    if not self.modal_processors:
                        self.logger.info("Initializing modal processors...")
                        self._initialize_processors()
                        self.logger.info("Modal processors initialized.")

                    self.logger.info(
                        "Pre-provided LightRAG instance verification complete."
                    )
                    # Continue with the rest of initialization instead of returning early

                except Exception as e:
                    error_msg = (
                        _("Failed to initialize pre-provided LightRAG instance: {}").format(str(e))
                    )
                    self.logger.error(error_msg, exc_info=True)
                    # Don't return error, let the exception propagate
                    raise RuntimeError(error_msg)

            self.logger.info("No pre-provided LightRAG instance. Building new one...")
            # Try build llm/vision functions from config if missing
            self.logger.info("Building LLM and Vision functions if not provided...")
            self._maybe_build_llm_functions()
            self.logger.info("LLM and Vision functions build process complete.")

            # Validate required functions for creating new LightRAG instance
            if self.llm_model_func is None:
                error_msg = "llm_model_func must be provided or buildable from config when LightRAG is not pre-initialized"
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                self.logger.info("Building embedding function...")
                try:
                    self.logger.info(
                        _("Attempting to build embedding function with provider: {}").format(self.config.embedding_provider)
                    )
                    self.embedding_func = build_embedding_func(
                        provider=self.config.embedding_provider,
                        model=self.config.embedding_model,
                        api_base=self.config.embedding_api_base or None,
                        api_key=self.config.embedding_api_key or None,
                        embedding_dim=self.config.embedding_dim,
                        max_token_size=8192,
                    )
                    self.logger.info(
                        _("Embedding function built successfully via provider: {}").format(self.config.embedding_provider)
                    )
                except Exception as e:
                    self.logger.warning(
                        _("Failed to build primary embedding function ({}): {}").format(self.config.embedding_provider, e)
                    )
                    # Fallback to Ollama embeddings locally
                    try:
                        self.logger.info("Attempting fallback to Ollama embeddings...")
                        self.embedding_func = build_embedding_func(
                            provider="ollama",
                            model="mxbai-embed-large",
                            api_base=self.config.embedding_api_base
                            or "http://172.24.160.1:11434",
                            max_token_size=8192,
                        )
                        self.logger.info(
                            "Embedding function built successfully via Ollama fallback."
                        )
                    except Exception as e2:
                        error_msg = _("Failed to build embedding function with all providers: {}").format(e2)
                        self.logger.error(error_msg)
                        return {"success": False, "error": error_msg}

            if self.embedding_func is None:
                error_msg = "Embedding function could not be built."
                self.logger.error(error_msg)
                return {"success": False, "error": error_msg}

            self.logger.info(
                "All required functions are ready. Creating LightRAG instance..."
            )
            # Create LightRAG instance
            self.lightrag = LightRAG(
                embedding_func=self.embedding_func,
                llm_model_func=self.llm_model_func,
                **self.lightrag_kwargs,
            )
            self.logger.info("LightRAG instance created. Initializing storages...")
            await self.lightrag.initialize_storages()
            self.logger.info("LightRAG storages initialized.")

            # Initialize pipeline status
            self.logger.info("Initializing pipeline status...")
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()
            self.logger.info("Pipeline status initialized.")

            # Initialize parse cache
            self.logger.info("Initializing parse cache...")
            self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                namespace="parse_cache",
                workspace=self.lightrag.workspace,
                global_config=self.lightrag.__dict__,
                embedding_func=self.embedding_func,
            )
            await self.parse_cache.initialize()
            self.logger.info("Parse cache initialized.")

            # Initialize processors
            self.logger.info("Initializing modal processors...")
            self._initialize_processors()
            self.logger.info("Modal processors initialized.")

            self.logger.info("_ensure_lightrag_initialized completed successfully.")
            # Don't return success status, just complete
            return
        except Exception as e:
            error_msg = _("An unexpected error occurred in _ensure_lightrag_initialized: {}").format(str(e))
            self.logger.error(error_msg, exc_info=True)
            # Don't return error, let the exception propagate
            raise RuntimeError(error_msg)

            from lightrag.kg.shared_storage import initialize_pipeline_status

            # Prepare LightRAG initialization parameters
            lightrag_params = {
                "working_dir": self.working_dir,
                "llm_model_func": self.llm_model_func,
                "embedding_func": self.embedding_func,
            }

            # Merge user-provided lightrag_kwargs, which can override defaults
            lightrag_params.update(self.lightrag_kwargs)

            # Ensure embedding_func is a pure async function wrapper
            # LightRAG calls priority_limit_async_func_call(embedding_func) in __post_init__
            # If embedding_func is a sync function returning a coroutine, or worse, a coroutine object itself,
            # priority_limit_async_func_call might fail with 'coroutine object is not callable'

            # Diagnostic log
            self.logger.debug(
                _("Pre-init embedding_func type: {}").format(type(self.embedding_func))
            )

            if self.embedding_func:
                # Check if it's already a coroutine object (BAD)
                if asyncio.iscoroutine(self.embedding_func):
                    self.logger.critical(
                        "embedding_func IS A COROUTINE OBJECT! This is invalid."
                    )

                # Force wrap in a clean async def function
                # This ensures it's definitely an async function, not a sync wrapper or partial
                original_ef = self.embedding_func

                async def clean_async_embedding_wrapper(texts: list[str]) -> Any:
                    if asyncio.iscoroutinefunction(original_ef):
                        return await original_ef(texts)
                    elif hasattr(
                        original_ef, "__call__"
                    ) and asyncio.iscoroutinefunction(original_ef.__call__):
                        res = original_ef(texts)
                        if asyncio.iscoroutine(res):
                            return await res
                        return res
                    else:
                        # Sync call
                        return original_ef(texts)

                # Copy attributes from original function if they exist
                if hasattr(original_ef, "embedding_dim"):
                    clean_async_embedding_wrapper.embedding_dim = (
                        original_ef.embedding_dim
                    )
                if hasattr(original_ef, "max_token_size"):
                    clean_async_embedding_wrapper.max_token_size = (
                        original_ef.max_token_size
                    )

                # IMPORTANT: Some LightRAG vector DB implementations (like NanoVectorDB) access embedding_dim directly
                # on the function object during __post_init__.
                # If we don't have it on the original function (e.g. because it's a partial or wrapped),
                # we must try to get it from the underlying model or set a default/dummy if appropriate,
                # or ensure the user provided it.
                if not hasattr(clean_async_embedding_wrapper, "embedding_dim"):
                    self.logger.warning(
                        "embedding_func missing 'embedding_dim' attribute. Attempting to infer or retrieve."
                    )
                    # Try to see if we can get it from the instance if it's a method
                    if hasattr(original_ef, "__self__") and hasattr(
                        original_ef.__self__, "embedding_dim"
                    ):
                        clean_async_embedding_wrapper.embedding_dim = (
                            original_ef.__self__.embedding_dim
                        )
                        self.logger.info(
                            f"Retrieved embedding_dim {clean_async_embedding_wrapper.embedding_dim} from bound method's self"
                        )
                    # Try to see if it's a LazyLangChainEmbeddingWrapper
                    elif hasattr(original_ef, "func") and hasattr(
                        original_ef.func, "embedding_dim"
                    ):
                        clean_async_embedding_wrapper.embedding_dim = (
                            original_ef.func.embedding_dim
                        )
                        self.logger.info(
                            f"Retrieved embedding_dim {clean_async_embedding_wrapper.embedding_dim} from underlying func"
                        )
                    # As a last resort, check if original_ef IS the wrapper class instance (callable)
                    elif hasattr(original_ef, "embedding_dim"):
                        clean_async_embedding_wrapper.embedding_dim = (
                            original_ef.embedding_dim
                        )
                        self.logger.info(
                            f"Retrieved embedding_dim {clean_async_embedding_wrapper.embedding_dim} from original_ef itself"
                        )
                    # Fallback to default if still missing (NanoVectorDB needs it)
                    else:
                        # Use a common default or try to invoke it once to get dimension (expensive but safe)
                        self.logger.warning(
                            "Could not find embedding_dim. Using default 1536 (OpenAI). This might be wrong!"
                        )
                        clean_async_embedding_wrapper.embedding_dim = 1536

                lightrag_params["embedding_func"] = clean_async_embedding_wrapper
                self.logger.info(
                    "Wrapped embedding_func in clean_async_embedding_wrapper for LightRAG initialization"
                )

            # Diagnostic log again
            self.logger.debug(
                _("Pre-init embedding_func type for LightRAG: {}").format(type(lightrag_params.get('embedding_func')))
            )
            if lightrag_params.get("embedding_func"):
                ef = lightrag_params["embedding_func"]
                self.logger.debug(
                    _("embedding_func attributes: embedding_dim={}").format(getattr(ef, 'embedding_dim', 'MISSING'))
                )

            # Log the parameters being used for initialization (excluding sensitive data)
            log_params = {
                k: v
                for k, v in lightrag_params.items()
                if not callable(v)
                and k not in ["llm_model_kwargs", "vector_db_storage_cls_kwargs"]
            }
            self.logger.info(_("Initializing LightRAG with parameters: {}").format(log_params))

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
                    self.logger.warning(
                        "Removing 'lock' attribute from LightRAG instance to prevent pickle issues"
                    )
                    try:
                        delattr(self.lightrag, "lock")
                    except Exception as e:
                        self.logger.warning(_("Failed to remove 'lock' attribute: {}").format(e))

                # Ensure embedding_func does not have RLock if it was attached during initialization
                if hasattr(self.lightrag, "embedding_func") and hasattr(
                    self.lightrag.embedding_func, "lock"
                ):
                    self.logger.warning(
                        "Removing 'lock' attribute from LightRAG embedding_func to prevent pickle issues"
                    )
                    try:
                        delattr(self.lightrag.embedding_func, "lock")
                    except Exception as e:
                        self.logger.warning(
                            _("Failed to remove 'lock' attribute from embedding_func: {}").format(e)
                        )

                # Also check for 'embedding_func' and 'llm_model_func' attached to instance
                # If they are bound methods of objects with locks, it might cause issues.
                # But we have already wrapped them in Lazy/Clean wrappers in their respective modules.

                self.logger.debug("Initializing LightRAG storages...")
                await self.lightrag.initialize_storages()
                self.logger.debug("LightRAG storages initialized.")
                await initialize_pipeline_status()

                # Initialize parse cache storage using LightRAG's KV storage
                # Use raw embedding_func for KV storage if it's not vector storage
                # For KV storage, embedding_func is usually not needed or should be sync if used for key hashing
                # Let's pass None if we can't determine, or just the raw function
                self.parse_cache = self.lightrag.key_string_value_json_storage_cls(
                    namespace="parse_cache",
                    workspace=self.lightrag.workspace,
                    global_config=self.lightrag.__dict__,
                    embedding_func=None,
                    # KV storage likely doesn't need embedding function for key generation if we use md5
                )
                await self.parse_cache.initialize()

                # Ensure all vector DBs have the correct embedding_func wrapper
                # LightRAG initializes VDBs with the provided embedding_func,
                # which we ensured is a LazyLangChainEmbeddingWrapper instance.
                # However, LightRAG might wrap it again or attach it differently.
                # Let's verify and force update if needed to ensure async compatibility.
                for vdb_name in ["chunks_vdb", "entities_vdb", "relationships_vdb"]:
                    if hasattr(self.lightrag, vdb_name) and hasattr(
                        getattr(self.lightrag, vdb_name), "embedding_func"
                    ):
                        vdb = getattr(self.lightrag, vdb_name)
                        current_func = vdb.embedding_func

                        # If it's a LightRAG EmbeddingFunc, the actual callable is in .func
                        if hasattr(current_func, "func"):
                            # UNWRAP IT! LightRAG's wrapper might be causing deadlocks with async functions.
                            # Since our wrappers (LazyLangChainEmbeddingWrapper/LocalEmbeddingWrapper)
                            # handle their own async/sync logic, we want chunks_vdb to call them directly.
                            self.logger.warning(
                                f"Unwrapping {vdb_name}.embedding_func to bypass LightRAG concurrency wrapper"
                            )
                            vdb.embedding_func = current_func.func
                            current_func = current_func.func

                        # CRITICAL FIX: Ensure embedding_func is wrapped correctly for LightRAG's priority_limit_async_func_call
                        # The error 'coroutine object is not callable' in LightRAG initialization (line 515)
                        # happens because LightRAG wraps embedding_func with priority_limit_async_func_call IMMEDIATELY.
                        # It seems priority_limit_async_func_call expects a function, but something might be returning a coroutine object directly.
                        # OR, more likely, we need to make sure what we pass to LightRAG IS a callable function.

                        self.logger.debug(
                            _("LightRAG {} embedding_func type: {}").format(vdb_name, type(current_func))
                        )

                        # Force wrapper to be strictly an async function if it's not already
                        # This ensures asyncio.iscoroutinefunction(vdb.embedding_func) returns True
                        # preventing LightRAG from treating it as sync and causing issues
                        if not asyncio.iscoroutinefunction(current_func):
                            # Case 1: Class instance with async __call__ (like LazyLangChainEmbeddingWrapper)
                            if hasattr(
                                current_func, "__call__"
                            ) and asyncio.iscoroutinefunction(current_func.__call__):
                                original_func = current_func

                                async def async_embedding_wrapper(
                                    texts: list[str],
                                ) -> Any:
                                    # Ensure the result is awaited if it's a coroutine
                                    res = original_func(texts)
                                    if asyncio.iscoroutine(res):
                                        return await res
                                    return res

                                vdb.embedding_func = async_embedding_wrapper
                                self.logger.info(
                                    f"Wrapped {vdb_name}.embedding_func in explicit async wrapper (from async __call__)"
                                )

                            # Case 2: Already a LightRAG priority_limit wrapper (function that returns a wrapper)
                            # LightRAG wraps functions with priority_limit_async_func_call which returns a wrapper
                            # If it's already wrapped, we might need to leave it, BUT we need to ensure the underlying
                            # function is correctly handled.
                            elif (
                                hasattr(current_func, "__name__")
                                and current_func.__name__ == "wrapper"
                                and "priority_limit" in repr(current_func)
                            ):
                                # It's likely already LightRAG's wrapper.
                                # If LightRAG wrapped a sync-looking function that returns a coroutine, it might be broken.
                                # We need to ensure it's an async function to prevent deadlocks or coroutine object errors
                                # However, if it's already wrapped, we can't easily change it.
                                # But we can verify if it's a coroutine function.
                                if not asyncio.iscoroutinefunction(current_func):
                                    # If it's a sync wrapper that returns a coroutine, we wrap it in an async def
                                    original_wrapper = current_func

                                    async def async_wrapper_fix(*args, **kwargs):
                                        res = original_wrapper(*args, **kwargs)
                                        if asyncio.iscoroutine(res):
                                            return await res
                                        return res

                                    # Preserve attributes
                                    if hasattr(original_wrapper, "embedding_dim"):
                                        async_wrapper_fix.embedding_dim = (
                                            original_wrapper.embedding_dim
                                        )
                                    if hasattr(original_wrapper, "max_token_size"):
                                        async_wrapper_fix.max_token_size = (
                                            original_wrapper.max_token_size
                                        )

                                    vdb.embedding_func = async_wrapper_fix
                                    self.logger.info(
                                        f"Fixed {vdb_name}.embedding_func wrapper to be explicitly async"
                                    )

                            # Case 3: Sync function that returns a coroutine (rare but possible)
                            # We can't easily detect this without calling it, which is risky.
                            # But if it's a sync function, LightRAG's priority_limit_async_func_call handles it:
                            # else: return func(*args, **kwargs)
                            # The caller then awaits the result if expected.

                            # Case 4: It's ALREADY a coroutine object (not a function). This is fatal.
                            elif asyncio.iscoroutine(current_func):
                                self.logger.critical(
                                    f"{vdb_name}.embedding_func is a COROUTINE OBJECT, not a callable! This will fail."
                                )
                                # Try to recover if possible (unlikely without re-instantiation)

                        # Diagnostic check for chunks_vdb (representative)
                        if hasattr(self.lightrag, "chunks_vdb") and hasattr(
                            self.lightrag.chunks_vdb, "embedding_func"
                        ):
                            current_func = self.lightrag.chunks_vdb.embedding_func

                            # IMPORTANT: Check if the embedding function is properly awaited
                            # LightRAG expects a sync-looking call that might return an awaitable.
                            # We inject a small diagnostic log here.
                            # Use a safer way to get name, handling functools.partial and other wrappers
                            func_name = getattr(current_func, "__name__", "unknown")
                            if func_name == "unknown" and hasattr(current_func, "func"):
                                func_name = getattr(
                                    current_func.func, "__name__", "unknown"
                                )
                            self.logger.debug(
                                _("Diagnostics: embedding_func name: {}").format(func_name)
                            )

                # Initialize processors after LightRAG is ready
                self._initialize_processors()

                self.logger.info(
                    "LightRAG, parse cache, and multimodal processors initialized"
                )
                return {"success": True}

            except Exception as e:
                # IMPORTANT: If initialization fails, we must not let this instance stay in a broken state
                # but 'coroutine' object is not callable usually means something was called as a function but was a coroutine
                # Let's try to get more info
                import traceback

                self.logger.error(_("Initialization traceback: {}").format(traceback.format_exc()))

                error_msg = _("Failed to initialize LightRAG instance: {}").format(str(e))
                self.logger.error(error_msg, exc_info=True)
                return {"success": False, "error": error_msg}

        except Exception as e:
            error_msg = _("Unexpected error during LightRAG initialization: {}").format(str(e))
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
                self.logger.error(_("Runtime error during storage finalization: {}").format(e))
        except Exception as e:
            self.logger.error(_("Error during storage finalization: {}").format(e))

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
            self.logger.info(_("Parser '{}' installation verified").format(self.config.parser))
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
                self.logger.debug(_("Set content source for {} processor").format(processor_name))
            except Exception as e:
                self.logger.error(
                    _("Failed to set content source for {}: {}").format(processor_name, e)
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
                self.logger.debug(_("Updated context config: {} = {}").format(key, value))
            else:
                self.logger.warning(_("Unknown context config parameter: {}").format(key))

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
                    _("New context configuration: {}").format(self._create_context_config())
                )
            except Exception as e:
                self.logger.error(_("Failed to update context configuration: {}").format(e))

    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        base_info = {
            "mineru_installed": MineruParser.check_installation(MineruParser()),
            "config": self.get_config_info(),
            "models": {
                "llm_model": (
                    "External function" if self.llm_model_func else "Not provided"
                ),
                "vision_model": (
                    "External function" if self.vision_model_func else "Not provided"
                ),
                "embedding_model": (
                    "External function" if self.embedding_func else "Not provided"
                ),
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

    def reload_config(self):
        """Reload configuration from source"""
        self.config = RAGAnythingConfig()
        self.working_dir = self.config.working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Re-initialize processors
        if self.lightrag:
            self._initialize_processors()
        self.logger.info("Configuration reloaded")

    async def get_system_stats(self):
        """Get system statistics"""
        total_docs = 0
        if self.lightrag and hasattr(self.lightrag, "doc_status_storage"):
            try:
                if hasattr(self.lightrag.doc_status_storage, "get_all"):
                    # Some storage backends return a dict or list
                    docs = await self.lightrag.doc_status_storage.get_all()
                    total_docs = len(docs)
                elif hasattr(self.lightrag.doc_status_storage, "__len__"):
                    total_docs = len(self.lightrag.doc_status_storage)
            except Exception:
                pass

        storage_usage = "0 B"
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.working_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)

            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if total_size < 1024:
                    storage_usage = _("{:.2f} {}").format(total_size, unit)
                    break
                total_size /= 1024
        except Exception:
            pass

        from dataclasses import make_dataclass

        Stats = make_dataclass(
            "Stats",
            [
                ("total_docs", int),
                ("queue_size", int),
                ("storage_usage", str),
                ("avg_processing_time", float),
            ],
        )
        return Stats(total_docs, 0, storage_usage, 0.0)

    async def delete_document(self, doc_id: str):
        """Delete a document by ID"""
        if not self.lightrag:
            raise RuntimeError(_("LightRAG not initialized"))

        # Ensure pipeline status is initialized before deletion
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()
            self.logger.debug("Pipeline status initialized for deletion")
        except Exception as e:
            self.logger.warning(_("Failed to initialize pipeline status: {}").format(e))

        # Use LightRAG's comprehensive document deletion method
        try:
            result = await self.lightrag.adelete_by_doc_id(doc_id)
            if result.status == "success":
                self.logger.info(
                    _("Successfully deleted document {}: {}").format(doc_id, result.message)
                )
            elif result.status == "not_found":
                self.logger.warning(_("Document {} not found for deletion").format(doc_id))
            else:
                self.logger.error(
                    _("Failed to delete document {}: {}").format(doc_id, result.message)
                )
            return result
        except Exception as e:
            self.logger.error(_("Error deleting document {}: {}").format(doc_id, e))
            raise

    async def cleanup_storage(self):
        """Cleanup temporary files"""
        deleted_count = 0
        # Clean up parser output dir if it exists
        parser_dir = self.config.parser_output_dir
        if os.path.exists(parser_dir):
            import shutil

            # Count files
            for _, _, files in os.walk(parser_dir):
                deleted_count += len(files)
            # Remove and recreate
            shutil.rmtree(parser_dir)
            os.makedirs(parser_dir, exist_ok=True)

        from dataclasses import make_dataclass

        CleanupResult = make_dataclass("CleanupResult", [("deleted_count", int)])
        return CleanupResult(deleted_count)
