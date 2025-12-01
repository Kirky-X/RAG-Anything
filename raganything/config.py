"""
Configuration classes for RAGAnything

Contains configuration dataclasses with environment variable support and optional TOML loading
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from lightrag.utils import get_env_value


def _import_toml_loader():
    try:
        import tomllib as _tl
        return _tl
    except Exception:
        try:
            import tomli as _tl  # type: ignore
            return _tl
        except Exception:
            return None


@dataclass
class DirectoryConfig:
    working_dir: str
    parser_output_dir: str


@dataclass
class ParsingConfig:
    parser: str
    parse_method: str
    display_content_stats: bool


@dataclass
class MultimodalConfig:
    enable_image_processing: bool
    enable_table_processing: bool
    enable_equation_processing: bool


@dataclass
class BatchConfig:
    max_concurrent_files: int
    supported_file_extensions: List[str]
    recursive_folder_processing: bool


@dataclass
class ContextSettings:
    context_window: int
    context_mode: str
    max_context_tokens: int
    include_headers: bool
    include_captions: bool
    context_filter_content_types: List[str]
    content_format: str


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_base: str
    api_key: str
    timeout: int
    max_retries: int


@dataclass
class EmbeddingConfig:
    provider: str
    model: str
    api_base: str
    api_key: str
    dim: int
    func_max_async: int
    batch_num: int


@dataclass
class VisionConfig:
    provider: str
    model: str
    api_base: str
    api_key: str
    timeout: int
    max_retries: int


@dataclass
class RAGAnythingConfig:
    directory: DirectoryConfig = field(
        default_factory=lambda: DirectoryConfig(
            working_dir=get_env_value("WORKING_DIR", "./rag_storage", str),
            parser_output_dir=get_env_value("OUTPUT_DIR", "./output", str),
        )
    )
    parsing: ParsingConfig = field(
        default_factory=lambda: ParsingConfig(
            parser=get_env_value("PARSER", "mineru", str),
            parse_method=get_env_value("PARSE_METHOD", "auto", str),
            display_content_stats=get_env_value("DISPLAY_CONTENT_STATS", True, bool),
        )
    )
    multimodal: MultimodalConfig = field(
        default_factory=lambda: MultimodalConfig(
            enable_image_processing=get_env_value("ENABLE_IMAGE_PROCESSING", True, bool),
            enable_table_processing=get_env_value("ENABLE_TABLE_PROCESSING", True, bool),
            enable_equation_processing=get_env_value("ENABLE_EQUATION_PROCESSING", True, bool),
        )
    )
    batch: BatchConfig = field(
        default_factory=lambda: BatchConfig(
            max_concurrent_files=get_env_value("MAX_CONCURRENT_FILES", 1, int),
            supported_file_extensions=get_env_value(
                "SUPPORTED_FILE_EXTENSIONS",
                ".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.gif,.webp,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md",
                str,
            ).split(","),
            recursive_folder_processing=get_env_value("RECURSIVE_FOLDER_PROCESSING", True, bool),
        )
    )
    context: ContextSettings = field(
        default_factory=lambda: ContextSettings(
            context_window=get_env_value("CONTEXT_WINDOW", 1, int),
            context_mode=get_env_value("CONTEXT_MODE", "page", str),
            max_context_tokens=get_env_value("MAX_CONTEXT_TOKENS", 2000, int),
            include_headers=get_env_value("INCLUDE_HEADERS", True, bool),
            include_captions=get_env_value("INCLUDE_CAPTIONS", True, bool),
            context_filter_content_types=get_env_value("CONTEXT_FILTER_CONTENT_TYPES", "text", str).split(","),
            content_format=get_env_value("CONTENT_FORMAT", "minerU", str),
        )
    )
    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            provider=get_env_value("LLM_PROVIDER", "openai", str),
            model=get_env_value("LLM_MODEL", "gpt-4o-mini", str),
            api_base=get_env_value("LLM_API_BASE", "", str),
            api_key=get_env_value("LLM_API_KEY", "", str),
            timeout=get_env_value("LLM_TIMEOUT", 60, int),
            max_retries=get_env_value("LLM_MAX_RETRIES", 2, int),
        )
    )
    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(
            provider=get_env_value("EMBEDDING_PROVIDER", "openai", str),
            model=get_env_value("EMBEDDING_MODEL", "text-embedding-3-small", str),
            api_base=get_env_value("EMBEDDING_API_BASE", "", str),
            api_key=get_env_value("EMBEDDING_API_KEY", "", str),
            dim=get_env_value("EMBEDDING_DIM", 1536, int),
            func_max_async=get_env_value("EMBEDDING_FUNC_MAX_ASYNC", 32, int),
            batch_num=get_env_value("EMBEDDING_BATCH_NUM", 16, int),
        )
    )
    vision: VisionConfig = field(
        default_factory=lambda: VisionConfig(
            provider=get_env_value("VISION_PROVIDER", "openai", str),
            model=get_env_value("VISION_MODEL", "gpt-4o-mini", str),
            api_base=get_env_value("VISION_API_BASE", "", str),
            api_key=get_env_value("VISION_API_KEY", "", str),
            timeout=get_env_value("VISION_TIMEOUT", 60, int),
            max_retries=get_env_value("VISION_MAX_RETRIES", 2, int),
        )
    )

    @dataclass
    class LoggingConfig:
        level: str = field(default=get_env_value("LOG_LEVEL", "INFO", str))
        verbose: bool = field(default=get_env_value("VERBOSE", False, bool))
        max_bytes: int = field(default=get_env_value("LOG_MAX_BYTES", 0, int))
        backup_count: int = field(default=get_env_value("LOG_BACKUP_COUNT", 0, int))
        dir: str = field(default=get_env_value("LOG_DIR", "", str))
        rotation: str = field(default="00:00")
        retention: str = field(default=get_env_value("RAG_LOG_RETENTION", "7 days", str))

    @dataclass
    class TiktokenConfig:
        cache_dir: str = field(default=get_env_value("TIKTOKEN_CACHE_DIR", "", str))

    @dataclass
    class QueryConfig:
        history_turns: int = field(default=3)
        cosine_threshold: float = field(default=0.2)
        top_k: int = field(default=60)
        max_token_text_chunk: int = field(default=4000)
        max_token_relation_desc: int = field(default=4000)
        max_token_entity_desc: int = field(default=4000)

    @dataclass
    class SummaryConfig:
        language: str = field(default="English")
        force_llm_summary_on_merge: int = field(default=6)
        max_token_summary: int = field(default=500)

    @dataclass
    class InsertConfig:
        max_parallel_insert: int = field(default=2)
        chunk_size: int = field(default=1200)
        chunk_overlap_size: int = field(default=100)

    @dataclass
    class ServerConfig:
        host: str = field(default="0.0.0.0")
        port: int = field(default=9621)
        workers: int = field(default=2)
        cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"])
        webui_title: str = field(default="My Graph KB")
        webui_description: str = field(default="Simple and Fast Graph Based RAG System")

    @dataclass
    class OllamaConfig:
        emulating_model_tag: str = field(default="latest")

    @dataclass
    class AuthConfig:
        accounts: str = field(default="")
        token_secret: str = field(default="")
        token_expire_hours: int = field(default=48)
        guest_token_expire_hours: int = field(default=24)
        jwt_algorithm: str = field(default="HS256")

    @dataclass
    class SSLConfig:
        enabled: bool = field(default=False)
        certfile: str = field(default="")
        keyfile: str = field(default="")

    @dataclass
    class APIConfig:
        lightrag_api_key: str = field(default="")
        whitelist_paths: List[str] = field(default_factory=lambda: ["/health", "/api/*"])

    @dataclass
    class RuntimeLLMConfig:
        enable_llm_cache: bool = field(default=True)
        enable_llm_cache_for_extract: bool = field(default=True)
        timeout: int = field(default=240)
        temperature: float = field(default=0.0)
        max_async: int = field(default=4)
        max_tokens: int = field(default=32768)
        binding: str = field(default="openai")
        binding_host: str = field(default="https://api.openai.com/v1")
        binding_api_key: str = field(default="")
        azure_openai_api_version: str = field(default="2024-08-01-preview")
        azure_openai_deployment: str = field(default="gpt-4o")

    @dataclass
    class StorageConfig:
        lightrag_kv_storage: str = field(default="")
        lightrag_vector_storage: str = field(default="")
        lightrag_doc_status_storage: str = field(default="")
        lightrag_graph_storage: str = field(default="")

    @dataclass
    class PostgresConfig:
        host: str = field(default="")
        port: int = field(default=5432)
        user: str = field(default="")
        password: str = field(default="")
        database: str = field(default="")
        max_connections: int = field(default=12)

    @dataclass
    class Neo4jConfig:
        uri: str = field(default="")
        username: str = field(default="")
        password: str = field(default="")

    @dataclass
    class MongoConfig:
        uri: str = field(default="")
        database: str = field(default="")

    @dataclass
    class MilvusConfig:
        uri: str = field(default="")
        db_name: str = field(default="")
        user: str = field(default="")
        password: str = field(default="")
        token: str = field(default="")

    @dataclass
    class QdrantConfig:
        url: str = field(default="")
        api_key: str = field(default="")

    @dataclass
    class RedisConfig:
        uri: str = field(default="")

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tiktoken: TiktokenConfig = field(default_factory=TiktokenConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    insert: InsertConfig = field(default_factory=InsertConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    api: APIConfig = field(default_factory=APIConfig)
    runtime_llm: RuntimeLLMConfig = field(default_factory=RuntimeLLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    mongo: MongoConfig = field(default_factory=MongoConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)

    def __post_init__(self):
        legacy_parse_method = get_env_value("MINERU_PARSE_METHOD", None, str)
        if legacy_parse_method and not get_env_value("PARSE_METHOD", None, str):
            self.parsing.parse_method = legacy_parse_method
            import warnings
            warnings.warn(
                "MINERU_PARSE_METHOD is deprecated. Use PARSE_METHOD instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        cfg_path_env = get_env_value("CONFIG_TOML", None, str)
        default_cfg_path = "config.toml"
        path = cfg_path_env or (default_cfg_path if _file_exists(default_cfg_path) else None)
        if path:
            self._load_from_toml(path)
        self._validate()

    def _load_from_toml(self, path: str):
        loader = _import_toml_loader()
        if loader is None:
            raise RuntimeError("TOML parsing requires Python 3.11+ or 'tomli' installed")
        with open(path, "rb") as f:
            data = loader.load(f)  # type: ignore
        self._merge_dict(data)

    def _merge_dict(self, cfg: Dict[str, Any]):
        def merge_dc(dc_obj, values: Dict[str, Any]):
            for k, v in (values or {}).items():
                if hasattr(dc_obj, k):
                    setattr(dc_obj, k, v)

        def normalize_context(values: Dict[str, Any]) -> Dict[str, Any]:
            v = dict(values or {})
            c = v.get("context_filter_content_types")
            if isinstance(c, str):
                v["context_filter_content_types"] = c.split(",")
            return v

        def normalize_embedding(values: Dict[str, Any]) -> Dict[str, Any]:
            v = dict(values or {})
            d = v.get("dim")
            if isinstance(d, str):
                try:
                    v["dim"] = int(d)
                except Exception:
                    pass
            return v

        nested = cfg.get("raganything") or {}

        if "directory" in cfg:
            merge_dc(self.directory, cfg.get("directory"))
        # Merge nested first
        if "parsing" in nested:
            merge_dc(self.parsing, nested.get("parsing"))
        if "multimodal" in nested:
            merge_dc(self.multimodal, nested.get("multimodal"))
        if "batch" in nested:
            merge_dc(self.batch, nested.get("batch"))
        if "context" in nested:
            merge_dc(self.context, normalize_context(nested.get("context")))
        if "llm" in nested:
            merge_dc(self.llm, nested.get("llm"))
        if "embedding" in nested:
            merge_dc(self.embedding, normalize_embedding(nested.get("embedding")))
        if "vision" in nested:
            merge_dc(self.vision, nested.get("vision"))

        # Then allow top-level to override
        if "parsing" in cfg:
            merge_dc(self.parsing, cfg.get("parsing"))
        if "multimodal" in cfg:
            merge_dc(self.multimodal, cfg.get("multimodal"))
        if "batch" in cfg:
            merge_dc(self.batch, cfg.get("batch"))
        if "context" in cfg:
            merge_dc(self.context, normalize_context(cfg.get("context")))
        if "llm" in cfg:
            merge_dc(self.llm, cfg.get("llm"))
        if "embedding" in cfg:
            merge_dc(self.embedding, normalize_embedding(cfg.get("embedding")))
        if "vision" in cfg:
            merge_dc(self.vision, cfg.get("vision"))
        if "logging" in cfg:
            merge_dc(self.logging, cfg.get("logging"))
        if "tiktoken" in cfg:
            merge_dc(self.tiktoken, cfg.get("tiktoken"))
        if "query" in cfg:
            merge_dc(self.query, cfg.get("query"))
        if "summary" in cfg:
            merge_dc(self.summary, cfg.get("summary"))
        if "insert" in cfg:
            merge_dc(self.insert, cfg.get("insert"))
        if "server" in cfg:
            merge_dc(self.server, cfg.get("server"))
        if "ollama" in cfg:
            merge_dc(self.ollama, cfg.get("ollama"))
        if "auth" in cfg:
            merge_dc(self.auth, cfg.get("auth"))
        if "ssl" in cfg:
            merge_dc(self.ssl, cfg.get("ssl"))
        if "api" in cfg:
            merge_dc(self.api, cfg.get("api"))
        if "runtime.llm" in cfg:
            merge_dc(self.runtime_llm, cfg.get("runtime.llm"))
        if "storage" in cfg:
            merge_dc(self.storage, cfg.get("storage"))
        if "postgres" in cfg:
            merge_dc(self.postgres, cfg.get("postgres"))
        if "neo4j" in cfg:
            merge_dc(self.neo4j, cfg.get("neo4j"))
        if "mongo" in cfg:
            merge_dc(self.mongo, cfg.get("mongo"))
        if "milvus" in cfg:
            merge_dc(self.milvus, cfg.get("milvus"))
        if "qdrant" in cfg:
            merge_dc(self.qdrant, cfg.get("qdrant"))
        if "redis" in cfg:
            merge_dc(self.redis, cfg.get("redis"))

    def _validate(self):
        valid_parsers = {"mineru", "docling"}
        if self.parsing.parser not in valid_parsers:
            raise ValueError("Invalid parser")
        valid_parse_methods = {"auto", "ocr", "txt"}
        if self.parsing.parse_method not in valid_parse_methods:
            raise ValueError("Invalid parse_method")
        valid_context_modes = {"page", "chunk"}
        if self.context.context_mode not in valid_context_modes:
            raise ValueError("Invalid context_mode")
        if self.batch.max_concurrent_files < 1:
            raise ValueError("max_concurrent_files must be >= 1")
        if self.context.context_window < 0:
            raise ValueError("context_window must be >= 0")
        if self.context.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be > 0")
        for t in [self.llm.timeout, self.vision.timeout]:
            if t <= 0:
                raise ValueError("timeout must be > 0")
        for r in [self.llm.max_retries, self.vision.max_retries]:
            if r < 0:
                raise ValueError("max_retries must be >= 0")
        if self.logging.max_bytes < 0:
            raise ValueError("logging.max_bytes must be >= 0")
        if self.logging.backup_count < 0:
            raise ValueError("logging.backup_count must be >= 0")

    @property
    def working_dir(self) -> str:
        return self.directory.working_dir

    @working_dir.setter
    def working_dir(self, value: str):
        self.directory.working_dir = value

    @property
    def parser_output_dir(self) -> str:
        return self.directory.parser_output_dir

    @parser_output_dir.setter
    def parser_output_dir(self, value: str):
        self.directory.parser_output_dir = value

    @property
    def parser(self) -> str:
        return self.parsing.parser

    @parser.setter
    def parser(self, value: str):
        self.parsing.parser = value

    @property
    def parse_method(self) -> str:
        return self.parsing.parse_method

    @parse_method.setter
    def parse_method(self, value: str):
        self.parsing.parse_method = value

    @property
    def display_content_stats(self) -> bool:
        return self.parsing.display_content_stats

    @display_content_stats.setter
    def display_content_stats(self, value: bool):
        self.parsing.display_content_stats = value

    @property
    def enable_image_processing(self) -> bool:
        return self.multimodal.enable_image_processing

    @enable_image_processing.setter
    def enable_image_processing(self, value: bool):
        self.multimodal.enable_image_processing = value

    @property
    def enable_table_processing(self) -> bool:
        return self.multimodal.enable_table_processing

    @enable_table_processing.setter
    def enable_table_processing(self, value: bool):
        self.multimodal.enable_table_processing = value

    @property
    def enable_equation_processing(self) -> bool:
        return self.multimodal.enable_equation_processing

    @enable_equation_processing.setter
    def enable_equation_processing(self, value: bool):
        self.multimodal.enable_equation_processing = value

    @property
    def max_concurrent_files(self) -> int:
        return self.batch.max_concurrent_files

    @max_concurrent_files.setter
    def max_concurrent_files(self, value: int):
        self.batch.max_concurrent_files = value

    @property
    def supported_file_extensions(self) -> List[str]:
        return self.batch.supported_file_extensions

    @supported_file_extensions.setter
    def supported_file_extensions(self, value: List[str]):
        self.batch.supported_file_extensions = value

    @property
    def recursive_folder_processing(self) -> bool:
        return self.batch.recursive_folder_processing

    @recursive_folder_processing.setter
    def recursive_folder_processing(self, value: bool):
        self.batch.recursive_folder_processing = value

    @property
    def context_window(self) -> int:
        return self.context.context_window

    @context_window.setter
    def context_window(self, value: int):
        self.context.context_window = value

    @property
    def context_mode(self) -> str:
        return self.context.context_mode

    @context_mode.setter
    def context_mode(self, value: str):
        self.context.context_mode = value

    @property
    def max_context_tokens(self) -> int:
        return self.context.max_context_tokens

    @max_context_tokens.setter
    def max_context_tokens(self, value: int):
        self.context.max_context_tokens = value

    @property
    def include_headers(self) -> bool:
        return self.context.include_headers

    @include_headers.setter
    def include_headers(self, value: bool):
        self.context.include_headers = value

    @property
    def include_captions(self) -> bool:
        return self.context.include_captions

    @include_captions.setter
    def include_captions(self, value: bool):
        self.context.include_captions = value

    @property
    def context_filter_content_types(self) -> List[str]:
        return self.context.context_filter_content_types

    @context_filter_content_types.setter
    def context_filter_content_types(self, value: List[str]):
        self.context.context_filter_content_types = value

    @property
    def content_format(self) -> str:
        return self.context.content_format

    @content_format.setter
    def content_format(self, value: str):
        self.context.content_format = value

    @property
    def llm_provider(self) -> str:
        return self.llm.provider

    @llm_provider.setter
    def llm_provider(self, value: str):
        self.llm.provider = value

    @property
    def llm_model(self) -> str:
        return self.llm.model

    @llm_model.setter
    def llm_model(self, value: str):
        self.llm.model = value

    @property
    def llm_api_base(self) -> str:
        return self.llm.api_base

    @llm_api_base.setter
    def llm_api_base(self, value: str):
        self.llm.api_base = value

    @property
    def llm_api_key(self) -> str:
        return self.llm.api_key

    @llm_api_key.setter
    def llm_api_key(self, value: str):
        self.llm.api_key = value

    @property
    def llm_timeout(self) -> int:
        return self.llm.timeout

    @llm_timeout.setter
    def llm_timeout(self, value: int):
        self.llm.timeout = value

    @property
    def llm_max_retries(self) -> int:
        return self.llm.max_retries

    @llm_max_retries.setter
    def llm_max_retries(self, value: int):
        self.llm.max_retries = value

    @property
    def embedding_provider(self) -> str:
        return self.embedding.provider

    @embedding_provider.setter
    def embedding_provider(self, value: str):
        self.embedding.provider = value

    @property
    def embedding_model(self) -> str:
        return self.embedding.model

    @embedding_model.setter
    def embedding_model(self, value: str):
        self.embedding.model = value

    @property
    def embedding_api_base(self) -> str:
        return self.embedding.api_base

    @embedding_api_base.setter
    def embedding_api_base(self, value: str):
        self.embedding.api_base = value

    @property
    def embedding_api_key(self) -> str:
        return self.embedding.api_key

    @embedding_api_key.setter
    def embedding_api_key(self, value: str):
        self.embedding.api_key = value

    @property
    def embedding_dim(self) -> int:
        return self.embedding.dim

    @embedding_dim.setter
    def embedding_dim(self, value: int):
        self.embedding.dim = value

    @property
    def embedding_func_max_async(self) -> int:
        return self.embedding.func_max_async

    @embedding_func_max_async.setter
    def embedding_func_max_async(self, value: int):
        self.embedding.func_max_async = value

    @property
    def embedding_batch_num(self) -> int:
        return self.embedding.batch_num

    @embedding_batch_num.setter
    def embedding_batch_num(self, value: int):
        self.embedding.batch_num = value

    @property
    def vision_provider(self) -> str:
        return self.vision.provider

    @vision_provider.setter
    def vision_provider(self, value: str):
        self.vision.provider = value

    @property
    def vision_model(self) -> str:
        return self.vision.model

    @vision_model.setter
    def vision_model(self, value: str):
        self.vision.model = value

    @property
    def vision_api_base(self) -> str:
        return self.vision.api_base

    @vision_api_base.setter
    def vision_api_base(self, value: str):
        self.vision.api_base = value

    @property
    def vision_api_key(self) -> str:
        return self.vision.api_key

    @vision_api_key.setter
    def vision_api_key(self, value: str):
        self.vision.api_key = value

    @property
    def vision_timeout(self) -> int:
        return self.vision.timeout

    @vision_timeout.setter
    def vision_timeout(self, value: int):
        self.vision.timeout = value

    @property
    def vision_max_retries(self) -> int:
        return self.vision.max_retries

    @vision_max_retries.setter
    def vision_max_retries(self, value: int):
        self.vision.max_retries = value

    @property
    def mineru_parse_method(self) -> str:
        import warnings
        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parsing.parse_method

    @mineru_parse_method.setter
    def mineru_parse_method(self, value: str):
        import warnings
        warnings.warn(
            "mineru_parse_method is deprecated. Use parse_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.parsing.parse_method = value


def _file_exists(path: str) -> bool:
    try:
        import os
        return os.path.exists(path)
    except Exception:
        return False
