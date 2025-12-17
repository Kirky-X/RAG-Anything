# Copyright (c) 2025 Kirky.X
# All rights reserved.

import sys
import importlib.util
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Type, TypeVar, List, Tuple
import re
import os
import json
from raganything.logger import logger
from types import ModuleType
import uuid
import numpy as np

# --- Re-implementation of missing utils functions ---



def get_env_value(
    env_key: str, default: any, value_type: type = str, special_none: bool = False
) -> any:
    """
    Get value from environment variable with type conversion
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    # Handle special case for "None" string
    if special_none and value == "None":
        return None

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")

    # Handle list type with JSON parsing
    if value_type is list:
        try:
            import json

            parsed_value = json.loads(value)
            # Ensure the parsed value is actually a list
            if isinstance(parsed_value, list):
                return parsed_value
            else:
                return default
        except (json.JSONDecodeError, ValueError):
            return default

    try:
        return value_type(value)
    except (ValueError, TypeError):
        return default


def compute_args_hash(*args: Any) -> str:
    from hashlib import md5
    # Convert all arguments to strings and join them
    args_str = "".join([str(arg) for arg in args])
    try:
        return md5(args_str.encode("utf-8")).hexdigest()
    except UnicodeEncodeError:
        safe_bytes = args_str.encode("utf-8", errors="replace")
        return md5(safe_bytes).hexdigest()

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + compute_args_hash(content)

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    func: Callable[[List[str]], Any]
    max_token_size: Optional[int] = None

    async def __call__(self, *args, **kwargs) -> Any:
        # Check if the internal func is async or returns a coroutine
        if asyncio.iscoroutinefunction(self.func):
            result = await self.func(*args, **kwargs)
        else:
            result = self.func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
        
        # LightRAG expects a numpy array, but our internal embedding function might return a list of lists or a tensor.
        # We need to ensure it's in the format LightRAG expects if possible, but for now we just return the result
        # and let the consumer handle it, or we could add conversion logic here if we knew the expected output type.
        # Based on typical usage, it likely expects a numpy array.
        import numpy as np
        if isinstance(result, list):
             # If it's a list of lists (embeddings), convert to numpy array
             if result and isinstance(result[0], list):
                 result = np.array(result)
             # If it's a list of floats (single embedding), convert to numpy array
             elif result and isinstance(result[0], (float, int)):
                 result = np.array(result)

        return result

def is_float_regex(value: str) -> bool:
    """Check if string is a float using regex"""
    return bool(re.match(r'^-?\d+(?:\.\d+)$', str(value)))

def sanitize_and_normalize_extracted_text(text: str) -> str:
    """Sanitize and normalize extracted text"""
    if not text:
        return ""
    # Basic normalization: strip and remove excessive whitespace
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def pack_user_ass_to_openai_messages(prompt: str, system_prompt: Optional[str] = None, history_messages: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Pack user prompt and assistant messages to OpenAI format"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    if history_messages:
        messages.extend(history_messages)
        
    messages.append({"role": "user", "content": prompt})
    return messages

def split_string_by_multi_markers(content: str, markers: List[str]) -> List[str]:
    """Split string by multiple markers"""
    if not markers:
        return [content]
    results = [content]
    for marker in markers:
        new_results = []
        for item in results:
            new_results.extend(item.split(marker))
        results = new_results
    return [r for r in results if r]

def truncate_list_by_token_size(list_data: List[Any], key: Callable[[Any], str], max_token_size: int) -> List[Any]:
    """Truncate a list of data by token size (Stub)"""
    return list_data

async def handle_cache(func, args, kwargs):
    """Handle cache (Stub)"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)

async def save_to_cache(key, value):
    """Save to cache (Stub)"""
    pass

@dataclass
class CacheData:
    args_hash: str
    content: Any

async def use_llm_func_with_cache(
    user_prompt: str,
    use_llm_func: Callable,
    llm_response_cache: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    max_tokens: Optional[int] = None,
    history_messages: Optional[List[Dict[str, str]]] = None,
    cache_type: str = 'extract',
    chunk_id: Optional[str] = None,
    cache_keys_collector: Optional[List] = None,
    **kwargs: Any
) -> Any:
    """Use LLM function with cache (Stub)"""
    # Simply call the function ignoring cache logic for now, or minimal pass-through
    # The actual implementation in LightRAG is complex, here we just need signature compatibility
    
    # Inject system_prompt and history_messages into kwargs if provided, 
    # as the underlying func might expect them
    if system_prompt is not None:
        kwargs['system_prompt'] = system_prompt
    if history_messages is not None:
        kwargs['history_messages'] = history_messages
    if max_tokens is not None:
        kwargs['max_tokens'] = max_tokens

    if asyncio.iscoroutinefunction(use_llm_func):
        result = await use_llm_func(user_prompt, **kwargs)
    else:
        result = use_llm_func(user_prompt, **kwargs)
        if asyncio.iscoroutine(result):
             result = await result

    # Return result and a dummy timestamp/metadata if expected
    # LightRAG often expects (result, timestamp/metadata) from this function
    return result, None

async def update_chunk_cache_list(key, value):
    """Update chunk cache list (Stub)"""
    pass

def remove_think_tags(text: str) -> str:
    """Remove <think> tags from text (Stub)"""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def pick_by_weighted_polling(items: List[Any], weights: List[float], count: int) -> List[Any]:
    """Pick items by weighted polling (Stub)"""
    if not items or count <= 0:
        return []
    return items[:count]

async def pick_by_vector_similarity(
    query_vectors: Any,
    candidate_vectors: Any,
    candidate_items: List[Any],
    top_k: int,
    similarity_threshold: float,
) -> List[Any]:
    """Pick items by vector similarity (Stub)"""
    return candidate_items[:top_k]

async def process_chunks_unified(
    process_name: str,
    chunks: List[Any],
    func: Callable,
    use_llm_func: Callable,
    batch_size: int,
) -> List[Any]:
    """Process chunks unified (Stub)"""
    # Simply iterate and call func for each chunk
    results = []
    for chunk in chunks:
        if asyncio.iscoroutinefunction(func):
            res = await func(chunk)
        else:
            res = func(chunk)
            if asyncio.iscoroutine(res):
                res = await res
        results.append(res)
    return results

async def safe_vdb_operation_with_exception(func, *args, **kwargs):
    """Safe vector database operation with exception (Stub)"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return func(*args, **kwargs)

def create_prefixed_exception(exc: Exception, prefix: str) -> Exception:
    """Create a new exception with a prefixed message (Stub)"""
    try:
        # Try to create with message first
        return type(exc)(f"{prefix}: {str(exc)}")
    except TypeError:
        # If that fails, try to create with no arguments and set args manually
        new_exc = type(exc)()
        new_exc.args = (f"{prefix}: {str(exc)}",)
        return new_exc

def fix_tuple_delimiter_corruption(content: str) -> str:
    """Fix tuple delimiter corruption (Stub)"""
    return content

def convert_to_user_format(content: str) -> str:
    """Convert to user format (Stub)"""
    return content

def generate_reference_list_from_chunks(chunks: List[Any]) -> str:
    """Generate reference list from chunks (Stub)"""
    return "references"

def apply_source_ids_limit(source_ids: List[str], limit: int) -> List[str]:
    """Apply source IDs limit (Stub)"""
    return source_ids[:limit]

def merge_source_ids(source_ids: List[str]) -> str:
    """Merge source IDs (Stub)"""
    return ",".join(source_ids) if source_ids else ""

def make_relation_chunk_key(chunk_key: str) -> str:
    """Make relation chunk key (Stub)"""
    return f"rel_{chunk_key}"

@dataclass
class TiktokenTokenizer:
    """Tiktoken tokenizer (Stub)"""
    model_name: str
    
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]
        
    def decode(self, tokens: list[int]) -> str:
        return "".join([chr(t) for t in tokens])

def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop (Stub)"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def lazy_external_import(module_name: str, class_name: str):
    """Lazy import wrapper (Stub)"""
    class Dummy:
        pass
    return Dummy

def priority_limit_async_func_call(max_async: int, llm_timeout: int, queue_name: str = "func"):
    """
    Priority limit async function call (Stub - Decorator Factory)
    Matches usage in LightRAG:
        self.embedding_func = priority_limit_async_func_call(
            self.embedding_func_max_async,
            llm_timeout=self.default_embedding_timeout,
            queue_name="Embedding func",
        )(self.embedding_func)
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            elif callable(func) and asyncio.iscoroutinefunction(func.__call__): # Handle callable objects that are async
                 result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
            
            return result
        
        # Copy attributes from original function if they exist
        if hasattr(func, "embedding_dim"):
            wrapper.embedding_dim = func.embedding_dim
        if hasattr(func, "max_token_size"):
            wrapper.max_token_size = func.max_token_size
            
        return wrapper
    return decorator

def get_content_summary(content: str, max_length: int = 100) -> str:
    """Get content summary (Stub)"""
    if not content:
        return ""
    return content[:max_length] + "..." if len(content) > max_length else content

@dataclass
class Tokenizer:
    model_name: str
    
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]
        
    def decode(self, tokens: list[int]) -> str:
        return "".join([chr(t) for t in tokens])


def sanitize_text_for_encoding(text: Any) -> str:
    if text is None:
        return ""
    return str(text)


def check_storage_env_vars(
    storage_name: str | None = None, env_vars: List[str] | None = None
) -> None:
    return


def generate_track_id(*args, **kwargs) -> str:
    """Generate track ID (UUID4)"""
    return str(uuid.uuid4())


def subtract_source_ids(source_ids: List[str], remove_ids: List[str]) -> List[str]:
    remove_set = set(remove_ids)
    return [sid for sid in source_ids if sid not in remove_set]


def normalize_source_ids_limit_method(method: str) -> str:
    return method


logger = logger


VERBOSE_DEBUG = os.getenv("VERBOSE", "false").lower() == "true"


def verbose_debug(msg: str, *args, **kwargs):
    if VERBOSE_DEBUG:
        logger.debug(msg, *args, **kwargs)
    else:
        if args:
            formatted_msg = msg % args
        else:
            formatted_msg = msg
        truncated_msg = (
            formatted_msg[:150] + "..." if len(formatted_msg) > 150 else formatted_msg
        )
        truncated_msg = re.sub(r"\n+", "\n", truncated_msg)
        logger.debug(truncated_msg, **kwargs)


def set_verbose_debug(enabled: bool):
    global VERBOSE_DEBUG
    VERBOSE_DEBUG = enabled


def wrap_embedding_func_with_attrs(*, embedding_dim: int, **extra_attrs):
    def decorator(func: Callable[[List[str]], Any]) -> EmbeddingFunc:
        ef = EmbeddingFunc(embedding_dim=embedding_dim, func=func)
        for name, value in extra_attrs.items():
            setattr(ef, name, value)
        return ef

    return decorator


def safe_unicode_decode(data: Any) -> str:
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("utf-8", errors="ignore")
    return str(data)

def write_json(data: Any, path: str):
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(path: str) -> Any:
    """Read data from a JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_json(path: str) -> Any:
    """Alias for read_json"""
    return read_json(path)

def get_pinyin_sort_key(s: str) -> str:
    """Get pinyin sort key (Stub)"""
    return s

# --- Patching Logic ---

def generate_cache_key(*args: Any) -> str:
    """Generate cache key using md5 hash of args"""
    from hashlib import md5
    args_str = "".join([str(arg) for arg in args])
    return md5(args_str.encode("utf-8")).hexdigest()

async def process_chunks_unified(
    process_name: str,
    chunks: List[Any],
    func: Callable,
    use_llm_func: Callable,
    batch_size: int,
) -> List[Any]:
    """Process chunks unified (Stub)"""
    results = []
    for chunk in chunks:
        if asyncio.iscoroutinefunction(func):
            res = await func(chunk)
        else:
            res = func(chunk)
            if asyncio.iscoroutine(res):
                res = await res
        results.append(res)
    return results

async def extract_entities(
    chunks: Dict[str, Any],
    global_config: Dict[str, Any],
    pipeline_status: Any,
    pipeline_status_lock: Any,
    llm_response_cache: Any,
    text_chunks_storage: Any,
) -> List[Tuple]:
    """Patch for LightRAG's extract_entities to handle coroutine issues"""
    try:
        # Import the original function from lightrag.operate
        from lightrag.operate import extract_entities as original_extract_entities
        
        # Call the original function
        result = await original_extract_entities(
            chunks=chunks,
            global_config=global_config,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=llm_response_cache,
            text_chunks_storage=text_chunks_storage,
        )
        
        # Ensure the result is not a coroutine
        if asyncio.iscoroutine(result):
            result = await result
            
        return result
        
    except Exception as e:
        # Fallback: return empty results for each chunk
        logger.warning(f"Error in extract_entities patch, using fallback: {e}")
        
        # Return minimal structure to prevent downstream errors
        results = []
        for chunk_id in chunks.keys():
            nodes = {f"fallback_entity_{chunk_id}": [{"source_id": chunk_id}]}
            edges = {}
            results.append((nodes, edges))
        
        return results

def patch_lightrag():
    if 'lightrag.utils' in sys.modules:
        module = sys.modules['lightrag.utils']
        if hasattr(module, 'get_env_value') and hasattr(module, 'verbose_debug') and hasattr(module, 'get_pinyin_sort_key'):
            return

    # print("DEBUG: Attempting to patch lightrag.utils...")

    # Find lightrag package path
    lightrag_path = None
    try:
        spec = importlib.util.find_spec("lightrag")
        if spec and spec.submodule_search_locations:
            lightrag_path = spec.submodule_search_locations[0]
    except ImportError:
        pass
    
    if not lightrag_path:
        lightrag_path = "/home/dev/.local/lib/python3.12/site-packages/lightrag"

    utils_init_path = os.path.join(lightrag_path, "utils", "__init__.py")
    
    if not os.path.exists(utils_init_path):
        print(f"DEBUG: Could not find lightrag/utils/__init__.py at {utils_init_path}")
        return

    # Create dummy lightrag module if needed to satisfy parent check during utils loading
    created_dummy = False
    if 'lightrag' not in sys.modules:
        dummy_lightrag = ModuleType('lightrag')
        dummy_lightrag.__path__ = [lightrag_path]
        sys.modules['lightrag'] = dummy_lightrag
        created_dummy = True

    # Manually load lightrag.utils
    try:
        spec = importlib.util.spec_from_file_location("lightrag.utils", utils_init_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["lightrag.utils"] = module
            spec.loader.exec_module(module)
            # print("DEBUG: Manually loaded lightrag.utils")
            
            # Inject functions
            module.get_env_value = get_env_value
            module.compute_mdhash_id = compute_mdhash_id
            module.EmbeddingFunc = EmbeddingFunc
            module.Tokenizer = Tokenizer
            module.TiktokenTokenizer = TiktokenTokenizer
            module.is_float_regex = is_float_regex
            module.sanitize_and_normalize_extracted_text = sanitize_and_normalize_extracted_text
            module.pack_user_ass_to_openai_messages = pack_user_ass_to_openai_messages
            module.split_string_by_multi_markers = split_string_by_multi_markers
            module.truncate_list_by_token_size = truncate_list_by_token_size
            module.compute_args_hash = compute_args_hash
            module.handle_cache = handle_cache
            module.save_to_cache = save_to_cache
            module.CacheData = CacheData
            module.use_llm_func_with_cache = use_llm_func_with_cache
            module.update_chunk_cache_list = update_chunk_cache_list
            module.remove_think_tags = remove_think_tags
            module.pick_by_weighted_polling = pick_by_weighted_polling
            module.pick_by_vector_similarity = pick_by_vector_similarity
            module.process_chunks_unified = process_chunks_unified
            module.extract_entities = extract_entities
            module.safe_vdb_operation_with_exception = safe_vdb_operation_with_exception
            module.create_prefixed_exception = create_prefixed_exception
            module.fix_tuple_delimiter_corruption = fix_tuple_delimiter_corruption
            module.convert_to_user_format = convert_to_user_format
            module.generate_reference_list_from_chunks = generate_reference_list_from_chunks
            module.apply_source_ids_limit = apply_source_ids_limit
            module.merge_source_ids = merge_source_ids
            module.make_relation_chunk_key = make_relation_chunk_key
            module.always_get_an_event_loop = always_get_an_event_loop
            module.lazy_external_import = lazy_external_import
            module.priority_limit_async_func_call = priority_limit_async_func_call
            module.get_content_summary = get_content_summary
            module.sanitize_text_for_encoding = sanitize_text_for_encoding
            module.check_storage_env_vars = check_storage_env_vars
            module.generate_track_id = generate_track_id
            module.subtract_source_ids = subtract_source_ids
            module.normalize_source_ids_limit_method = normalize_source_ids_limit_method
            module.logger = logger
            module.VERBOSE_DEBUG = VERBOSE_DEBUG
            module.verbose_debug = verbose_debug
            module.set_verbose_debug = set_verbose_debug
            module.wrap_embedding_func_with_attrs = wrap_embedding_func_with_attrs
            module.safe_unicode_decode = safe_unicode_decode
            module.write_json = write_json
            module.read_json = read_json
            module.load_json = load_json
            module.get_pinyin_sort_key = get_pinyin_sort_key
            module.generate_cache_key = generate_cache_key

            # Update __all__
            if hasattr(module, '__all__'):
                functions_to_add = [
                    'get_env_value', 'compute_mdhash_id', 'EmbeddingFunc', 'Tokenizer', 'TiktokenTokenizer',
                    'is_float_regex', 'sanitize_and_normalize_extracted_text', 'pack_user_ass_to_openai_messages',
                    'split_string_by_multi_markers', 'truncate_list_by_token_size', 'compute_args_hash',
                    'handle_cache', 'save_to_cache', 'CacheData', 'use_llm_func_with_cache', 'update_chunk_cache_list',
                    'remove_think_tags', 'pick_by_weighted_polling', 'pick_by_vector_similarity',
                    'process_chunks_unified', 'extract_entities', 'safe_vdb_operation_with_exception', 'create_prefixed_exception',
                    'fix_tuple_delimiter_corruption', 'convert_to_user_format', 'generate_reference_list_from_chunks',
                    'apply_source_ids_limit', 'merge_source_ids', 'make_relation_chunk_key', 'always_get_an_event_loop',
                    'lazy_external_import', 'priority_limit_async_func_call', 'get_content_summary',
                    'sanitize_text_for_encoding', 'check_storage_env_vars', 'generate_track_id',
                    'subtract_source_ids', 'normalize_source_ids_limit_method', 'logger',
                    'VERBOSE_DEBUG', 'verbose_debug', 'set_verbose_debug',
                    'wrap_embedding_func_with_attrs', 'safe_unicode_decode',
                    'write_json', 'read_json', 'load_json', 'get_pinyin_sort_key', 'generate_cache_key'
                ]
                for func_name in functions_to_add:
                    if func_name not in module.__all__:
                        module.__all__.append(func_name)
            
            # print("DEBUG: Injected functions into lightrag.utils")

    except Exception as e:
        print(f"DEBUG: Failed to patch lightrag.utils: {e}")

    # Remove dummy lightrag if we created it, to allow real import later
    if created_dummy:
        del sys.modules['lightrag']
        # print("DEBUG: Removed dummy lightrag module")

# Execute patch
patch_lightrag()
