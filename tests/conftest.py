import os
import sys
import pytest
from dataclasses import dataclass
import numpy as np

# --- Patching lightrag.utils START ---
# The installed lightrag-hku package has a structure conflict (utils.py vs utils/ dir).
# We define the missing components here and inject them into lightrag.utils.

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

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    func: callable
    max_token_size: int | None = None  # deprecated keep it for compatible only

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

# Apply the patch
try:
    import lightrag.utils
    # Inject if missing
    if not hasattr(lightrag.utils, "get_env_value"):
        lightrag.utils.get_env_value = get_env_value
    if not hasattr(lightrag.utils, "EmbeddingFunc"):
        lightrag.utils.EmbeddingFunc = EmbeddingFunc
    
    # Also inject into sys.modules if needed, but patching the module object should suffice
    # for 'from lightrag.utils import get_env_value' if lightrag.utils is already loaded.
except ImportError:
    # If lightrag.utils cannot be imported at all, we might need to mock it or fail.
    # But usually it exists as a package.
    pass
# --- Patching lightrag.utils END ---

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Fixture to provide a temporary directory for storage tests."""
    d = tmp_path / "storage"
    d.mkdir()
    return str(d)
