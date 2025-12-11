# Copyright (c) 2025 Kirky.X
# All rights reserved.

def validate_provider(provider: str) -> bool:
    """Validate if the given provider is supported.

    Args:
        provider (str): The provider name to validate.

    Returns:
        bool: True if the provider is supported, False otherwise.
    """
    return (provider or "").lower() in {
        "openai",
        "openrouter",
        "azure-openai",
        "ollama",
        "mock",
        "offline",
    }


def ensure_non_empty(value: str) -> bool:
    """Check if a string value is non-empty after stripping whitespace.

    Args:
        value (str): The string value to check.

    Returns:
        bool: True if the value is non-empty, False otherwise.
    """
    return bool(value and str(value).strip())
