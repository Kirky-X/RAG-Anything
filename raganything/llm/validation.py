def validate_provider(provider: str) -> bool:
    return (provider or "").lower() in {
        "openai",
        "openrouter",
        "azure-openai",
        "ollama",
        "mock",
    }


def ensure_non_empty(value: str) -> bool:
    return bool(value and str(value).strip())

