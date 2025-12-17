import os
import tomllib
from typing import Any, Dict

import tomli_w


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override dictionary into base dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def main():
    base_path = "env.example"
    override_path = "config.toml"
    output_path = "config.toml"

    if not os.path.exists(base_path):
        print(f"Error: {base_path} not found.")
        return

    # Load base config
    with open(base_path, "rb") as f:
        base_config = tomllib.load(f)

    # Load override config if exists
    override_config = {}
    if os.path.exists(override_path):
        with open(override_path, "rb") as f:
            override_config = tomllib.load(f)
        print(f"Loaded existing {override_path}")
    else:
        print(f"{override_path} not found, creating from {base_path}")

    # Merge
    final_config = merge_dicts(base_config, override_config)

    # Write back
    with open(output_path, "wb") as f:
        tomli_w.dump(final_config, f)

    print(f"Successfully merged {base_path} and {override_path} into {output_path}")


if __name__ == "__main__":
    main()
