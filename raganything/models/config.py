# Copyright (c) 2025 Kirky.X
# All rights reserved.

from dataclasses import dataclass, field
from typing import Dict, Optional
from raganything.i18n import _


@dataclass
class ModelInfo:
    """Data class representing model information."""

    model_id: str
    revision: Optional[str] = None
    task_type: str = "auto"  # asr, tts, llm, etc.
    local_path: Optional[str] = None


@dataclass
class ModelsConfig:
    """Configuration class for managing model information."""

    # Audio Models
    sense_voice_small: ModelInfo = field(
        default_factory=lambda: ModelInfo(
            model_id="iic/SenseVoiceSmall", task_type="asr"
        )
    )

    # Add more models here as needed

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by attribute name.

        Args:
            name (str): The name of the model attribute to retrieve.

        Returns:
            Optional[ModelInfo]: The ModelInfo object if found, None otherwise.
        """
        if hasattr(self, name):
            return getattr(self, name)
        return None


# Default configuration
default_models_config = ModelsConfig()
