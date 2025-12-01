from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ModelInfo:
    model_id: str
    revision: Optional[str] = None
    task_type: str = "auto"  # asr, tts, llm, etc.
    local_path: Optional[str] = None

@dataclass
class ModelsConfig:
    # Audio Models
    sense_voice_small: ModelInfo = field(
        default_factory=lambda: ModelInfo(
            model_id="iic/SenseVoiceSmall",
            task_type="asr"
        )
    )
    
    # Add more models here as needed
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by attribute name."""
        if hasattr(self, name):
            return getattr(self, name)
        return None

# Default configuration
default_models_config = ModelsConfig()
