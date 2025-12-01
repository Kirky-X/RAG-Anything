import os
from pathlib import Path
from typing import Optional, Union
import shutil

try:
    from modelscope import snapshot_download
    from modelscope.utils.constant import Tasks
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

from raganything.logger import logger
from raganything.models.config import ModelInfo, default_models_config
from raganything.config import RAGAnythingConfig

class ModelManager:
    """
    Manages model downloading, caching, and loading using ModelScope.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.environ.get("MODELSCOPE_CACHE", "./models_cache")
        self._ensure_cache_dir()
        
        if not MODELSCOPE_AVAILABLE:
            logger.warning("ModelScope is not installed. Model management features will be limited.")

    def _ensure_cache_dir(self):
        """Ensure the model cache directory exists."""
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def download_model(self, model_info: Union[str, ModelInfo]) -> str:
        """
        Download a model from ModelScope and return its local path.
        
        Args:
            model_info: ModelInfo object or model ID string
            
        Returns:
            str: Local path to the downloaded model
        """
        if not MODELSCOPE_AVAILABLE:
            raise ImportError("ModelScope is required for downloading models. Please install it with `pip install modelscope`.")

        if isinstance(model_info, str):
            # If it's just an ID string, create a temporary ModelInfo
            model_info = ModelInfo(model_id=model_info)

        logger.info(f"Checking/Downloading model: {model_info.model_id}...")
        
        try:
            model_path = snapshot_download(
                model_id=model_info.model_id,
                revision=model_info.revision,
                cache_dir=self.cache_dir
            )
            logger.info(f"Model available at: {model_path}")
            
            # Update the local path in the config object if provided
            if isinstance(model_info, ModelInfo):
                model_info.local_path = model_path
                
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model {model_info.model_id}: {e}")
            raise

    def get_sense_voice_model_path(self) -> str:
        """Helper to get the SenseVoiceSmall model path."""
        return self.download_model(default_models_config.sense_voice_small)

# Global instance
model_manager = ModelManager()
