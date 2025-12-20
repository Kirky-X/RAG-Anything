# Copyright (c) 2025 Kirky.X
# All rights reserved.

import os
import shutil
from pathlib import Path
from typing import Optional, Union

try:
    from modelscope import snapshot_download
    from modelscope.utils.constant import Tasks

    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

from raganything.config import RAGAnythingConfig
from raganything.i18n_logger import get_i18n_logger
from raganything.models.config import ModelInfo, default_models_config
from raganything.i18n import _


class ModelManager:
    """
    Manages model downloading, caching, and loading using ModelScope.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the ModelManager.

        Args:
            cache_dir (Optional[str]): Directory to cache models. Defaults to environment variable MODELSCOPE_CACHE or './models_cache'.
        """
        self.logger = get_i18n_logger()
        self.cache_dir = cache_dir or os.environ.get(
            "MODELSCOPE_CACHE", "./models_cache"
        )
        self._ensure_cache_dir()

        if not MODELSCOPE_AVAILABLE:
            self.logger.warning(
                "ModelScope is not installed. Model management features will be limited."
            )

    def _ensure_cache_dir(self):
        """Ensure the model cache directory exists."""

    def download_model(self, model_info: Union[str, ModelInfo]) -> str:
        """
        Download a model from ModelScope and return its local path.

        Args:
            model_info: ModelInfo object or model ID string

        Returns:
            str: Local path to the downloaded model
        """
        if not MODELSCOPE_AVAILABLE:
            raise ImportError(
                "ModelScope is required for downloading models. Please install it with `pip install modelscope`."
            )

        if isinstance(model_info, str):
            # If it's just an ID string, create a temporary ModelInfo
            model_info = ModelInfo(model_id=model_info)

        self.logger.info(_("Checking/Downloading model: {}...").format(model_info.model_id))

        try:
            model_path = snapshot_download(
                model_id=model_info.model_id,
                revision=model_info.revision,
                cache_dir=self.cache_dir,
            )
            self.logger.info(_("Model available at: {}").format(model_path))

            # Update the local path in the config object if provided
            if isinstance(model_info, ModelInfo):
                model_info.local_path = model_path

            return model_path
        except Exception as e:
            self.logger.error(_("Failed to download model {}: {}").format(model_info.model_id, e))
            raise

    def get_sense_voice_model_path(self) -> str:
        """Helper to get the SenseVoiceSmall model path.

        Returns:
            str: Local path to the SenseVoiceSmall model.
        """
        return self.download_model(default_models_config.sense_voice_small)


# Global instance
model_manager = ModelManager()
