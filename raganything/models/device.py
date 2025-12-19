# Copyright (c) 2025 Kirky.X
# All rights reserved.

import platform
from typing import Optional

import psutil

# Try to import torch, but don't fail if it's not installed yet
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from raganything.logger import logger


class DeviceManager:
    """
    Singleton class for managing compute resources (CPU/GPU).
    Automatically detects available hardware and prioritizes GPU.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the DeviceManager singleton instance."""
        if self._initialized:
            return
        self._device = self._detect_device()
        self._log_resource_status()
        self._initialized = True

    def _detect_device(self) -> str:
        """
        Detect the best available device.
        Priority: CUDA > MPS (Apple Silicon) > CPU

        Returns:
            str: The detected device name (cuda, mps, or cpu).
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not found. Defaulting to CPU.")
            return "cpu"

        if torch.cuda.is_available():
            try:
                # Verify we can actually use the GPU
                _ = torch.tensor([1.0]).cuda()
                return "cuda"
            except Exception as e:
                logger.warning(
                    f"CUDA available but failed to initialize: {e}. Falling back to CPU."
                )

        if torch.backends.mps.is_available():
            try:
                # Verify MPS
                _ = torch.tensor([1.0]).to("mps")
                return "mps"
            except Exception as e:
                logger.warning(
                    f"MPS available but failed to initialize: {e}. Falling back to CPU."
                )

        return "cpu"

    def is_gpu_available(self) -> bool:
        """Check if a GPU (CUDA or MPS) is currently available and selected.

        Returns:
            bool: True if a GPU is available and selected, False otherwise.
        """
        return self._device in ("cuda", "mps")

    def get_available_memory(self) -> float:
        """
        Get available system memory in GB.
        Useful for making decisions about loading large models.

        Returns:
            float: Available system memory in GB.
        """
        memory = psutil.virtual_memory()
        return memory.available / (1024**3)

    def _log_resource_status(self):
        """Log current resource usage and device info."""
        logger.info("=" * 30)
        logger.info("Compute Resource Status")
        logger.info("-" * 30)
        logger.info(f"Selected Device: {self._device.upper()}")

        # CPU Info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        logger.info(f"CPU Usage: {cpu_percent}%")
        logger.info(
            f"RAM Usage: {memory.percent}% ({memory.used / (1024**3):.2f}GB / {memory.total / (1024**3):.2f}GB)"
        )

        # GPU Info
        if self._device == "cuda" and TORCH_AVAILABLE:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024**2)
                gpu_mem_res = torch.cuda.memory_reserved(0) / (1024**2)
                logger.info(f"GPU Model: {gpu_name}")
                logger.info(f"GPU Memory Allocated: {gpu_mem_alloc:.2f}MB")
                logger.info(f"GPU Memory Reserved: {gpu_mem_res:.2f}MB")
            except Exception as e:
                logger.error(f"Failed to get GPU details: {e}")

        logger.info("=" * 30)

    @property
    def device(self) -> str:
        """Get the selected device string.

        Returns:
            str: The selected device string.
        """
        return self._device

    @property
    def torch_device(self):
        """Get the torch.device object.

        Returns:
            torch.device or None: The torch.device object if PyTorch is available, None otherwise.
        """
        if not TORCH_AVAILABLE:
            return None
        return torch.device(self._device)

    def get_available_memory(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024**3)

    def is_gpu_available(self) -> bool:
        return self._device in ["cuda", "mps"]


# Global instance
device_manager = DeviceManager()
