# Copyright (c) 2025 Kirky.X
# All rights reserved.

"""
Storage module for RAG-Anything.

This module provides storage backends and management functionality for storing
and retrieving files, models, and other data.
"""

from .core.interfaces import StorageBackend
from .core.factory import StorageFactory
from .manager.storage_manager import StorageManager
from .backends.local_backend import LocalFileSystemBackend
from .backends.minio_backend import MinIOStorageBackend

__all__ = [
    "StorageBackend",
    "StorageFactory",
    "StorageManager",
    "LocalFileSystemBackend",
    "MinIOStorageBackend",
]
