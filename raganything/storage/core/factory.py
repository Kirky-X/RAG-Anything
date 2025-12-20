# Copyright (c) 2025 Kirky.X
# All rights reserved.

import logging
from typing import Dict, Optional

from raganything.storage.backends.local_backend import LocalFileSystemBackend
from raganything.storage.core.interfaces import StorageBackend
from raganything.storage.manager.storage_manager import StorageManager
from raganything.i18n import _

# 尝试导入MinIO，如果失败则记录日志并继续
MINIO_AVAILABLE = False
try:
    from raganything.storage.backends.minio_backend import MinIOStorageBackend

    MINIO_AVAILABLE = True
except ImportError as e:
    logging.warning(
        f"MinIO backend not available: {e}. Falling back to local storage for MinIO requests."
    )


class StorageFactory:
    """
    Factory to create StorageManager instances based on configuration.
    Defined in section 5.3 (Configuration) and 12.3 (Plugin Architecture idea).
    """

    @staticmethod
    def create_backend(config: Dict) -> StorageBackend:
        backend_type = config.get("backend", "local").lower()

        if backend_type == "local":
            root_dir = config.get("local_root", "./storage")
            return LocalFileSystemBackend(root_dir)

        elif backend_type == "minio":
            if not MINIO_AVAILABLE:
                logging.warning(
                    "MinIO requested but not available. Falling back to local storage."
                )
                root_dir = config.get("local_root", "./storage")
                return LocalFileSystemBackend(root_dir)

            return MinIOStorageBackend(
                endpoint=config.get("minio_endpoint", "localhost:9000"),
                access_key=config.get("minio_access_key", ""),
                secret_key=config.get("minio_secret_key", ""),
                bucket_name=config.get("minio_bucket", "rag-storage"),
                secure=config.get("minio_secure", False),
            )
        else:
            raise ValueError(_("Unsupported storage backend: {}").format(backend_type))

    @staticmethod
    def create_manager(config: Dict) -> StorageManager:
        backend = StorageFactory.create_backend(config)
        return StorageManager(backend)
