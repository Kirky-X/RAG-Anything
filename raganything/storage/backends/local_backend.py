# Copyright (c) 2025 Kirky.X
# All rights reserved.

import asyncio
import json
import os
import shutil
from typing import Dict, List

import aiofiles

from raganything.storage.core.interfaces import StorageBackend


class LocalFileSystemBackend(StorageBackend):
    """
    Implementation of StorageBackend for local file system.
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.metadata_dir = os.path.join(self.root_dir, ".metadata")
        os.makedirs(self.root_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    def _get_abs_path(self, file_path: str) -> str:
        """Convert relative path to absolute path and prevent directory traversal."""
        abs_path = os.path.abspath(os.path.join(self.root_dir, file_path))
        if not abs_path.startswith(self.root_dir):
            raise ValueError("Invalid file path: outside root directory")
        return abs_path

    def _get_metadata_path(self, file_path: str) -> str:
        """Get the path for the metadata file."""
        # Use a flat structure or mirrored structure for metadata?
        # Design doc says: "元数据以JSON文件形式存储在独立的.metadata目录中"
        # Let's use a hashed or relative path structure in metadata to avoid collisions.
        # Simple approach: Mirror the directory structure inside .metadata
        rel_path = file_path
        if os.path.isabs(file_path):
            rel_path = os.path.relpath(file_path, self.root_dir)

        meta_path = os.path.join(self.metadata_dir, rel_path + ".json")
        return meta_path

    async def store_file(self, file_path: str, content: bytes, metadata: Dict) -> str:
        abs_path = self._get_abs_path(file_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Store file content
        async with aiofiles.open(abs_path, "wb") as f:
            await f.write(content)

        # Store metadata
        meta_path = self._get_metadata_path(file_path)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(metadata, indent=2))

        return file_path

    async def retrieve_file(self, file_id: str) -> bytes:
        abs_path = self._get_abs_path(file_id)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {file_id}")

        async with aiofiles.open(abs_path, "rb") as f:
            return await f.read()

    async def delete_file(self, file_id: str) -> bool:
        abs_path = self._get_abs_path(file_id)
        meta_path = self._get_metadata_path(file_id)

        try:
            if os.path.exists(abs_path):
                os.remove(abs_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            return True
        except Exception:
            return False

    async def list_files(self, prefix: str = "") -> List[str]:
        # Walk through the root directory
        files_list = []
        search_dir = self.root_dir

        # If prefix implies a directory, narrow down search
        if prefix:
            # Ensure prefix doesn't escape root
            full_prefix = os.path.join(self.root_dir, prefix)
            if not os.path.abspath(full_prefix).startswith(self.root_dir):
                raise ValueError("Invalid prefix")

        for root, dirs, files in os.walk(search_dir):
            # Skip metadata dir
            if ".metadata" in root:
                continue

            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, self.root_dir)

                if rel_path.startswith(prefix):
                    files_list.append(rel_path)

        return files_list

    async def get_metadata(self, file_id: str) -> Dict:
        meta_path = self._get_metadata_path(file_id)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found for: {file_id}")

        async with aiofiles.open(meta_path, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def update_metadata(self, file_id: str, metadata: Dict) -> bool:
        meta_path = self._get_metadata_path(file_id)

        current_metadata = {}
        if os.path.exists(meta_path):
            try:
                current_metadata = await self.get_metadata(file_id)
            except Exception:
                pass

        # Merge updates
        current_metadata.update(metadata)

        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(current_metadata, indent=2))

        return True
