# Copyright (c) 2025 Kirky.X
# All rights reserved.

import json
import io
import asyncio
from typing import Dict, List
from minio import Minio
from minio.error import S3Error
from raganything.storage.core.interfaces import StorageBackend

class MinIOStorageBackend(StorageBackend):
    """
    Implementation of StorageBackend for MinIO/S3 object storage.
    """
    
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)

    async def store_file(self, file_path: str, content: bytes, metadata: Dict) -> str:
        # MinIO/S3 metadata values must be strings. 
        # We need to serialize our rich metadata dict into something S3 compatible, 
        # or store it as a separate object.
        # The design doc says: "元数据存储在对象的自定义头部属性中" (Metadata stored in custom header attributes)
        # However, headers have size limits and character restrictions.
        # A robust way is to store metadata as a separate object or JSON string in a single header if small.
        # Let's try to store as user metadata headers, but flattened/serialized.
        
        # NOTE: MinIO python client 'metadata' arg expects Dict[str, str].
        # We will JSON serialize the whole metadata dict into a single header 'x-amz-meta-custom-data'
        # OR better, stick to the design doc recommendation if possible, but practical limits apply.
        # Let's store the full metadata as a sidecar object to ensure no data loss/limit issues,
        # similar to the local backend. This is safer for large metadata.
        # BUT, the design doc explicitly says: "元数据存储在对象的自定义头部属性中" (Section 8.1.2).
        # We will follow the design doc but warn about limits if we could. 
        # For this implementation, let's serialize to JSON and put in a header, 
        # but if it's too big, we might have issues.
        
        # To be safe and robust (Production Grade), we should probably do BOTH or prefer sidecar if large.
        # Given "Strictly follow design doc", we try headers.
        
        # S3 user metadata keys are prefixed with x-amz-meta-
        # The Minio client handles this prefix.
        
        # We'll encode the JSON to a string. 
        # Warning: S3 headers must be ASCII. We need to base64 encode if it contains non-ASCII.
        import base64
        meta_json = json.dumps(metadata)
        meta_b64 = base64.b64encode(meta_json.encode('utf-8')).decode('ascii')
        
        s3_metadata = {
            "custom-data-b64": meta_b64
        }
        
        # Upload content
        content_stream = io.BytesIO(content)
        
        # Minio put_object is synchronous. In an async app, run in executor.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.put_object(
                self.bucket_name,
                file_path,
                content_stream,
                len(content),
                metadata=s3_metadata
            )
        )
        
        return file_path

    async def retrieve_file(self, file_id: str) -> bytes:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.client.get_object(self.bucket_name, file_id)
            )
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found: {file_id}")
            raise

    async def delete_file(self, file_id: str) -> bool:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.remove_object(self.bucket_name, file_id)
            )
            return True
        except Exception:
            return False

    async def list_files(self, prefix: str = "") -> List[str]:
        loop = asyncio.get_event_loop()
        objects = await loop.run_in_executor(
            None,
            lambda: list(self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True))
        )
        return [obj.object_name for obj in objects]

    async def get_metadata(self, file_id: str) -> Dict:
        loop = asyncio.get_event_loop()
        try:
            stat = await loop.run_in_executor(
                None,
                lambda: self.client.stat_object(self.bucket_name, file_id)
            )
            
            # Retrieve our custom metadata
            # Minio client puts user metadata in stat.metadata (keys might be lowercased)
            # Keys usually come back as 'x-amz-meta-custom-data-b64' or similar, 
            # but minio client might strip prefix. Let's check 'custom-data-b64'.
            
            # Note: minio python client stores metadata in `metadata` attribute.
            # It seems it preserves the case or lowercases it.
            
            meta_b64 = None
            # Search for our key (case insensitive)
            if stat.metadata:
                for k, v in stat.metadata.items():
                    if k.lower().endswith("custom-data-b64"):
                        meta_b64 = v
                        break
            
            if meta_b64:
                import base64
                meta_json = base64.b64decode(meta_b64).decode('utf-8')
                return json.loads(meta_json)
            else:
                return {}
                
        except S3Error as e:
            if e.code == 'NoSuchKey':
                raise FileNotFoundError(f"File not found: {file_id}")
            raise

    async def update_metadata(self, file_id: str, metadata: Dict) -> bool:
        # S3/MinIO doesn't support updating metadata in place easily.
        # We must copy the object to itself with new metadata.
        
        current_metadata = {}
        try:
            current_metadata = await self.get_metadata(file_id)
        except Exception:
            pass
            
        current_metadata.update(metadata)
        
        import base64
        meta_json = json.dumps(current_metadata)
        meta_b64 = base64.b64encode(meta_json.encode('utf-8')).decode('ascii')
        
        new_s3_metadata = {
            "custom-data-b64": meta_b64
        }
        
        from minio.commonconfig import CopySource
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.client.copy_object(
                    self.bucket_name,
                    file_id,
                    CopySource(self.bucket_name, file_id),
                    metadata=new_s3_metadata,
                    metadata_directive="REPLACE"
                )
            )
            return True
        except Exception:
            return False
