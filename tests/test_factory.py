import pytest
from unittest.mock import MagicMock
from raganything.storage.core.factory import StorageFactory
from raganything.storage.backends.local_backend import LocalFileSystemBackend
from raganything.storage.backends.minio_backend import MinIOStorageBackend

def test_create_local_backend():
    config = {
        "backend": "local",
        "local_root": "/tmp/test_storage"
    }
    backend = StorageFactory.create_backend(config)
    assert isinstance(backend, LocalFileSystemBackend)
    assert backend.root_dir == "/tmp/test_storage"

def test_create_minio_backend():
    # We need to mock Minio client creation inside backend
    with pytest.MonkeyPatch.context() as m:
        m.setattr("raganything.storage.backends.minio_backend.Minio", MagicMock())
        
        config = {
            "backend": "minio",
            "minio_endpoint": "localhost:9000",
            "minio_access_key": "user",
            "minio_secret_key": "pass",
            "minio_bucket": "test-bucket"
        }
        backend = StorageFactory.create_backend(config)
        assert isinstance(backend, MinIOStorageBackend)
        assert backend.bucket_name == "test-bucket"

def test_invalid_backend():
    config = {"backend": "invalid"}
    with pytest.raises(ValueError):
        StorageFactory.create_backend(config)
