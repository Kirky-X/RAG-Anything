import pytest
import pytest_asyncio
from moto import mock_aws
import boto3
import os
import json
from raganything.storage.backends.minio_backend import MinIOStorageBackend

# Since we are using 'moto' to mock S3, and 'minio' client connects to S3 compatible APIs.
# We need to trick minio client to connect to moto's server.
# Moto usually mocks boto3. 
# It's hard to make Minio client talk to Moto directly without starting a server.
# Alternatively, we can mock the Minio client itself or use a Moto server mode.

# However, for simplicity and speed in this environment, mocking the Minio client wrapper behavior 
# might be safer if we can't spin up a real service.
# BUT the requirement is "Production Grade" and "Real implementations".
# So we should use `moto_server` or just rely on boto3 based implementation?
# The `MinIOStorageBackend` uses `minio` library. 
# To test it properly without a real MinIO, we need `moto[server]`.

# Let's try to mock the Minio client object calls instead, 
# ensuring logic coverage (metadata handling, async wrapping).
# Or we can write a test that assumes a local MinIO is present (but it likely isn't).
# Best approach here: Mock the Minio client instance inside the backend.

from unittest.mock import MagicMock, ANY

@pytest.fixture
def mock_minio_client():
    client = MagicMock()
    # Setup bucket exists check
    client.bucket_exists.return_value = True
    return client

@pytest_asyncio.fixture
async def minio_backend(mock_minio_client):
    # We patch the Minio constructor to return our mock
    with pytest.MonkeyPatch.context() as m:
        m.setattr("raganything.storage.backends.minio_backend.Minio", lambda *args, **kwargs: mock_minio_client)
        backend = MinIOStorageBackend("endpoint", "key", "secret", "bucket")
        return backend

@pytest.mark.asyncio
async def test_minio_store_file(minio_backend, mock_minio_client):
    file_path = "test/file.txt"
    content = b"content"
    metadata = {"key": "value"}
    
    await minio_backend.store_file(file_path, content, metadata)
    
    mock_minio_client.put_object.assert_called_once()
    args, kwargs = mock_minio_client.put_object.call_args
    assert args[1] == file_path
    # Check metadata encoding
    s3_meta = kwargs.get("metadata", {})
    assert "custom-data-b64" in s3_meta

@pytest.mark.asyncio
async def test_minio_retrieve_file(minio_backend, mock_minio_client):
    mock_response = MagicMock()
    mock_response.read.return_value = b"content"
    mock_minio_client.get_object.return_value = mock_response
    
    data = await minio_backend.retrieve_file("id")
    assert data == b"content"
    mock_response.close.assert_called()
    mock_response.release_conn.assert_called()

@pytest.mark.asyncio
async def test_minio_get_metadata(minio_backend, mock_minio_client):
    mock_stat = MagicMock()
    # Simulate returned metadata (base64 encoded json)
    import base64
    meta = {"key": "value"}
    b64 = base64.b64encode(json.dumps(meta).encode()).decode()
    mock_stat.metadata = {"x-amz-meta-custom-data-b64": b64} 
    
    mock_minio_client.stat_object.return_value = mock_stat
    
    result = await minio_backend.get_metadata("id")
    assert result == meta

@pytest.mark.asyncio
async def test_minio_delete_file(minio_backend, mock_minio_client):
    await minio_backend.delete_file("id")
    mock_minio_client.remove_object.assert_called_with("bucket", "id")

@pytest.mark.asyncio
async def test_minio_list_files(minio_backend, mock_minio_client):
    obj1 = MagicMock()
    obj1.object_name = "file1"
    obj2 = MagicMock()
    obj2.object_name = "file2"
    
    mock_minio_client.list_objects.return_value = [obj1, obj2]
    
    files = await minio_backend.list_files()
    assert files == ["file1", "file2"]
