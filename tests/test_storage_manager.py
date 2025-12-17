import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
import os
from raganything.storage.manager.storage_manager import StorageManager
from raganything.storage.core.interfaces import StorageBackend

@pytest.fixture
def mock_backend():
    backend = Mock(spec=StorageBackend)
    backend.store_file = AsyncMock(return_value="stored_path")
    backend.retrieve_file = AsyncMock(return_value=b"content")
    backend.get_metadata = AsyncMock(return_value={})
    return backend

@pytest.fixture
def storage_manager(mock_backend):
    return StorageManager(mock_backend)

@pytest.mark.asyncio
async def test_store_model(storage_manager, mock_backend, tmp_path):
    # Create dummy file
    d = tmp_path / "model.bin"
    d.write_bytes(b"model_data")
    
    result = await storage_manager.store_model("gpt-4", str(d))
    
    assert result == "stored_path"
    mock_backend.store_file.assert_called_once()
    args, _ = mock_backend.store_file.call_args
    assert args[0].startswith("models/general/gpt-4/latest/")
    assert args[1] == b"model_data"
    assert args[2]["type"] == "model"

@pytest.mark.asyncio
async def test_store_document(storage_manager, mock_backend, tmp_path):
    d = tmp_path / "doc.pdf"
    d.write_bytes(b"pdf_data")
    
    result = await storage_manager.store_document(str(d), "user123")
    
    assert result == "stored_path"
    mock_backend.store_file.assert_called_once()
    args, _ = mock_backend.store_file.call_args
    assert args[0].startswith("documents/user123/")
    assert args[2]["custom_metadata"]["user_id"] == "user123"

@pytest.mark.asyncio
async def test_store_rag_data(storage_manager, mock_backend):
    content = b"vector_index"
    result = await storage_manager.store_rag_data("vector", content, "source_doc_id")
    
    assert result == "stored_path"
    mock_backend.store_file.assert_called_once()
    args, _ = mock_backend.store_file.call_args
    assert args[0].startswith("rag_data/vector/")
    assert args[2]["source_id"] == "source_doc_id"

@pytest.mark.asyncio
async def test_retrieval(storage_manager, mock_backend):
    await storage_manager.retrieve_model("id")
    mock_backend.retrieve_file.assert_called_with("id")
    
    await storage_manager.retrieve_document("id")
    mock_backend.retrieve_file.assert_called_with("id")
    
    await storage_manager.retrieve_rag_data("id")
    mock_backend.retrieve_file.assert_called_with("id")
