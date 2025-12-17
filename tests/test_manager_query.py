from unittest.mock import Mock, AsyncMock

import pytest
import pytest_asyncio

from raganything.storage.manager.storage_manager import StorageManager


@pytest_asyncio.fixture
async def manager_with_query_mock():
    backend = Mock()
    backend.list_files = AsyncMock()
    backend.get_metadata = AsyncMock()
    return StorageManager(backend)


@pytest.mark.asyncio
async def test_find_docs_by_user(manager_with_query_mock):
    manager, backend = manager_with_query_mock, manager_with_query_mock.backend

    # Mock list_files to return some paths
    backend.list_files.return_value = ["documents/u1/d1.pdf", "documents/u1/d2.txt"]

    # Mock metadata retrieval
    async def get_meta_side_effect(file_id):
        return {"id": file_id.split("/")[-1], "user_id": "u1"}

    backend.get_metadata.side_effect = get_meta_side_effect

    results = await manager.find_docs_by_user("u1")

    backend.list_files.assert_called_with(prefix="documents/u1/")
    assert len(results) == 2
    assert results[0]["user_id"] == "u1"


@pytest.mark.asyncio
async def test_find_rag_data_by_source_doc(manager_with_query_mock):
    manager, backend = manager_with_query_mock, manager_with_query_mock.backend

    backend.list_files.return_value = ["rag_data/v/1.bin", "rag_data/v/2.bin"]

    async def get_meta_side_effect(file_id):
        if "1.bin" in file_id:
            return {"id": "1", "source_id": "target_doc"}
        else:
            return {"id": "2", "source_id": "other_doc"}

    backend.get_metadata.side_effect = get_meta_side_effect

    results = await manager.find_rag_data_by_source_doc("target_doc")

    backend.list_files.assert_called_with(prefix="rag_data/")
    assert len(results) == 1
    assert results[0]["id"] == "1"
