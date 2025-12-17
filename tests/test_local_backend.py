import pytest
import pytest_asyncio

from raganything.storage.backends.local_backend import LocalFileSystemBackend


@pytest.fixture
def test_storage_dir(tmp_path):
    """Fixture to provide a temporary directory for storage tests."""
    d = tmp_path / "storage_test"
    d.mkdir()
    return str(d)


@pytest_asyncio.fixture
async def backend(test_storage_dir):
    """Fixture to provide an initialized LocalFileSystemBackend."""
    return LocalFileSystemBackend(test_storage_dir)


@pytest.mark.asyncio
async def test_store_and_retrieve_file(backend):
    file_path = "test_file.txt"
    content = b"Hello, World!"
    metadata = {"type": "text", "author": "tester"}

    # Store
    stored_id = await backend.store_file(file_path, content, metadata)
    assert stored_id == file_path

    # Retrieve content
    retrieved_content = await backend.retrieve_file(file_path)
    assert retrieved_content == content

    # Retrieve metadata
    retrieved_meta = await backend.get_metadata(file_path)
    assert retrieved_meta["type"] == "text"
    assert retrieved_meta["author"] == "tester"


@pytest.mark.asyncio
async def test_update_metadata(backend):
    file_path = "meta_test.txt"
    await backend.store_file(file_path, b"data", {"v": 1})

    await backend.update_metadata(file_path, {"v": 2, "new": "field"})

    new_meta = await backend.get_metadata(file_path)
    assert new_meta["v"] == 2
    assert new_meta["new"] == "field"


@pytest.mark.asyncio
async def test_delete_file(backend):
    file_path = "del_test.txt"
    await backend.store_file(file_path, b"data", {})

    assert await backend.delete_file(file_path) is True

    with pytest.raises(FileNotFoundError):
        await backend.retrieve_file(file_path)

    with pytest.raises(FileNotFoundError):
        await backend.get_metadata(file_path)


@pytest.mark.asyncio
async def test_list_files(backend):
    files = ["a.txt", "b.txt", "sub/c.txt"]
    for f in files:
        await backend.store_file(f, b"content", {})

    # List all
    listed = await backend.list_files()
    assert len(listed) == 3
    assert "sub/c.txt" in listed

    # List with prefix
    sub_listed = await backend.list_files("sub")
    assert len(sub_listed) == 1
    assert sub_listed[0] == "sub/c.txt"


@pytest.mark.asyncio
async def test_directory_traversal_prevention(backend):
    with pytest.raises(ValueError):
        await backend.store_file("../hack.txt", b"bad", {})
