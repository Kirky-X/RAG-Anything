import pytest

from raganything.storage.core.interfaces import StorageBackend


class TestStorageBackendInterface:
    def test_cannot_instantiate_abstract_class(self):
        """Test that StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageBackend()

    def test_subclass_must_implement_methods(self):
        """Test that a subclass must implement all abstract methods."""

        class IncompleteBackend(StorageBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_compliant_subclass_instantiation(self):
        """Test that a compliant subclass can be instantiated."""

        class CompliantBackend(StorageBackend):
            async def store_file(self, file_path, content, metadata): return "id"

            async def retrieve_file(self, file_id): return b""

            async def delete_file(self, file_id): return True

            async def list_files(self, prefix=""): return []

            async def get_metadata(self, file_id): return {}

            async def update_metadata(self, file_id, metadata): return True

        backend = CompliantBackend()
        assert isinstance(backend, StorageBackend)
