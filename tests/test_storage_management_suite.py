"""
存储管理测试套件

合并了以下测试文件：
- test_storage_manager.py: 存储管理器基础功能测试
- test_manager_query.py: 存储管理器查询功能测试
- test_storage_backends.py: 存储后端实现测试（本地、MinIO、工厂）
- test_interfaces.py: 存储后端接口测试

功能覆盖：
- 文件存储（模型、文档、RAG数据）
- 文件检索
- 查询功能（用户文档查询、源文档RAG数据查询）
- 存储后端实现（本地文件系统、MinIO）
- 存储后端工厂模式
- 存储后端接口验证
"""

from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from raganything.storage.core.interfaces import StorageBackend
from raganything.storage.manager.storage_manager import StorageManager


# Storage Manager Basic Fixtures
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


# Manager Query Fixtures
@pytest_asyncio.fixture
async def manager_with_query_mock():
    backend = Mock()
    backend.list_files = AsyncMock()
    backend.get_metadata = AsyncMock()
    return StorageManager(backend)


class TestStorageManagerBasic:
    """存储管理器基础功能测试类"""

    @pytest.mark.asyncio
    async def test_store_model(self, storage_manager, mock_backend, tmp_path):
        """测试模型文件存储"""
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
    async def test_store_document(self, storage_manager, mock_backend, tmp_path):
        """测试文档文件存储"""
        d = tmp_path / "doc.pdf"
        d.write_bytes(b"pdf_data")

        result = await storage_manager.store_document(str(d), "user123")

        assert result == "stored_path"
        mock_backend.store_file.assert_called_once()
        args, _ = mock_backend.store_file.call_args
        assert args[0].startswith("documents/user123/")
        assert args[2]["custom_metadata"]["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_store_rag_data(self, storage_manager, mock_backend):
        """测试RAG数据存储"""
        content = b"vector_index"
        result = await storage_manager.store_rag_data("vector", content, "source_doc_id")

        assert result == "stored_path"
        mock_backend.store_file.assert_called_once()
        args, _ = mock_backend.store_file.call_args
        assert args[0].startswith("rag_data/vector/")
        assert args[2]["source_id"] == "source_doc_id"

    @pytest.mark.asyncio
    async def test_retrieval(self, storage_manager, mock_backend):
        """测试文件检索功能"""
        await storage_manager.retrieve_model("id")
        mock_backend.retrieve_file.assert_called_with("id")

        await storage_manager.retrieve_document("id")
        mock_backend.retrieve_file.assert_called_with("id")

        await storage_manager.retrieve_rag_data("id")
        mock_backend.retrieve_file.assert_called_with("id")


class TestStorageManagerQuery:
    """存储管理器查询功能测试类"""

    @pytest.mark.asyncio
    async def test_find_docs_by_user(self, manager_with_query_mock):
        """测试按用户查询文档"""
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
    async def test_find_rag_data_by_source_doc(self, manager_with_query_mock):
        """测试按源文档查询RAG数据"""
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


# ==================== 存储后端实现测试 ====================

import json
from unittest.mock import MagicMock

from raganything.storage.backends.local_backend import LocalFileSystemBackend
from raganything.storage.backends.minio_backend import MinIOStorageBackend
from raganything.storage.core.factory import StorageFactory


@pytest.fixture
def test_storage_dir(tmp_path):
    """为存储测试提供临时目录的fixture"""
    d = tmp_path / "storage_test"
    d.mkdir()
    return str(d)


@pytest_asyncio.fixture
async def backend(test_storage_dir):
    """提供初始化后的LocalFileSystemBackend的fixture"""
    return LocalFileSystemBackend(test_storage_dir)


class TestStorageBackends:
    """存储后端实现测试类"""
    
    @pytest.mark.asyncio
    async def test_local_store_and_retrieve_file(self, backend):
        """测试本地文件存储和检索"""
        file_path = "test_file.txt"
        content = b"Hello, World!"
        metadata = {"type": "text", "author": "tester"}

        # 存储
        stored_id = await backend.store_file(file_path, content, metadata)
        assert stored_id == file_path

        # 检索内容
        retrieved_content = await backend.retrieve_file(file_path)
        assert retrieved_content == content

        # 检索元数据
        retrieved_meta = await backend.get_metadata(file_path)
        assert retrieved_meta["type"] == "text"
        assert retrieved_meta["author"] == "tester"

    @pytest.mark.asyncio
    async def test_local_update_metadata(self, backend):
        """测试本地元数据更新"""
        file_path = "meta_test.txt"
        await backend.store_file(file_path, b"data", {"v": 1})

        await backend.update_metadata(file_path, {"v": 2, "new": "field"})

        new_meta = await backend.get_metadata(file_path)
        assert new_meta["v"] == 2
        assert new_meta["new"] == "field"

    @pytest.mark.asyncio
    async def test_local_delete_file(self, backend):
        """测试本地文件删除"""
        file_path = "del_test.txt"
        await backend.store_file(file_path, b"data", {})

        assert await backend.delete_file(file_path) is True

        with pytest.raises(FileNotFoundError):
            await backend.retrieve_file(file_path)

        with pytest.raises(FileNotFoundError):
            await backend.get_metadata(file_path)

    @pytest.mark.asyncio
    async def test_local_list_files(self, backend):
        """测试本地文件列表"""
        files = ["a.txt", "b.txt", "sub/c.txt"]
        for f in files:
            await backend.store_file(f, b"content", {})

        # 列出所有文件
        listed = await backend.list_files()
        assert len(listed) == 3
        assert "sub/c.txt" in listed

        # 按前缀列出
        sub_listed = await backend.list_files("sub")
        assert len(sub_listed) == 1
        assert sub_listed[0] == "sub/c.txt"

    @pytest.mark.asyncio
    async def test_local_directory_traversal_prevention(self, backend):
        """测试本地目录遍历防护"""
        with pytest.raises(ValueError):
            await backend.store_file("../hack.txt", b"bad", {})


class TestMinIOBackends:
    """MinIO存储后端测试类"""
    
    @pytest.fixture
    def mock_minio_client(self):
        """Mock MinIO客户端"""
        client = MagicMock()
        # 设置存储桶存在检查
        client.bucket_exists.return_value = True
        return client

    @pytest_asyncio.fixture
    async def minio_backend(self, mock_minio_client):
        """提供初始化后的MinIO后端的fixture（使用mock）"""
        # 我们patch Minio构造函数以返回我们的mock
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "raganything.storage.backends.minio_backend.Minio",
                lambda *args, **kwargs: mock_minio_client,
            )
            backend = MinIOStorageBackend("endpoint", "key", "secret", "bucket")
            return backend

    @pytest.mark.asyncio
    async def test_minio_store_file(self, minio_backend, mock_minio_client):
        """测试MinIO文件存储"""
        file_path = "test/file.txt"
        content = b"content"
        metadata = {"key": "value"}

        await minio_backend.store_file(file_path, content, metadata)

        mock_minio_client.put_object.assert_called_once()
        args, kwargs = mock_minio_client.put_object.call_args
        assert args[1] == file_path
        # 检查元数据编码
        s3_meta = kwargs.get("metadata", {})
        assert "custom-data-b64" in s3_meta

    @pytest.mark.asyncio
    async def test_minio_retrieve_file(self, minio_backend, mock_minio_client):
        """测试MinIO文件检索"""
        mock_response = MagicMock()
        mock_response.read.return_value = b"content"
        mock_minio_client.get_object.return_value = mock_response

        data = await minio_backend.retrieve_file("id")
        assert data == b"content"
        mock_response.close.assert_called()
        mock_response.release_conn.assert_called()

    @pytest.mark.asyncio
    async def test_minio_get_metadata(self, minio_backend, mock_minio_client):
        """测试MinIO元数据检索"""
        mock_stat = MagicMock()
        # 模拟返回的元数据（base64编码的json）
        import base64

        meta = {"key": "value"}
        b64 = base64.b64encode(json.dumps(meta).encode()).decode()
        mock_stat.metadata = {"x-amz-meta-custom-data-b64": b64}

        mock_minio_client.stat_object.return_value = mock_stat

        result = await minio_backend.get_metadata("id")
        assert result == meta

    @pytest.mark.asyncio
    async def test_minio_delete_file(self, minio_backend, mock_minio_client):
        """测试MinIO文件删除"""
        await minio_backend.delete_file("id")
        mock_minio_client.remove_object.assert_called_with("bucket", "id")

    @pytest.mark.asyncio
    async def test_minio_list_files(self, minio_backend, mock_minio_client):
        """测试MinIO文件列表"""
        obj1 = MagicMock()
        obj1.object_name = "file1"
        obj2 = MagicMock()
        obj2.object_name = "file2"

        mock_minio_client.list_objects.return_value = [obj1, obj2]

        files = await minio_backend.list_files()
        assert files == ["file1", "file2"]


class TestStorageFactory:
    """存储后端工厂测试类"""
    
    def test_create_local_backend(self):
        """测试创建本地存储后端"""
        config = {"backend": "local", "local_root": "/tmp/test_storage"}
        backend = StorageFactory.create_backend(config)
        assert isinstance(backend, LocalFileSystemBackend)
        assert backend.root_dir == "/tmp/test_storage"
    
    def test_create_minio_backend(self):
        """测试创建MinIO存储后端"""
        # 我们需要mock后端中的Minio客户端创建
        with pytest.MonkeyPatch.context() as m:
            m.setattr("raganything.storage.backends.minio_backend.Minio", MagicMock())
            
            config = {
                "backend": "minio",
                "minio_endpoint": "localhost:9000",
                "minio_access_key": "user",
                "minio_secret_key": "pass",
                "minio_bucket": "test-bucket",
            }
            backend = StorageFactory.create_backend(config)
            assert isinstance(backend, MinIOStorageBackend)
            assert backend.bucket_name == "test-bucket"
    
    def test_invalid_backend(self):
        """测试无效后端类型"""
        config = {"backend": "invalid"}
        with pytest.raises(ValueError):
            StorageFactory.create_backend(config)


class TestStorageBackendInterface:
    """存储后端接口测试类"""

    def test_cannot_instantiate_abstract_class(self):
        """测试StorageBackend不能直接实例化"""
        with pytest.raises(TypeError):
            StorageBackend()

    def test_subclass_must_implement_methods(self):
        """测试子类必须实现所有抽象方法"""

        class IncompleteBackend(StorageBackend):
            pass

        with pytest.raises(TypeError):
            IncompleteBackend()

    def test_compliant_subclass_instantiation(self):
        """测试兼容的子类可以实例化"""

        class CompliantBackend(StorageBackend):
            async def store_file(self, file_path, content, metadata):
                return "id"

            async def retrieve_file(self, file_id):
                return b""

            async def delete_file(self, file_id):
                return True

            async def list_files(self, prefix=""):
                return []

            async def get_metadata(self, file_id):
                return {}

            async def update_metadata(self, file_id, metadata):
                return True

        backend = CompliantBackend()
        assert isinstance(backend, StorageBackend)