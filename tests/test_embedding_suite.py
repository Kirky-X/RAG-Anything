"""
嵌入功能测试套件

合并了以下测试文件：
- test_embedding.py: 嵌入功能基础测试
- test_serialization.py: 嵌入和LLM序列化测试
- test_llm_langchain_builder.py: LLM LangChain构建器测试
- test_device_manager.py: 设备管理器测试

功能覆盖：
- 嵌入函数构建（Ollama等）
- 懒加载包装器
- 嵌入序列化
- LLM序列化
- 设备管理（CPU/CUDA/MPS检测与选择）
"""

import pickle
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pytest

from raganything.llm.embedding import (LazyLangChainEmbeddingWrapper,
                                       build_embedding_func)
from raganything.llm.llm import LLMProviderConfig, build_llm


# Common mock classes
class MockEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def aembed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]

    def embed_documents(self, texts):
        return [[0.1] * 1024 for _ in texts]


class MockLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, messages):
        return "mock response"


class TestEmbeddingBasic:
    """嵌入功能基础测试类"""

    @pytest.mark.asyncio
    async def test_lazy_wrapper_initialization(self):
        """测试LazyLangChainEmbeddingWrapper初始化和调用"""
        init_kwargs = {"model": "test-model", "base_url": "http://localhost:11434"}

        # Mock the underlying embedding class
        with patch(
            "langchain_community.embeddings.OllamaEmbeddings", side_effect=MockEmbeddings
        ) as MockClass:
            wrapper = LazyLangChainEmbeddingWrapper(MockClass, init_kwargs)

            # Test calling the wrapper
            texts = ["Hello world", "Test sentence"]
            vectors = await wrapper(texts)

            assert len(vectors) == 2
            assert len(vectors[0]) == 1024

            # Verify initialization happened
            MockClass.assert_called_once()
            call_args = MockClass.call_args[1]
            assert call_args["model"] == "test-model"
            assert call_args["base_url"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_build_embedding_func_ollama(self):
        """测试Ollama提供者的嵌入函数构建"""
        # Use new=MockEmbeddings instead of side_effect so that initialization works correctly without calling real __init__
        # and the class itself is replaced, making it safe for instantiation in the wrapper.
        with patch("langchain_community.embeddings.OllamaEmbeddings", new=MockEmbeddings):
            # We also need to patch the one imported in raganything.llm.embedding if it exists,
            # or if it's imported inside the function.
            # Since the function uses `from langchain_community.embeddings import OllamaEmbeddings` locally,
            # patching `langchain_community.embeddings.OllamaEmbeddings` should be enough if sys.modules is patched.

            embedding_func = build_embedding_func(
                provider="ollama",
                model="bge-m3:567m",
                api_base="http://localhost:11434",
                embedding_dim=1024,
            )

            assert embedding_func is not None
            assert embedding_func.embedding_dim == 1024

            # Test functionality
            wrapper = embedding_func.func
            texts = ["Test"]
            vectors = await wrapper(texts)
            assert len(vectors) == 1
            assert len(vectors[0]) == 1024

    @pytest.mark.asyncio
    async def test_build_embedding_func_openai(self):
        """测试OpenAI提供者的嵌入函数构建（占位符）"""
        # We need to mock OpenAIEmbeddings if we were testing that path
        # For now, just ensuring the switch works if we had mocked it
        pass


class TestEmbeddingSerialization:
    """嵌入序列化测试类"""

    def test_pickle_embedding_func(self):
        """测试嵌入函数的序列化"""
        # We use new=MockEmbeddings so that the class itself is replaced by our mock class,
        # rather than a MagicMock object. This ensures the class reference in LazyLangChainEmbeddingWrapper
        # is picklable.
        with patch("langchain_community.embeddings.OllamaEmbeddings", new=MockEmbeddings):
            embed_func_wrapper = build_embedding_func(
                provider="ollama",
                model="nomic-embed-text:latest",
                api_base="http://localhost:11434",
            )

            # Try pickling the wrapper
            pickled = pickle.dumps(embed_func_wrapper)
            unpickled = pickle.loads(pickled)

            assert unpickled is not None
            assert unpickled.embedding_dim == embed_func_wrapper.embedding_dim


class TestLLMSerialization:
    """LLM序列化测试类"""

    @pytest.mark.asyncio
    async def test_pickle_llm(self):
        """测试LLM函数的序列化"""
        config = LLMProviderConfig(
            provider="ollama", model="qwen3:1.7b", api_base="http://localhost:11434"
        )

        # Mock dependencies that might be missing or problematic during test
        # We need to mock them BEFORE importing raganything.llm.ollama_client
        mock_lc_messages = MagicMock()
        mock_lc_messages.BaseMessage = MagicMock
        mock_lc_messages.AIMessage = MagicMock

        mock_lc_outputs = MagicMock()
        mock_lc_outputs.ChatResult = MagicMock

        mock_lc_ollama = MagicMock()
        mock_lc_ollama.ChatOllama = MagicMock

        with patch.dict(
            sys.modules,
            {
                "langchain_core.messages": mock_lc_messages,
                "langchain_core.outputs": mock_lc_outputs,
                "langchain_ollama": mock_lc_ollama,
            },
        ):
            # Now we can import the module that was failing

            # Now patch the class in that module
            with patch("raganything.llm.ollama_client.RobustOllamaClient", new=MockLLM):
                llm_instance = build_llm(config)

                # Try pickling
                pickled = pickle.dumps(llm_instance)
                unpickled = pickle.loads(pickled)

                assert unpickled is not None
                # The unpickled object's client should be an instance of our MockLLM
                assert isinstance(unpickled.chat_model.client, MockLLM)


# Mock classes for LangChain builder tests
class _FakeAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        return _FakeAIMessage("ok-openai")


class _FakeChatOllama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def ainvoke(self, messages):
        return _FakeAIMessage("ok-ollama")


def _install_fake_langchain_openai():
    m = types.ModuleType("langchain_openai")
    m.ChatOpenAI = _FakeChatOpenAI
    sys.modules["langchain_openai"] = m

    core = types.ModuleType("langchain_core.messages")

    class _Sys:
        def __init__(self, content):
            self.content = content

    class _Human:
        def __init__(self, content):
            self.content = content

    class _AI:
        def __init__(self, content):
            self.content = content

    core.SystemMessage = _Sys
    core.HumanMessage = _Human
    core.AIMessage = _AI
    sys.modules["langchain_core.messages"] = core


def _install_fake_langchain_ollama():
    m = types.ModuleType("langchain_ollama")
    m.ChatOllama = _FakeChatOllama
    sys.modules["langchain_ollama"] = m

    if "langchain_core.messages" not in sys.modules:
        _install_fake_langchain_openai()


class TestLLMLangChainBuilder:
    """LLM LangChain构建器测试类"""

    def test_build_openai_and_call_text(self):
        """测试构建OpenAI LLM并调用文本接口"""
        _install_fake_langchain_openai()
        cfg = LLMProviderConfig(provider="openai", model="gpt-4o-mini", api_key="k")
        
        with build_llm(cfg) as llm:
            import asyncio
            out = asyncio.run(llm("hello"))
            assert "ok-openai" in out

    def test_build_openrouter_and_call_messages(self):
        """测试构建OpenRouter LLM并调用消息接口"""
        _install_fake_langchain_openai()
        cfg = LLMProviderConfig(
            provider="openrouter",
            model="openrouter/model",
            api_base="https://openrouter.ai/api/v1",
            api_key="rk",
        )
        with build_llm(cfg) as llm:
            msgs = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "t"}]},
            ]
            import asyncio

            out = asyncio.run(llm("", messages=msgs))
            assert "ok-openai" in out

    def test_build_ollama_and_call_text(self):
        """测试构建Ollama LLM并调用文本接口"""
        _install_fake_langchain_ollama()
        cfg = LLMProviderConfig(
            provider="ollama", model="qwen3:1.7b", api_base="http://localhost:11434"
        )
        with build_llm(cfg) as llm:
            import asyncio

            out = asyncio.run(llm("hello"))
            assert "ok-ollama" in out


class TestDeviceManager(unittest.TestCase):
    """设备管理器测试类"""

    def setUp(self):
        """测试前准备"""
        # Reset singleton before each test
        from raganything.models.device import DeviceManager
        DeviceManager._instance = None

    def test_singleton(self):
        """测试DeviceManager遵循单例模式"""
        from raganything.models.device import DeviceManager
        dm1 = DeviceManager()
        dm2 = DeviceManager()
        self.assertIs(dm1, dm2)

    def test_device_properties(self):
        """测试基本设备属性"""
        from raganything.models.device import DeviceManager
        dm = DeviceManager()
        self.assertIn(dm.device, ("cpu", "cuda", "mps"))
        mem = dm.get_available_memory()
        self.assertIsInstance(mem, float)
        self.assertGreaterEqual(mem, 0)

    def test_cpu_fallback_when_torch_unavailable(self):
        """测试当torch不可用时回退到CPU"""
        import raganything.models.device as dev
        from raganything.models.device import DeviceManager
        with patch.object(dev, "TORCH_AVAILABLE", False):
            # Recreate singleton
            DeviceManager._instance = None
            dm = DeviceManager()
            self.assertEqual(dm.device, "cpu")
            self.assertIsNone(dm.torch_device)
            self.assertFalse(dm.is_gpu_available())

    def test_selects_cuda_when_available(self):
        """测试当CUDA可用时选择CUDA"""
        import raganything.models.device as dev
        from raganything.models.device import DeviceManager

        class FakeTensor:
            def cuda(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_name(_idx):
                return "FakeGPU"

            @staticmethod
            def memory_allocated(_idx):
                return 0

            @staticmethod
            def memory_reserved(_idx):
                return 0

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FakeTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "cuda")
                self.assertTrue(dm.is_gpu_available())

    def test_selects_mps_when_available(self):
        """测试当MPS可用时选择MPS（macOS）"""
        import raganything.models.device as dev
        from raganything.models.device import DeviceManager

        class FakeTensor:
            def cuda(self):
                return self

            def to(self, *_args, **_kwargs):
                return self

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return True

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FakeTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "mps")
                self.assertTrue(dm.is_gpu_available())

    def test_init_failure_fallback(self):
        """测试如果GPU初始化失败则回退到CPU"""
        import raganything.models.device as dev
        from raganything.models.device import DeviceManager

        class FailingTensor:
            def cuda(self):
                raise RuntimeError("cuda fail")

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        class FakeBackends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        class FakeTorch:
            cuda = FakeCuda()
            backends = FakeBackends()

            def tensor(self, _arr):
                return FailingTensor()

            def device(self, d):
                return d

        with patch.object(dev, "TORCH_AVAILABLE", True):
            with patch.object(dev, "torch", FakeTorch()):
                DeviceManager._instance = None
                dm = DeviceManager()
                self.assertEqual(dm.device, "cpu")