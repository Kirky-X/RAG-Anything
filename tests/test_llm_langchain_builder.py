import types
import sys

from raganything.llm import LLMProviderConfig, build_llm


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


def test_build_openai_and_call_text():
    _install_fake_langchain_openai()
    cfg = LLMProviderConfig(provider="openai", model="gpt-4o-mini", api_key="k")
    llm = build_llm(cfg)
    import asyncio
    out = asyncio.run(llm("hello"))
    assert "ok-openai" in out


def test_build_openrouter_and_call_messages():
    _install_fake_langchain_openai()
    cfg = LLMProviderConfig(provider="openrouter", model="openrouter/model", api_base="https://openrouter.ai/api/v1", api_key="rk")
    llm = build_llm(cfg)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": "t"}]},
    ]
    import asyncio
    out = asyncio.run(llm("", messages=msgs))
    assert "ok-openai" in out


def test_build_ollama_and_call_text():
    _install_fake_langchain_ollama()
    cfg = LLMProviderConfig(provider="ollama", model="llama3.1", api_base="http://localhost:11434")
    llm = build_llm(cfg)
    import asyncio
    out = asyncio.run(llm("hello"))
    assert "ok-ollama" in out
