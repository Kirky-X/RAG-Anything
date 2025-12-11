# Copyright (c) 2025 Kirky.X
# All rights reserved.

from typing import Any, Dict, List, Optional, Union
import os
import base64
from io import BytesIO


class LLMProviderConfig:
    """Configuration class for LLM providers."""
    def __init__(
        self,
        provider: str,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Initialize LLM provider configuration.

        Args:
            provider (str): The LLM provider name (openai, ollama, etc.).
            model (str): The model name to use.
            api_base (Optional[str]): Optional API base URL.
            api_key (Optional[str]): Optional API key for authentication.
            timeout (Optional[float]): Optional timeout for API requests.
            max_retries (int): Maximum number of retries for failed requests. Defaults to 2.
            extra (Optional[Dict[str, Any]]): Extra parameters for the provider.
        """
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra = extra or {}


class LLM:
    """Lightweight LLM wrapper for various providers."""
    def __init__(
        self,
        chat_model: Any,
        force_text_messages: bool = False,
        direct_base_url: Optional[str] = None,
        direct_model: Optional[str] = None,
    ):
        """Initialize LLM wrapper.

        Args:
            chat_model (Any): The underlying chat model.
            force_text_messages (bool): Whether to force text-only messages.
            direct_base_url (Optional[str]): Direct base URL for API calls.
            direct_model (Optional[str]): Direct model name for API calls.
        """
        self.chat_model = chat_model
        self.force_text_messages = force_text_messages
        self.direct_base_url = direct_base_url
        self.direct_model = direct_model

    async def __call__(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """Invoke the LLM with the given prompt and parameters.

        Args:
            prompt (str): The main prompt to send to the model.
            system_prompt (Optional[str]): Optional system prompt.
            history_messages (Optional[List[Dict[str, Any]]]): Optional conversation history.
            **kwargs: Additional arguments for the model.

        Returns:
            str: The model's response.
        """
        messages = kwargs.get("messages")
        image_data = kwargs.get("image_data")

        if messages:
            if self.force_text_messages:
                messages = self._normalize_messages_to_text(messages)
                if self.direct_base_url:
                    return await self._invoke_direct_v2(messages)
            result = await self._invoke_messages(messages)
            return getattr(result, "content", str(result))

        if image_data:
            if self.force_text_messages:
                text = prompt or ""
                text = (text + "\n[IMAGE]").strip()
                from langchain_core.messages import HumanMessage, SystemMessage
                msgs = []
                if system_prompt:
                    msgs.append(SystemMessage(content=system_prompt))
                msgs.append(HumanMessage(content=text))
                result = await self.chat_model.ainvoke(msgs)
            else:
                msgs = build_messages(prompt, system_prompt, image_data)
                if self.force_text_messages and self.direct_base_url:
                    msgs = self._normalize_messages_to_text(msgs)
                    return await self._invoke_direct_v2(msgs)
                result = await self._invoke_messages(msgs)
            return getattr(result, "content", str(result))

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages: List[Union[AIMessage, HumanMessage, SystemMessage]] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        lc_messages.append(HumanMessage(content=prompt))

        result = await self.chat_model.ainvoke(lc_messages)
        return getattr(result, "content", str(result))

    async def _invoke_messages(self, messages: List[Dict]):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        lc_messages: List[Union[AIMessage, HumanMessage, SystemMessage]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                if isinstance(content, list):
                    if self.force_text_messages:
                        txt = self._content_list_to_text(content)
                        lc_messages.append(HumanMessage(content=txt))
                    else:
                        lc_messages.append(HumanMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=str(content)))
            else:
                lc_messages.append(AIMessage(content=str(content)))
        return await self.chat_model.ainvoke(lc_messages)

    def _normalize_messages_to_text(self, messages: List[Dict]) -> List[Dict]:
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                txt = self._content_list_to_text(content)
                out.append({"role": role, "content": txt})
            else:
                out.append({"role": role, "content": str(content)})
        return out

    def _content_list_to_text(self, content_list: List[Dict]) -> str:
        parts: List[str] = []
        for it in content_list:
            t = it.get("type")
            if t == "text":
                parts.append(str(it.get("text", "")))
            elif t == "image_url":
                parts.append("[IMAGE]")
            else:
                parts.append(str(it))
        return "\n".join([p for p in parts if p])

    async def _invoke_direct_v2(self, messages: List[Dict[str, str]]) -> str:
        import json
        import urllib.request
        import urllib.error

        url = self.direct_base_url.rstrip("/") + "/chat"
        payload = {
            "model": self.direct_model or "test",
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8")
                d = json.loads(body)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                d = json.loads(body)
            except Exception:
                raise
        except Exception as e:
            raise

        if isinstance(d, dict):
            ch = d.get("choices")
            if isinstance(ch, list) and ch:
                msg = ch[0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return json.dumps(d, ensure_ascii=False)


class LazyChatModelWrapper:
    def __init__(self, provider_cls, init_kwargs: Dict[str, Any]):
        self.provider_cls = provider_cls
        self.init_kwargs = init_kwargs
        self._client = None

    def __getstate__(self):
        """Support pickling by excluding the initialized client (which may have locks)"""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        """Restore state and reset client to None"""
        self.__dict__.update(state)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self.provider_cls(**self.init_kwargs)
        return self._client

    async def ainvoke(self, messages, **kwargs):
        return await self.client.ainvoke(messages, **kwargs)


class OfflineChatModel:
    async def ainvoke(self, lc_messages: List[Any]) -> Any:
        try:
            # Extract prompt and optional image from messages
            prompt_text = ""
            image_b64: Optional[str] = None
            for m in lc_messages:
                content = getattr(m, "content", None)
                if isinstance(content, list):
                    for it in content:
                        if isinstance(it, dict) and it.get("type") == "text":
                            prompt_text = str(it.get("text", ""))
                        elif isinstance(it, dict) and it.get("type") == "image_url":
                            url = it.get("image_url", {}).get("url")
                            if isinstance(url, str) and url.startswith("data:image"):
                                try:
                                    image_b64 = url.split(",", 1)[1]
                                except Exception:
                                    image_b64 = None
                elif isinstance(content, str):
                    prompt_text = content

            desc_parts: List[str] = []
            if prompt_text:
                desc_parts.append(f"Prompt: {prompt_text[:60]}")

            if image_b64:
                try:
                    from PIL import Image
                    img = Image.open(BytesIO(base64.b64decode(image_b64)))
                    w, h = img.size
                    mode = img.mode
                    # Compute simple statistics
                    try:
                        import numpy as np
                        arr = np.array(img.convert('L'))
                        brightness = float(arr.mean())
                        desc_parts.append(f"Image size: {w}x{h}, mode: {mode}, avg_brightness: {brightness:.1f}")
                    except Exception:
                        desc_parts.append(f"Image size: {w}x{h}, mode: {mode}")
                except Exception as e:
                    desc_parts.append(f"Image decode error: {e}")

            if not desc_parts:
                desc_parts.append("Offline description: no content")

            return " | ".join(desc_parts)
        except Exception as e:
            return f"Offline LLM error: {e}"


def build_llm(cfg: LLMProviderConfig) -> LLM:
    provider = cfg.provider.lower()

    if provider == "openai" or provider == "azure-openai" or provider == "openrouter":
        # Fail fast if API key is missing to allow fallback logic to work
        if not cfg.api_key and not os.environ.get("OPENAI_API_KEY"):
             raise ValueError("OpenAI API key is required but not provided in arguments or environment variables.")

        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            raise ValueError(f"OpenAI integration unavailable: {e}")

        init_kwargs: Dict[str, Any] = {"model": cfg.model}
        if cfg.api_key:
            os.environ.setdefault("OPENAI_API_KEY", cfg.api_key)
            init_kwargs["api_key"] = cfg.api_key

        if provider == "azure-openai":
            if cfg.api_base:
                init_kwargs["base_url"] = cfg.api_base.rstrip("/") + "/openai/v1"
            api_key = cfg.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            if api_key:
                os.environ.setdefault("AZURE_OPENAI_API_KEY", api_key)
                init_kwargs["api_key"] = api_key
        elif provider == "openrouter":
            if cfg.api_base:
                init_kwargs["base_url"] = cfg.api_base
            api_key = cfg.api_key or os.getenv("OPENROUTER_API_KEY")
            if api_key:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
                init_kwargs["api_key"] = api_key
            os.environ.setdefault("OPENAI_BASE_URL", init_kwargs.get("base_url", "https://openrouter.ai/api/v1"))
        else:
            if cfg.api_base:
                init_kwargs["base_url"] = cfg.api_base

        if cfg.timeout is not None:
            init_kwargs["timeout"] = cfg.timeout
        if cfg.max_retries is not None:
            init_kwargs["max_retries"] = cfg.max_retries

        if init_kwargs.get("base_url") and not init_kwargs.get("api_key"):
            init_kwargs["api_key"] = os.environ.get("OPENAI_API_KEY", "sk-local")
            os.environ.setdefault("OPENAI_API_KEY", init_kwargs["api_key"])

        chat = LazyChatModelWrapper(ChatOpenAI, init_kwargs)
        force = False
        direct_base_url = None
        direct_model = cfg.model
        if cfg.api_base and ("/v2" in cfg.api_base or ":8001" in cfg.api_base):
            force = True
            direct_base_url = cfg.api_base
        return LLM(
            chat,
            force_text_messages=force,
            direct_base_url=direct_base_url,
            direct_model=direct_model,
        )

    if provider == "ollama":
        try:
            # Use our robust wrapper instead of direct ChatOllama
            from raganything.llm.ollama_client import RobustOllamaClient
            provider_cls = RobustOllamaClient
        except ImportError:
            # Fallback or raise error if module missing (should be there)
            try:
                from langchain_ollama import ChatOllama
                provider_cls = ChatOllama
            except ImportError as e:
                raise ValueError(f"Ollama integration unavailable: {e}")

        init_kwargs: Dict[str, Any] = {"model": cfg.model}
        if cfg.api_base:
            init_kwargs["base_url"] = cfg.api_base
        if cfg.timeout is not None:
            init_kwargs["timeout"] = cfg.timeout
        if cfg.max_retries is not None:
            init_kwargs["max_retries"] = cfg.max_retries
            
        init_kwargs.update(cfg.extra or {})
        
        # Use LazyChatModelWrapper for picklability
        chat = LazyChatModelWrapper(provider_cls, init_kwargs)
            
        return LLM(chat)

    if provider == "mock":
        chat = OfflineChatModel()
        return LLM(chat)

    if provider == "offline":
        chat = OfflineChatModel()
        return LLM(chat)

    raise ValueError(f"Unsupported provider: {cfg.provider}")


def build_messages(
    prompt: str,
    system_prompt: Optional[str],
    image_base64: Optional[str],
) -> List[Dict[str, Any]]:
    base_system = (
        "You are a helpful assistant that can analyze both text and image content."
    )
    if system_prompt:
        sys = base_system + " " + system_prompt
    else:
        sys = base_system
    return [
        {"role": "system", "content": sys},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        },
    ]
