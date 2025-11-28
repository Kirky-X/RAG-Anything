# RAG-Anything LangChain LLM 模块

## 概述
- 统一封装 OpenAI、Ollama、OpenRouter 三类模型接入，使用 LangChain 作为适配层。
- 提供 `LangChainLLM` 可直接作为 `llm_model_func` / `vision_model_func` 注入现有管线。
- 支持通过环境变量或配置对象动态配置端点、认证、超时与重试。

## 入口与构建
- `raganything.llm.llm.build_llm(cfg)`：根据 `LLMProviderConfig` 返回 `LLM`。
- `raganything.llm.embedding.build_embedding_func(...)`：返回 `EmbeddingFunc`，用于 LightRAG。

## 统一调用接口
- `await llm(prompt, system_prompt=None, history_messages=[], **kwargs)`
  - 纯文本：传入 `prompt` 与可选 `system_prompt`
  - 多模态：传入 `messages`（OpenAI 风格），或传入 `image_data`（Base64），自动构造消息

## 环境变量
- 文本 LLM：`LLM_PROVIDER`、`LLM_MODEL`、`LLM_API_BASE`、`LLM_API_KEY`、`LLM_TIMEOUT`、`LLM_MAX_RETRIES`
- 视觉 LLM：`VISION_PROVIDER`、`VISION_MODEL`、`VISION_API_BASE`、`VISION_API_KEY`、`VISION_TIMEOUT`、`VISION_MAX_RETRIES`
- 嵌入：`EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`EMBEDDING_API_BASE`、`EMBEDDING_API_KEY`、`EMBEDDING_DIM`、`EMBEDDING_FUNC_MAX_ASYNC`、`EMBEDDING_BATCH_NUM`

## 使用示例
```python
from raganything.llm import LLMProviderConfig, build_langchain_llm
cfg = LLMProviderConfig(provider="openrouter", model="openrouter/anthropic/claude-3.5-sonnet", api_base="https://openrouter.ai/api/v1", api_key="sk-...")
llm = build_llm(cfg)
result = await llm("Hello", system_prompt="You are helpful")
```

多模态：
```python
msgs = [
  {"role": "system", "content": "You analyze images"},
  {"role": "user", "content": [
    {"type": "text", "text": "What is in the picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
  ]}
]
result = await llm("", messages=msgs)
```

## 与现有管线集成
- 在 `RAGAnything._maybe_build_llm_functions()` 中，若未显式提供 `llm_model_func`/`vision_model_func`，将自动按配置构建。
- 现有 `QueryMixin`、`ModalProcessor` 保持不变，兼容新的统一接口。

## 错误处理与配置校验
- 构建器在依赖缺失时抛出 `ValueError`，日志模块记录警告，不影响其他功能。
- 调用失败在上层按既有模式捕获并记录。

## 兼容非标准端点（/v2/chat）
- 某些本地或代理端点只接受“纯字符串的 messages.content”，不支持多模态 `content` 数组，也不支持单独 `prompt` 字段，典型路径如 `http://localhost:8001/v2/chat`。
- 为兼容此类端点，`LangChainLLM` 内置“文本化降级 + 直连调用”机制：
  - 判定：当 `api_base` 包含 `/v2` 或端口 `8001` 时自动启用；或通过构建器 `extra` 指示（内部已按端点启用）。
  - 降级：将 `messages` 中的数组型 `content` 聚合为单一字符串，并以 `[IMAGE]` 占位符标记图像内容；保留 `system/user` 角色。
  - 直连：降级后直接以 HTTP POST 访问 `api_base + '/chat'`，请求体包含 `model/messages/max_tokens/temperature`，返回优先提取 `choices[0].message.content`。
- 使用方式：
  ```python
  from raganything.llm import LLMProviderConfig, build_langchain_llm
  cfg = LLMProviderConfig(provider='openai', model='gpt-4o-mini', api_base='http://localhost:8001/v2')
  llm = build_langchain_llm(cfg)  # 自动启用降级与直连

  msgs = [
      {"role": "system", "content": "You are concise."},
      {"role": "user", "content": [
          {"type": "text", "text": "说一个2+2结果，只输出数字"}
      ]},
  ]
  result = await llm("", messages=msgs)  # 输出："4"
  ```
- 注意事项：
  - 此模式下 `OPENAI_API_KEY` 不要求；适配器会为代理场景设置安全的本地占位值。
  - 端点若支持标准 OpenAI Chat Completions（字符串 `messages.content`），仍可保持结构化返回（`choices/...`）。
  - 若端点不接受数组型 content，也不接受 `prompt` 字段，必须走降级后的字符串 `messages`。
