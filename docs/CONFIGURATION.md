# 配置与环境变量

## 总览
- 模块支持通过环境变量或显式传参配置 LLM、VLM、Embedding 的端点与认证、超时与重试。

## 关键环境变量
- `LLM_PROVIDER`：`openai` | `ollama` | `openrouter` | `azure-openai`
- `LLM_MODEL`：模型名
- `LLM_API_BASE`：服务端点 URL
- `LLM_API_KEY`：认证密钥
- `LLM_TIMEOUT`：超时秒数
- `LLM_MAX_RETRIES`：最大重试次数

- `VISION_PROVIDER`、`VISION_MODEL`、`VISION_API_BASE`、`VISION_API_KEY`、`VISION_TIMEOUT`、`VISION_MAX_RETRIES`

- `EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`EMBEDDING_API_BASE`、`EMBEDDING_API_KEY`、`EMBEDDING_DIM`、`EMBEDDING_FUNC_MAX_ASYNC`、`EMBEDDING_BATCH_NUM`

## 使用方式
- 通过 `.env` 或进程环境注入上述变量，无需改动代码即可切换不同 sink。

## 兼容性
- 未安装 LangChain 对应适配包时，构建器会抛出错误但不影响非 LLM 功能；请按需要安装 `langchain-openai` 或 `langchain-ollama`。

