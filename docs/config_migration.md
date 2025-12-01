# 配置文件迁移说明（.env → TOML）

本项目支持从 `.env` 环境变量迁移到 `config.toml` 的嵌套结构配置，既提升可读性，也便于分层管理。迁移后功能保持等价，且仍保留对原有环境变量的兼容读取。

## 新配置结构

顶层采用多级 section，核心库使用的配置位于 `raganything.*` 命名空间：

- `raganything.parsing`：解析器与解析方法
- `raganything.multimodal`：多模态处理开关
- `raganything.batch`：批处理参数
- `raganything.context`：上下文抽取设置
- `raganything.llm`：LLM 提供方配置
- `raganything.embedding`：嵌入模型配置
- `raganything.vision`：视觉模型配置
- `directory`：工作目录与解析输出目录

其余如 `server`、`auth`、`ssl`、`storage`、`postgres` 等为运行环境或外部服务参考配置，库本身不直接读取。

## 使用方式

1. 将 `env.example` 复制为 `config.toml` 并按需修改值
2. 或设置环境变量 `CONFIG_TOML=/absolute/path/to/config.toml`
3. 若不提供 TOML，库会继续读取环境变量以保持兼容
4. 服务端工程参考：通过 `raganything.server_config.load_server_configs()` 读取 `server`、`ssl`、`api` sections，并优先使用环境变量覆盖敏感字段（如 `LIGHTRAG_API_KEY`）

## 字段映射示例

| 旧环境变量 | 新 TOML 路径 |
| --- | --- |
| `WORKING_DIR` | `directory.working_dir` |
| `OUTPUT_DIR` | `directory.parser_output_dir` |
| `PARSER` | `raganything.parsing.parser` |
| `PARSE_METHOD` | `raganything.parsing.parse_method` |
| `DISPLAY_CONTENT_STATS` | `raganything.parsing.display_content_stats` |
| `ENABLE_IMAGE_PROCESSING` | `raganything.multimodal.enable_image_processing` |
| `ENABLE_TABLE_PROCESSING` | `raganything.multimodal.enable_table_processing` |
| `ENABLE_EQUATION_PROCESSING` | `raganything.multimodal.enable_equation_processing` |
| `MAX_CONCURRENT_FILES` | `raganything.batch.max_concurrent_files` |
| `SUPPORTED_FILE_EXTENSIONS` | `raganything.batch.supported_file_extensions` (数组) |
| `RECURSIVE_FOLDER_PROCESSING` | `raganything.batch.recursive_folder_processing` |
| `CONTEXT_WINDOW` | `raganything.context.context_window` |
| `CONTEXT_MODE` | `raganything.context.context_mode` |
| `MAX_CONTEXT_TOKENS` | `raganything.context.max_context_tokens` |
| `INCLUDE_HEADERS` | `raganything.context.include_headers` |
| `INCLUDE_CAPTIONS` | `raganything.context.include_captions` |
| `CONTEXT_FILTER_CONTENT_TYPES` | `raganything.context.context_filter_content_types` (数组) |
| `CONTENT_FORMAT` | `raganything.context.content_format` |
| `LLM_PROVIDER` | `raganything.llm.provider` |
| `LLM_MODEL` | `raganything.llm.model` |
| `LLM_API_BASE` | `raganything.llm.api_base` |
| `LLM_API_KEY` | `raganything.llm.api_key` |
| `LLM_TIMEOUT` | `raganything.llm.timeout` |
| `LLM_MAX_RETRIES` | `raganything.llm.max_retries` |
| `EMBEDDING_PROVIDER` | `raganything.embedding.provider` |
| `EMBEDDING_MODEL` | `raganything.embedding.model` |
| `EMBEDDING_API_BASE` | `raganything.embedding.api_base` |
| `EMBEDDING_API_KEY` | `raganything.embedding.api_key` |
| `EMBEDDING_DIM` | `raganything.embedding.dim` |
| `EMBEDDING_FUNC_MAX_ASYNC` | `raganything.embedding.func_max_async` |
| `EMBEDDING_BATCH_NUM` | `raganything.embedding.batch_num` |
| `VISION_PROVIDER` | `raganything.vision.provider` |
| `VISION_MODEL` | `raganything.vision.model` |
| `VISION_API_BASE` | `raganything.vision.api_base` |
| `VISION_API_KEY` | `raganything.vision.api_key` |
| `VISION_TIMEOUT` | `raganything.vision.timeout` |
| `VISION_MAX_RETRIES` | `raganything.vision.max_retries` |

服务端相关：

| 环境变量 | 新 TOML 路径 |
| --- | --- |
| `HOST` / `SERVER_HOST` | `server.host` |
| `PORT` / `SERVER_PORT` | `server.port` |
| `WORKERS` / `SERVER_WORKERS` | `server.workers` |
| `CORS_ORIGINS` | `server.cors_origins` (数组) |
| `SSL` | `ssl.enabled` |
| `SSL_CERTFILE` | `ssl.certfile` |
| `SSL_KEYFILE` | `ssl.keyfile` |
| `LIGHTRAG_API_KEY` | `api.lightrag_api_key` |
| `WHITELIST_PATHS` | `api.whitelist_paths` (数组) |

> 兼容说明：若同时存在 TOML 与环境变量，TOML 会覆盖同名字段；未在 TOML 中出现的字段继续由环境变量提供默认值。

## 版本与兼容

- Python 3.11 以上使用标准库 `tomllib` 解析；Python 3.10 自动使用 `tomli`（已加入依赖）
- 旧字段 `MINERU_PARSE_METHOD` 仍受支持，会触发弃用警告并映射到 `raganything.parsing.parse_method`
- `RAGAnythingConfig` 保留原有属性访问接口（如 `config.parse_method`、`config.working_dir` 等），内部已映射到嵌套结构，确保不破坏外部代码
- 服务端配置读取通过 `raganything.server_config`，敏感字段（如 `LIGHTRAG_API_KEY`）优先从环境变量注入；TOML 可留空以降低泄露风险

## 校验与类型

加载后会进行如下校验：

- `parsing.parser` ∈ {`mineru`, `docling`}
- `parsing.parse_method` ∈ {`auto`, `ocr`, `txt`}
- `context.context_mode` ∈ {`page`, `chunk`}
- 数值边界：并发、窗口、token、重试、超时均进行下限检查

## 性能与运行时

类型检查与校验为轻量逻辑，不增加显著运行时开销；仅在配置对象创建时执行一次。

## 迁移步骤摘要

1. 复制并编辑 `env.example` 为 `config.toml`
2. 在部署环境设置 `CONFIG_TOML=/path/to/config.toml` 或将文件置于项目根目录
3. 保留必要的环境变量（如密钥、路径），逐步迁移至 TOML 中对应的 section
4. 运行测试确保兼容性与功能等价

## FAQ

- 不提供 TOML 是否可用？可用，仍按原有环境变量工作
- 两者同时提供时如何处理？同名字段以 TOML 为准，其余从环境变量补齐
