# 日志模块使用说明

本项目使用 `loguru` 实现统一的日志模块，位于 `raganything/logger.py`，提供控制台彩色输出与文件轮转输出两个通道。

## 初始化与获取

- 模块导入即完成初始化：
  ```python
  from raganything.logger import logger
  logger.info("系统启动")
  ```

- 需要重配置时使用：
  ```python
  from raganything.logger import init_logger
  init_logger(level="DEBUG", rotation="00:00", retention="7 days")
  ```

## 输出通道

- 控制台：彩色格式，显示时间、级别、模块名、函数与行号、消息
- 文件：写入项目根目录 `logs/raganything.log`，每日轮转（默认），支持保留策略

## 环境变量

- `RAG_LOG_LEVEL`：默认 `INFO`
- `RAG_LOG_DIR`：默认项目根目录 `logs/`
- `RAG_LOG_RETENTION`：默认 `7 days`

## 级别

支持 `DEBUG/INFO/WARNING/ERROR/CRITICAL`。

## 多线程

默认开启 `enqueue=True`，保证多线程写入顺序与安全。测试覆盖见 `tests/`。

## 与原日志替换

- 所有 `print` 或 `logging` 已逐步替换为统一 `logger`。
- 新代码请直接依赖 `raganything.logger.logger`。

## 注意事项

- 生产环境建议保留策略 `retention` 与轮转策略 `rotation` 按需调整。
- 谨慎开启 `diagnose=True`，可能泄漏敏感变量值；本模块默认关闭。

