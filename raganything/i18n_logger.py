"""
国际化日志包装器，为现有日志系统提供多语言支持。

功能特性：
- 包装现有logger，提供国际化消息支持
- 支持日志级别名称的翻译
- 保持与原有logger API的兼容性
"""

from __future__ import annotations

from typing import Any, Optional
from loguru import logger as _original_logger


class I18nLogger:
    """国际化日志包装器。
    
    该类包装了原有的logger，在记录日志时自动翻译消息内容。
    保持与原有logger API的完全兼容性。
    """
    
    def __init__(self, logger=_original_logger):
        """初始化国际化日志包装器。
        
        Args:
            logger: 要包装的logger实例
        """
        self._logger = logger
    
    def _translate_message(self, message: str) -> str:
        """翻译日志消息。
        
        Args:
            message: 原始消息
            
        Returns:
            str: 翻译后的消息
        """
        try:
            from raganything.i18n import _
            return _(message)
        except ImportError:
            return message
    
    def _translate_level(self, level: str) -> str:
        """翻译日志级别。
        
        Args:
            level: 原始级别
            
        Returns:
            str: 翻译后的级别
        """
        try:
            from raganything.i18n import _
            return _(level)
        except ImportError:
            return level
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """记录调试日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.debug(translated_msg, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """记录信息日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.info(translated_msg, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """记录警告日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.warning(translated_msg, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """记录错误日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.error(translated_msg, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """记录严重错误日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.critical(translated_msg, *args, **kwargs)
    
    def success(self, message: str, *args, **kwargs) -> None:
        """记录成功日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.success(translated_msg, *args, **kwargs)
    
    def trace(self, message: str, *args, **kwargs) -> None:
        """记录跟踪日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.trace(translated_msg, *args, **kwargs)
    
    def log(self, level: str, message: str, *args, **kwargs) -> None:
        """记录指定级别的日志。
        
        Args:
            level: 日志级别
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.log(level, translated_msg, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """记录异常日志。
        
        Args:
            message: 日志消息
            *args: 格式化参数
            **kwargs: 其他参数
        """
        translated_msg = self._translate_message(message)
        self._logger.exception(translated_msg, *args, **kwargs)
    
    def opt(self, **kwargs) -> Any:
        """获取logger选项。
        
        Args:
            **kwargs: 选项参数
            
        Returns:
            Any: logger选项
        """
        return self._logger.opt(**kwargs)
    
    def bind(self, **kwargs) -> "I18nLogger":
        """绑定额外上下文。
        
        Args:
            **kwargs: 上下文参数
            
        Returns:
            I18nLogger: 新的logger实例
        """
        new_logger = self._logger.bind(**kwargs)
        return I18nLogger(new_logger)
    
    def contextualize(self, **kwargs) -> Any:
        """添加上下文。
        
        Args:
            **kwargs: 上下文参数
            
        Returns:
            Any: 上下文管理器
        """
        return self._logger.contextualize(**kwargs)
    
    def add(self, *args, **kwargs) -> int:
        """添加sink。
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            int: sink ID
        """
        return self._logger.add(*args, **kwargs)
    
    def remove(self, *args, **kwargs) -> None:
        """移除sink。
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        self._logger.remove(*args, **kwargs)
    
    def complete(self) -> Any:
        """完成日志配置。
        
        Returns:
            Any: logger实例
        """
        return self._logger.complete()
    
    @property
    def level(self) -> Any:
        """获取当前日志级别。
        
        Returns:
            Any: 日志级别
        """
        return self._logger.level


# 创建全局国际化logger实例
logger = I18nLogger()


def get_i18n_logger() -> I18nLogger:
    """获取国际化logger实例。
    
    Returns:
        I18nLogger: 国际化logger实例
    """
    return logger


def refresh_logger_translations() -> None:
    """刷新logger翻译。
    
    当语言切换时调用此函数刷新翻译。
    """
    global logger
    # 重新创建logger实例以确保使用最新的翻译函数
    logger = I18nLogger()