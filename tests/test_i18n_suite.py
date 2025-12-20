"""
国际化功能测试套件

合并了以下测试文件：
- test_i18n.py: 国际化系统测试

功能覆盖：
- 翻译加载和切换
- 日志消息翻译
- 变量插值翻译
- 缺失翻译回退机制
- 多线程安全性
- 完整工作流测试
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from raganything.i18n import (
    get_available_languages,
    get_current_language,
    get_system_language,
    init_i18n,
    switch_language,
)
from raganything.i18n_logger import get_i18n_logger


class TestI18N(unittest.TestCase):
    """国际化系统测试类"""

    def setUp(self):
        """测试前准备"""
        # 保存原始环境变量
        self.original_lang = os.environ.get('RAG_LANGUAGE')
        # 清理语言设置
        if 'RAG_LANGUAGE' in os.environ:
            del os.environ['RAG_LANGUAGE']

    def tearDown(self):
        """测试后清理"""
        # 恢复原始环境变量
        if self.original_lang is not None:
            os.environ['RAG_LANGUAGE'] = self.original_lang
        elif 'RAG_LANGUAGE' in os.environ:
            del os.environ['RAG_LANGUAGE']

    def test_system_language_detection(self):
        """测试系统语言检测"""
        # 测试环境变量优先
        with patch.dict(os.environ, {'RAG_LANGUAGE': 'zh_CN'}):
            lang = get_system_language()
            self.assertEqual(lang, 'zh_CN')

        # 测试默认情况
        if 'RAG_LANGUAGE' in os.environ:
            del os.environ['RAG_LANGUAGE']
        lang = get_system_language()
        self.assertIsInstance(lang, str)
        self.assertIn('_', lang)  # 应该包含下划线分隔符

    def test_language_initialization(self):
        """测试语言初始化"""
        # 测试中文初始化
        init_i18n('zh_CN')
        self.assertEqual(get_current_language(), 'zh_CN')

        # 测试英文初始化
        init_i18n('en_US')
        self.assertEqual(get_current_language(), 'en_US')

    def test_available_languages(self):
        """测试可用语言列表"""
        languages = get_available_languages()
        self.assertIsInstance(languages, list)
        self.assertIn('en_US', languages)
        self.assertIn('zh_CN', languages)

    def test_language_switching(self):
        """测试语言切换"""
        # 初始化英文
        init_i18n('en_US')
        self.assertEqual(get_current_language(), 'en_US')

        # 切换到中文
        success = switch_language('zh_CN')
        self.assertTrue(success)
        self.assertEqual(get_current_language(), 'zh_CN')

        # 切换回英文
        success = switch_language('en_US')
        self.assertTrue(success)
        self.assertEqual(get_current_language(), 'en_US')

    def test_invalid_language_fallback(self):
        """测试无效语言的回退机制"""
        # 测试不存在的语言
        success = switch_language('invalid_lang')
        self.assertTrue(success)  # 应该回退到默认翻译

    def test_translation_basic(self):
        """测试基本翻译功能"""
        # 测试中文翻译
        init_i18n('zh_CN')
        from raganything.i18n import _
        
        # 测试日志级别翻译
        self.assertEqual(_('INFO'), '信息')
        self.assertEqual(_('DEBUG'), '调试')
        self.assertEqual(_('WARNING'), '警告')
        self.assertEqual(_('ERROR'), '错误')
        self.assertEqual(_('CRITICAL'), '严重错误')

        # 测试英文翻译
        init_i18n('en_US')
        self.assertEqual(_('INFO'), 'INFO')
        self.assertEqual(_('DEBUG'), 'DEBUG')

    def test_translation_with_variables(self):
        """测试带变量的翻译"""
        init_i18n('zh_CN')
        from raganything.i18n import _
        
        # 测试变量插值
        template = "Processing file: {filename}"
        translated = _(template).format(filename="test.txt")
        # 由于模板本身没有翻译，应该返回原模板
        self.assertEqual(translated, "Processing file: test.txt")

    def test_logger_translation(self):
        """测试日志翻译"""
        # 测试中文日志
        init_i18n('zh_CN')
        logger = get_i18n_logger()
        
        # 测试日志级别翻译
        self.assertEqual(logger._translate_level('INFO'), '信息')
        self.assertEqual(logger._translate_level('DEBUG'), '调试')
        self.assertEqual(logger._translate_level('WARNING'), '警告')
        self.assertEqual(logger._translate_level('ERROR'), '错误')
        self.assertEqual(logger._translate_level('CRITICAL'), '严重错误')

        # 测试英文日志
        init_i18n('en_US')
        logger = get_i18n_logger()
        self.assertEqual(logger._translate_level('INFO'), 'INFO')
        self.assertEqual(logger._translate_level('DEBUG'), 'DEBUG')

    def test_logger_message_translation(self):
        """测试日志消息翻译"""
        init_i18n('zh_CN')
        logger = get_i18n_logger()
        
        # 测试日志消息翻译
        self.assertEqual(logger._translate_message('RAGAnything CLI logging initialized'), 
                        'RAGAnything CLI日志已初始化')

    def test_missing_translation_fallback(self):
        """测试缺失翻译的回退机制"""
        init_i18n('zh_CN')
        from raganything.i18n import _
        
        # 测试不存在的翻译键
        untranslated = "This message does not exist in translations"
        result = _(untranslated)
        self.assertEqual(result, untranslated)  # 应该返回原文

    def test_logger_api_compatibility(self):
        """测试日志API兼容性"""
        init_i18n('zh_CN')
        logger = get_i18n_logger()
        
        # 测试所有日志级别方法存在
        self.assertTrue(hasattr(logger, 'debug'))
        self.assertTrue(hasattr(logger, 'info'))
        self.assertTrue(hasattr(logger, 'warning'))
        self.assertTrue(hasattr(logger, 'error'))
        self.assertTrue(hasattr(logger, 'critical'))

        # 测试日志方法可调用
        try:
            logger.debug("Test debug message")
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            logger.critical("Test critical message")
        except Exception as e:
            self.fail(f"Logger API compatibility test failed: {e}")

    def test_multithread_safety(self):
        """测试多线程安全性"""
        results = []

        def thread_function(lang, message):
            try:
                init_i18n(lang)
                logger = get_i18n_logger()
                logger.info(message)
                results.append(f"{lang}: {get_current_language()}")
            except Exception as e:
                results.append(f"{lang}: ERROR - {e}")

        # 启动多个线程
        threads = []
        threads.append(threading.Thread(target=thread_function, args=('zh_CN', '中文消息')))
        threads.append(threading.Thread(target=thread_function, args=('en_US', 'English message')))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # 验证所有线程都成功执行
        self.assertEqual(len(results), 2)
        self.assertNotIn('ERROR', str(results))


class TestI18NIntegration(unittest.TestCase):
    """国际化集成测试类"""

    def test_full_workflow_zh_cn(self):
        """测试完整的中文工作流"""
        # 设置中文环境
        os.environ['RAG_LANGUAGE'] = 'zh_CN'
        
        # 初始化i18n
        init_i18n()
        
        # 获取i18n日志器
        logger = get_i18n_logger()
        
        # 测试日志输出
        logger.info("RAGAnything CLI logging initialized")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        # 验证当前语言
        self.assertEqual(get_current_language(), 'zh_CN')

    def test_full_workflow_en_us(self):
        """测试完整的英文工作流"""
        # 设置英文环境
        os.environ['RAG_LANGUAGE'] = 'en_US'
        
        # 初始化i18n
        init_i18n()
        
        # 获取i18n日志器
        logger = get_i18n_logger()
        
        # 测试日志输出
        logger.info("RAGAnything CLI logging initialized")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        
        # 验证当前语言
        self.assertEqual(get_current_language(), 'en_US')

    def test_language_switching_workflow(self):
        """测试语言切换工作流"""
        # 初始化为英文
        init_i18n('en_US')
        logger = get_i18n_logger()
        
        # 记录英文日志
        logger.info("Starting in English")
        
        # 切换到中文
        switch_language('zh_CN')
        logger = get_i18n_logger()  # 重新获取日志器
        
        # 记录中文日志
        logger.info("Switched to Chinese")
        
        # 验证切换成功
        self.assertEqual(get_current_language(), 'zh_CN')


if __name__ == '__main__':
    # 设置测试环境
    os.environ['RAG_LOG_LEVEL'] = 'ERROR'  # 减少测试时的日志输出
    
    # 运行测试
    unittest.main(verbosity=2)