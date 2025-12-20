"""
国际化支持模块，基于gettext实现多语言日志系统。

功能特性：
- 自动检测系统语言环境
- 支持日志消息的国际化
- 提供_()函数用于字符串翻译
- 支持动态语言切换

环境变量：
- RAG_LANGUAGE: 设置语言环境，如zh_CN、en_US等
"""

from __future__ import annotations

import gettext
import locale
import os
from pathlib import Path
from typing import Optional

from .i18n_logger import get_i18n_logger

# 获取当前模块目录
LOCALE_DIR = Path(__file__).parent.parent / "locale"
DOMAIN = "raganything"

# 全局翻译器实例
_translator = None

# 当前语言环境
current_language = None


class Translator:
    """翻译器类，封装翻译功能。"""
    
    def __init__(self):
        self._translations = {}
        self._gettext_translator = None
        
    def gettext(self, text: str) -> str:
        """获取翻译文本。"""
        if self._gettext_translator:
            return self._gettext_translator.gettext(text)
        else:
            return self._translations.get(text, text)
    
    def set_gettext_translator(self, translator):
        """设置gettext翻译器。"""
        self._gettext_translator = translator
        self._translations = {}  # 清空字典翻译
        
    def set_translations_dict(self, translations: dict[str, str]):
        """设置翻译字典。"""
        self._translations = translations
        self._gettext_translator = None  # 清空gettext翻译器


def get_system_language() -> str:
    """获取系统默认语言。
    
    Returns:
        str: 语言代码，如zh_CN、en_US等
    """
    try:
        # 尝试从环境变量获取
        lang = os.getenv("RAG_LANGUAGE")
        if lang:
            return lang
            
        # 获取系统locale (使用现代API，避免弃用警告)
        try:
            system_lang = locale.getlocale()[0]
            if system_lang:
                return system_lang.replace("_", "-").replace("-", "_")
        except (locale.Error, AttributeError):
            # 如果getlocale失败，尝试备用方法
            try:
                system_lang = locale.getdefaultlocale()[0]
                if system_lang:
                    return system_lang.replace("_", "-").replace("-", "_")
            except (locale.Error, AttributeError):
                pass
            
        # 默认返回英文
        return "en_US"
    except Exception:
        return "en_US"


def init_i18n(language: Optional[str] = None) -> None:
    """初始化国际化系统。
    
    Args:
        language: 指定语言代码，如果为None则自动检测
    """
    global _translator, current_language
    
    if language is None:
        language = get_system_language()
    
    current_language = language
    
    # 创建新的翻译器实例
    _translator = Translator()
    
    try:
        # 首先尝试使用gettext
        lang = gettext.translation(
            DOMAIN,
            localedir=str(LOCALE_DIR),
            languages=[language],
            fallback=True
        )
        
        # 设置gettext翻译器
        _translator.set_gettext_translator(lang)
        
        # 尝试安装到内建命名空间
        try:
            lang.install(names=['gettext', 'ngettext'])
        except Exception:
            pass
        
        # 测试翻译是否工作
        test_result = lang.gettext('INFO')
        if test_result == 'INFO' and language == 'zh_CN':
            # 如果中文翻译返回英文，说明gettext有问题，强制使用polib
            raise Exception("gettext not working properly for Chinese")
            
    except Exception as e:
        # 如果gettext失败或工作不正常，使用polib直接加载.mo文件
        logger = get_i18n_logger()
        logger.warning(_("gettext failed for {}: {}, trying polib fallback").format(language, e))
        try:
            import polib
            mo_file = LOCALE_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.mo"
            if mo_file.exists():
                mo = polib.mofile(str(mo_file))
                # 创建翻译字典
                translations = {}
                for entry in mo:
                    if entry.msgstr:
                        translations[entry.msgid] = entry.msgstr
                
                _translator.set_translations_dict(translations)
                logger.info(_("Loaded translations for {} using polib").format(language))
            else:
                logger.warning(_("MO file not found for {}: {}").format(language, mo_file))
        except Exception as e2:
            # 如果翻译文件不存在，使用默认实现
            logger.error(_("Failed to load translation for {}: {}, fallback to polib failed: {}").format(language, e, e2))
    
    # 刷新logger翻译
    try:
        from raganything.i18n_logger import refresh_logger_translations
        refresh_logger_translations()
    except Exception:
        pass
            
    except Exception as e:
        # 如果gettext失败，尝试使用polib直接加载.mo文件
        try:
            import polib
            mo_file = LOCALE_DIR / language / "LC_MESSAGES" / f"{DOMAIN}.mo"
            if mo_file.exists():
                mo = polib.mofile(str(mo_file))
                # 创建翻译字典
                translations = {}
                for entry in mo:
                    if entry.msgstr:
                        translations[entry.msgid] = entry.msgstr
                
                # 定义翻译函数 - 使用新的闭包确保正确的字典引用
                def translate_func(text):
                    return translations.get(text, text)
                
                _ = translate_func
                logger.info(_("Loaded translations for {} using polib").format(language))
            else:
                logger.warning(_("MO file not found for {}: {}").format(language, mo_file))
                def identity_func2(x):
                    return x
                _ = identity_func2
        except Exception as e2:
            # 如果翻译文件不存在，使用默认实现
            logger.error(_("Failed to load translation for {}: {}, fallback to polib failed: {}").format(language, e, e2))
            def identity_func3(x):
                return x
            _ = identity_func3
    
    # 刷新logger翻译
    try:
        from raganything.i18n_logger import refresh_logger_translations
        refresh_logger_translations()
    except Exception:
        pass


def get_available_languages() -> list[str]:
    """获取可用的语言列表。
    
    Returns:
        list[str]: 可用的语言代码列表
    """
    available = []
    try:
        if LOCALE_DIR.exists():
            for lang_dir in LOCALE_DIR.iterdir():
                if lang_dir.is_dir():
                    mo_file = lang_dir / "LC_MESSAGES" / f"{DOMAIN}.mo"
                    if mo_file.exists():
                        available.append(lang_dir.name)
    except Exception:
        pass
    
    return available if available else ["en_US"]


def switch_language(language: str) -> bool:
    """切换语言环境。
    
    Args:
        language: 目标语言代码
        
    Returns:
        bool: 是否切换成功
    """
    try:
        init_i18n(language)
        return True
    except Exception:
        return False


def get_current_language() -> str:
    """获取当前语言环境。
    
    Returns:
        str: 当前语言代码
    """
    return current_language or get_system_language()


def _(text: str) -> str:
    """翻译函数。"""
    if _translator:
        return _translator.gettext(text)
    return text


# 初始化国际化系统
init_i18n()