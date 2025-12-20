#!/usr/bin/env python3
"""
RAG-Anything 国际化工具集

统一的国际化处理工具，提供翻译字符串提取、编译、异常处理等功能。
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple, Optional
import polib


class I18NExtractor:
    """翻译字符串提取器"""
    
    def __init__(self):
        self.messages: Set[str] = set()
        self.logger_messages: Set[str] = set()
        self.exception_messages: Set[str] = set()
    
    def extract_from_file(self, file_path: Path) -> None:
        """从单个Python文件中提取翻译字符串"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 提取 _() 函数调用的参数
            pattern = r'_\(["\'](.*?)["\']\)'
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            self.messages.update(matches)
            
            # 提取logger消息
            logger_pattern = r'logger\.(debug|info|warning|error|critical)\(["\'](.*?)["\']\)'
            logger_matches = re.findall(logger_pattern, content, re.IGNORECASE)
            self.logger_messages.update([match[1] for match in logger_matches])
            
            # 提取异常消息
            exception_patterns = [
                (r'raise\s+(\w+)\(["\'](.*?)["\']\)', 'string'),
                (r'raise\s+HTTPException\(.*detail=["\'](.*?)["\']\)', 'http_detail'),
                (r'raise\s+HTTPException\(.*detail=f["\'](.*?)["\']\)', 'http_f_detail'),
            ]
            
            for pattern, pattern_type in exception_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if pattern_type.startswith('http'):
                    self.exception_messages.update(matches)
                else:
                    self.exception_messages.update([match[1] for match in matches])
                    
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")


class I18NCompiler:
    """翻译文件编译器"""
    
    @staticmethod
    def compile_po_to_mo(po_file_path: str, mo_file_path: Optional[str] = None) -> bool:
        """将.po文件编译为.mo文件"""
        po_path = Path(po_file_path)
        
        if not po_path.exists():
            print(f"错误: 找不到PO文件: {po_file_path}")
            return False
        
        if mo_file_path is None:
            mo_file_path = po_path.with_suffix('.mo')
        
        try:
            po = polib.pofile(po_file_path)
            po.save_as_mofile(mo_file_path)
            print(f"成功编译 {po_file_path} -> {mo_file_path}")
            print(f"编译了 {len(po)} 条消息")
            return True
        except Exception as e:
            print(f"编译 {po_file_path} 时出错: {e}")
            return False
    
    @staticmethod
    def compile_all_translations(locale_dir: str = "locale") -> None:
        """编译locale目录下的所有.po文件"""
        locale_path = Path(locale_dir)
        
        if not locale_path.exists():
            print(f"错误: 找不到locale目录: {locale_dir}")
            return
        
        po_files = list(locale_path.rglob("*.po"))
        
        if not po_files:
            print("未找到任何.po文件")
            return
        
        success_count = 0
        for po_file in po_files:
            if I18NCompiler.compile_po_to_mo(str(po_file)):
                success_count += 1
        
        print(f"\n编译完成: {success_count}/{len(po_files)} 个文件编译成功")


class I18NExceptionHandler:
    """异常消息国际化处理器"""
    
    @staticmethod
    def get_exception_translations() -> Dict[str, str]:
        """获取异常消息的翻译映射"""
        return {
            # 常用异常消息
            "Audio conversion timed out": "音频转换超时",
            "Audio dependencies (funasr, pydub) are not installed.": "音频依赖库 (funasr, pydub) 未安装。",
            "Empty response received from model": "从模型接收到空响应",
            "Invalid API key": "无效的API密钥",
            "Invalid context_mode": "无效的context_mode",
            "Invalid file path: outside root directory": "无效的文件路径：超出根目录范围",
            "Invalid parse_method": "无效的parse_method",
            "Invalid parser": "无效的解析器",
            "Invalid prefix": "无效的前缀",
            "LightRAG not initialized": "LightRAG未初始化",
            "Missing required fields in entity_info": "entity_info中缺少必填字段",
            "Missing required fields in response": "响应中缺少必填字段",
            "Parsing failed: No content was extracted": "解析失败：未提取到内容",
            "RAGAnything not initialized": "RAGAnything未初始化",
            "Subclasses must implement this method": "子类必须实现此方法",
            "Unknown error in RobustOllamaClient": "RobustOllamaClient发生未知错误",
            "context_window must be >= 0": "context_window必须 >= 0",
            "gettext not working properly for Chinese": "gettext中文支持未正常工作",
            "logging.backup_count must be >= 0": "logging.backup_count必须 >= 0",
            "logging.max_bytes must be >= 0": "logging.max_bytes必须 >= 0",
            "max_concurrent_files must be >= 1": "max_concurrent_files必须 >= 1",
            "max_context_tokens must be > 0": "max_context_tokens必须 > 0",
            "max_retries must be >= 0": "max_retries必须 >= 0",
            "parse_document must be implemented by subclasses": "子类必须实现parse_document方法",
            "parse_image must be implemented by subclasses": "子类必须实现parse_image方法",
            "parse_pdf must be implemented by subclasses": "子类必须实现parse_pdf方法",
            "pydub is not installed; audio analysis unavailable.": "pydub未安装；音频分析不可用。",
            "pydub is not installed; audio conversion unavailable.": "pydub未安装；音频转换不可用。",
            "pydub not found. Install dependencies.": "pydub未找到。请安装依赖项。",
            "timeout must be > 0": "timeout必须 > 0",
        }
    
    @staticmethod
    def get_key_translations() -> Dict[str, str]:
        """获取关键翻译映射"""
        return {
            # 日志级别
            "DEBUG": "调试",
            "INFO": "信息", 
            "WARNING": "警告",
            "ERROR": "错误",
            "CRITICAL": "严重",
            "SUCCESS": "成功",
            "TRACE": "跟踪",
            
            # 常用日志消息
            "RAGAnything CLI logging initialized": "RAGAnything CLI日志已初始化",
            "RAGAnything initialized successfully": "RAGAnything初始化成功",
            "Processing document": "正在处理文档",
            "Document processed successfully": "文档处理成功",
            "Query completed": "查询完成",
            "Error processing document": "处理文档时出错",
            "Initializing RAGAnything": "正在初始化RAGAnything",
            "Loading configuration": "正在加载配置",
            "Configuration loaded": "配置已加载",
            "Starting server": "正在启动服务器",
            "Server started": "服务器已启动",
            "Shutting down": "正在关闭",
            "Shutdown complete": "关闭完成",
            
            # 通用消息
            "Success": "成功",
            "Failed": "失败",
            "Error": "错误",
            "Warning": "警告",
            "Info": "信息",
        }


def extract_translations(project_root: Optional[str] = None) -> None:
    """提取项目中的所有翻译字符串"""
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)
    
    raganything_dir = project_root / 'raganything'
    
    if not raganything_dir.exists():
        print(f"错误: 找不到raganything目录: {raganything_dir}")
        return
    
    extractor = I18NExtractor()
    
    # 扫描所有Python文件
    python_files = list(raganything_dir.rglob("*.py"))
    
    for file_path in python_files:
        # 跳过测试目录和工具目录
        if any(skip_dir in str(file_path) for skip_dir in ['tests', 'tools', '__pycache__', '.git']):
            continue
        extractor.extract_from_file(file_path)
    
    # 合并所有消息
    all_messages = extractor.messages | extractor.logger_messages | extractor.exception_messages
    
    print(f"提取到 {len(all_messages)} 条翻译字符串")
    
    # 生成.pot文件
    pot_content = generate_pot_content(all_messages)
    pot_file = project_root / "locale" / "messages.pot"
    pot_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(pot_file, 'w', encoding='utf-8') as f:
        f.write(pot_content)
    
    print(f"已生成pot文件: {pot_file}")


def generate_pot_content(messages: Set[str]) -> str:
    """生成POT文件内容"""
    content = """# RAG-Anything Translation Template
# Copyright (C) 2024 RAG-Anything Project
# This file is distributed under the same license as the RAG-Anything package.
#
msgid ""
msgstr ""
"Project-Id-Version: RAG-Anything 1.0\\n"
"POT-Creation-Date: 2024-01-01 00:00+0000\\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"
"Language-Team: LANGUAGE <LL@li.org>\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"

"""
    
    for message in sorted(messages):
        content += f'#: (multiple files)\n'
        content += f'msgid "{message}"\n'
        content += f'msgstr ""\n\n'
    
    return content


def fix_syntax_errors(project_root: Optional[str] = None) -> None:
    """修复国际化过程中产生的语法错误"""
    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)
    
    raganything_dir = project_root / 'raganything'
    
    if not raganything_dir.exists():
        print(f"错误: 找不到raganything目录: {raganything_dir}")
        return
    
    python_files = list(raganything_dir.rglob("*.py"))
    fixed_count = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 修复多余的括号模式
            pattern = r'_\((["\'].*?(?:%[^"\']*)?["\']\)\)'
            new_content = re.sub(pattern, r'_(\1)', content)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"✓ 修复了 {file_path} 中的语法错误")
                fixed_count += 1
                
        except Exception as e:
            print(f"✗ 处理 {file_path} 时出错: {e}")
    
    print(f"\n修复完成: 修复了 {fixed_count} 个文件")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python i18n_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  extract     - 提取翻译字符串")
        print("  compile     - 编译.po文件为.mo文件")
        print("  fix-syntax  - 修复语法错误")
        print("  help        - 显示帮助信息")
        return
    
    command = sys.argv[1]
    
    if command == "extract":
        extract_translations()
    elif command == "compile":
        if len(sys.argv) > 2:
            I18NCompiler.compile_po_to_mo(sys.argv[2])
        else:
            I18NCompiler.compile_all_translations()
    elif command == "fix-syntax":
        fix_syntax_errors()
    elif command == "help":
        print("RAG-Anything 国际化工具集")
        print("\n用法: python i18n_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  extract [project_root]     - 提取翻译字符串")
        print("  compile [po_file]          - 编译.po文件，如果不指定文件则编译所有")
        print("  fix-syntax [project_root]  - 修复语法错误")
        print("  help                       - 显示帮助信息")
    else:
        print(f"未知命令: {command}")
        print("使用 'python i18n_tools.py help' 查看帮助")


if __name__ == "__main__":
    main()