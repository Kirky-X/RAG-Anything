#!/usr/bin/env python3
"""
翻译字符串提取工具。

该脚本扫描项目中的所有Python文件，提取需要翻译的字符串，
生成gettext的.pot模板文件。
"""

import ast
import os
import re
from pathlib import Path
from typing import Set, List, Dict
import sys


class TranslationExtractor(ast.NodeVisitor):
    """AST节点访问器，用于提取翻译字符串。"""
    
    def __init__(self):
        self.messages: Set[str] = set()
        self.logger_messages: Set[str] = set()
        self.current_file = ""
    
    def visit_Call(self, node: ast.Call) -> None:
        """访问函数调用节点。"""
        # 提取_()函数调用的参数
        if (isinstance(node.func, ast.Name) and 
            node.func.id == "_" and 
            node.args and 
            isinstance(node.args[0], ast.Constant) and
            isinstance(node.args[0].value, str)):
            self.messages.add(node.args[0].value)
        
        # 提取logger调用的消息
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id in ["logger", "_logger"] and
            node.func.attr in ["debug", "info", "warning", "error", "critical"] and
            node.args and
            isinstance(node.args[0], ast.Constant) and
            isinstance(node.args[0].value, str)):
            self.logger_messages.add(node.args[0].value)
        
        # 提取getLogger等调用
        if (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == "logging" and
            node.func.attr in ["getLogger"]):
            # 这些通常不需要翻译
            pass
        
        self.generic_visit(node)


def extract_from_file(file_path: Path) -> Dict[str, Set[str]]:
    """从单个Python文件中提取翻译字符串。
    
    Args:
        file_path: Python文件路径
        
    Returns:
        Dict包含messages和logger_messages
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST
        tree = ast.parse(content, filename=str(file_path))
        
        # 提取翻译字符串
        extractor = TranslationExtractor()
        extractor.current_file = str(file_path)
        extractor.visit(tree)
        
        # 额外提取一些常见的日志消息模式
        additional_patterns = [
            r'logger\.(debug|info|warning|error|critical)\s*\(\s*["\'](.*?)["\']',
            r'_logger\.(debug|info|warning|error|critical)\s*\(\s*["\'](.*?)["\']',
            r'print\s*\(\s*["\'](.*?)["\']',  # 也提取print语句
            r'f["\'](.*?)["\'].*?\.format\s*\(',  # f-string with format
        ]
        
        for pattern in additional_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    msg = match[-1]  # 取最后一个分组
                else:
                    msg = match
                if msg and len(msg.strip()) > 0:
                    extractor.logger_messages.add(msg.strip())
        
        return {
            "messages": extractor.messages,
            "logger_messages": extractor.logger_messages
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {"messages": set(), "logger_messages": set()}


def extract_from_project(project_root: Path) -> Dict[str, Set[str]]:
    """从整个项目中提取翻译字符串。
    
    Args:
        project_root: 项目根目录
        
    Returns:
        Dict包含所有提取的messages和logger_messages
    """
    all_messages = set()
    all_logger_messages = set()
    
    # 只处理核心项目文件，避免提取示例和测试中的内容字符串
    include_dirs = ["raganything", "tools"]
    exclude_dirs = {".venv", "venv", "__pycache__", ".git", "build", "dist", "examples", "tests", "scripts", "debug"}
    
    for include_dir in include_dirs:
        dir_path = project_root / include_dir
        if not dir_path.exists():
            continue
            
        python_files = list(dir_path.rglob("*.py"))
        
        for py_file in python_files:
            # 检查是否应该排除
            should_skip = False
            for exclude_dir in exclude_dirs:
                if exclude_dir in str(py_file):
                    should_skip = True
                    break
            
            if should_skip:
                continue
                
            print(f"Processing {py_file}...")
            result = extract_from_file(py_file)
            all_messages.update(result["messages"])
            all_logger_messages.update(result["logger_messages"])
    
    return {
        "messages": all_messages,
        "logger_messages": all_logger_messages
    }


def generate_pot_file(messages: Set[str], output_path: Path, project_name: str = "RAG-Anything") -> None:
    """生成.pot模板文件。
    
    Args:
        messages: 需要翻译的消息集合
        output_path: 输出文件路径
        project_name: 项目名称
    """
    from datetime import datetime
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    pot_content = f"""# Translation template for {project_name}.
# Copyright (C) {datetime.now().year} ORGANIZATION
# This file is distributed under the same license as the {project_name} project.
# FIRST AUTHOR <EMAIL@ADDRESS>, {datetime.now().year}.
#
#, fuzzy
msgid ""
msgstr ""
"Project-Id-Version: {project_name} 1.0\\n"
"POT-Creation-Date: {now}\\n"
"PO-Revision-Date: {now}\\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"
"Language-Team: LANGUAGE <LL@li.org>\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=utf-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Generated-By: custom-extractor 1.0\\n"

"""
    
    # 添加日志级别翻译
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS", "TRACE"]
    for level in log_levels:
        pot_content += f"#: Log level\\n"
        pot_content += f"msgid \"{level}\"\\n"
        pot_content += f"msgstr \"\"\\n\\n"
    
    # 添加提取的消息
    for message in sorted(messages):
        if message.strip():  # 跳过空消息
            # 处理多行消息
            escaped_msg = message.replace('\\', '\\\\').replace('"', '\\"')
            if '\\n' in escaped_msg:
                # 多行消息使用特殊的格式
                lines = escaped_msg.split('\\n')
                pot_content += f"msgid \"\"\\n"
                for line in lines:
                    pot_content += f'\"{line}\\n\"\\n'
                pot_content += f"msgstr \"\"\\n"
                for _ in lines:
                    pot_content += f'\"\\n\"\\n'
                pot_content += "\\n"
            else:
                pot_content += f"#: Extracted message\\n"
                pot_content += f"msgid \"{escaped_msg}\"\\n"
                pot_content += f"msgstr \"\"\\n\\n"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pot_content)
    
    print(f"Generated POT file: {output_path}")
    print(f"Total messages: {len(messages) + len(log_levels)}")


def main():
    """主函数。"""
    project_root = Path(__file__).parent.parent
    
    print("Extracting translation strings from project...")
    result = extract_from_project(project_root)
    
    print(f"\nExtracted {len(result['messages'])} explicit translation strings")
    print(f"Extracted {len(result['logger_messages'])} logger messages")
    
    # 合并所有消息
    all_messages = result['messages'] | result['logger_messages']
    
    # 生成POT文件
    pot_path = project_root / "locale" / "raganything.pot"
    generate_pot_file(all_messages, pot_path)
    
    print(f"\nPOT file generated: {pot_path}")
    print("You can now create .po files for specific languages:")
    print(f"  msginit -i {pot_path} -o locale/zh_CN/LC_MESSAGES/raganything.po -l zh_CN")
    print(f"  msginit -i {pot_path} -o locale/en_US/LC_MESSAGES/raganything.po -l en_US")


if __name__ == "__main__":
    main()