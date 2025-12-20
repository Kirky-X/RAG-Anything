#!/usr/bin/env python3
"""
异常消息国际化工具

这个脚本帮助识别和国际化代码中的异常消息。
"""

import ast
import os
import re
from pathlib import Path
from typing import List, Tuple


def extract_exception_messages(file_path: str) -> List[Tuple[int, str, str]]:
    """从Python文件中提取异常消息。
    
    Args:
        file_path: Python文件路径
        
    Returns:
        包含(行号, 异常类型, 消息)的列表
    """
    messages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 使用正则表达式匹配raise语句
        patterns = [
            (r'raise\s+(\w+)\(["\'](.*?)["\']\)', 'string'),
            (r'raise\s+(\w+)\(f["\'](.*?)["\']\)', 'f-string'),
            (r'raise\s+(\w+)\(["\'].*?["\'].*?\+\s*(.*?)\)', 'concat'),
            (r'raise\s+HTTPException\(.*detail=["\'](.*?)["\']\)', 'http_detail'),
            (r'raise\s+HTTPException\(.*detail=f["\'](.*?)["\']\)', 'http_f_detail'),
            (r'raise\s+HTTPException\(.*detail=str\((.*?)\)\)', 'http_str_detail'),
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern, pattern_type in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    if pattern_type.startswith('http'):
                        exc_type = 'HTTPException'
                        message = match.group(1)
                    else:
                        exc_type = match.group(1)
                        message = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    
                    # 只处理包含英文文本的消息
                    if message and any(c.isalpha() for c in message):
                        messages.append((i, exc_type, message.strip()))
                        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        
    return messages


def find_python_files(directory: str) -> List[str]:
    """查找目录下的所有Python文件。
    
    Args:
        directory: 要搜索的目录
        
    Returns:
        Python文件路径列表
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 跳过测试目录和工具目录
        dirs[:] = [d for d in dirs if d not in ['tests', 'tools', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                python_files.append(os.path.join(root, file))
                
    return python_files


def main():
    """主函数。"""
    project_root = Path(__file__).parent.parent
    raganything_dir = project_root / 'raganything'
    
    print("扫描异常消息...")
    python_files = find_python_files(str(raganything_dir))
    
    all_messages = {}
    
    for file_path in python_files:
        messages = extract_exception_messages(file_path)
        if messages:
            all_messages[file_path] = messages
    
    print(f"\n找到 {len(all_messages)} 个文件包含异常消息:")
    
    # 输出结果
    for file_path, messages in all_messages.items():
        relative_path = os.path.relpath(file_path, str(project_root))
        print(f"\n{relative_path}:")
        for line_num, exc_type, message in messages:
            print(f"  行 {line_num}: {exc_type} - \"{message}\"")
    
    # 生成翻译条目建议
    print("\n" + "="*60)
    print("建议的翻译条目:")
    print("="*60)
    
    unique_messages = set()
    for messages in all_messages.values():
        for _, _, message in messages:
            # 跳过包含变量或格式化的消息
            if '{' not in message and '%' not in message:
                unique_messages.add(message)
    
    for message in sorted(unique_messages):
        print(f"# {message}")
        print(f'msgid "{message}"')
        print('msgstr ""')
        print()


if __name__ == "__main__":
    main()