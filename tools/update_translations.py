#!/usr/bin/env python3
"""
更新翻译文件工具

这个脚本更新.pot和.po文件，添加新的翻译条目。
"""

import os
import re
import subprocess
from pathlib import Path


def extract_translatable_strings():
    """从源代码中提取可翻译字符串。
    
    Returns:
        包含所有可翻译字符串的集合
    """
    project_root = Path(__file__).parent.parent
    raganything_dir = project_root / 'raganything'
    
    translatable_strings = set()
    
    # 扫描所有Python文件
    for root, dirs, files in os.walk(str(raganything_dir)):
        # 跳过测试目录和工具目录
        if any(skip_dir in root for skip_dir in ['tests', 'tools', '__pycache__', '.git']):
            continue
            
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 提取 _("message") 格式的字符串
                    pattern = r'_\(["\'](.*?)["\']\)'
                    matches = re.findall(pattern, content)
                    translatable_strings.update(matches)
                    
                    # 提取 _("message") % 格式的字符串
                    pattern = r'_\(["\'](.*?%[^"\']*)["\']\)'
                    matches = re.findall(pattern, content)
                    translatable_strings.update(matches)
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    
    return translatable_strings


def update_pot_file(strings_set):
    """更新.pot文件。
    
    Args:
        strings_set: 可翻译字符串集合
    """
    project_root = Path(__file__).parent.parent
    pot_file = project_root / 'locale' / 'raganything.pot'
    
    # 读取现有的.pot文件内容
    existing_content = ""
    if pot_file.exists():
        with open(pot_file, 'r', encoding='utf-8') as f:
            existing_content = f.read()
    
    # 提取现有的msgid
    existing_ids = set()
    if existing_content:
        pattern = r'msgid\s+"([^"]*)"'
        existing_ids = set(re.findall(pattern, existing_content))
    
    # 合并新的字符串
    all_strings = existing_ids.union(strings_set)
    
    # 生成新的.pot文件内容
    pot_content = f"""# RAG-Anything Translation Template
# Copyright (C) 2025 RAG-Anything Project
# This file is distributed under the same license as the RAG-Anything project.
#
msgid ""
msgstr ""
"Project-Id-Version: RAG-Anything 1.0\\n"
"POT-Creation-Date: 2025-01-01 00:00+0000\\n"
"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"
"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"
"Language-Team: LANGUAGE <LL@li.org>\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Generated-By: update_translations.py\\n"

"""
    
    # 添加所有字符串
    for string in sorted(all_strings):
        if string.strip():  # 跳过空字符串
            pot_content += f"#: Multiple files\n"
            pot_content += f'msgid "{string}"\n'
            pot_content += f'msgstr ""\n\n'
    
    # 写入.pot文件
    with open(pot_file, 'w', encoding='utf-8') as f:
        f.write(pot_content)
    
    print(f"✓ 已更新 {pot_file}")


def update_po_file(po_file, pot_file):
    """更新.po文件。
    
    Args:
        po_file: .po文件路径
        pot_file: .pot文件路径
    """
    if not po_file.exists():
        # 创建新的.po文件
        po_content = f"""# Chinese (Simplified) translations for RAG-Anything
# Copyright (C) 2025 RAG-Anything Project
# This file is distributed under the same license as the RAG-Anything project.
#
msgid ""
msgstr ""
"Project-Id-Version: RAG-Anything 1.0\\n"
"POT-Creation-Date: 2025-01-01 00:00+0000\\n"
"PO-Revision-Date: 2025-01-01 00:00+0000\\n"
"Last-Translator: RAG-Anything Team\\n"
"Language: zh_CN\\n"
"Language-Team: Chinese (Simplified)\\n"
"MIME-Version: 1.0\\n"
"Content-Type: text/plain; charset=UTF-8\\n"
"Content-Transfer-Encoding: 8bit\\n"
"Plural-Forms: nplurals=1; plural=0;\\n"

"""
        
        # 从.pot文件复制内容
        if pot_file.exists():
            with open(pot_file, 'r', encoding='utf-8') as f:
                pot_content = f.read()
            
            # 提取msgid并创建中文翻译
            pattern = r'msgid\s+"([^"]*)"'
            msgids = re.findall(pattern, pot_content)
            
            for msgid in msgids:
                if msgid.strip() and msgid != "":
                    # 简单的翻译映射（实际项目中应该使用专业的翻译）
                    translation = get_simple_translation(msgid)
                    po_content += f'msgid "{msgid}"\n'
                    po_content += f'msgstr "{translation}"\n\n'
        
        with open(po_file, 'w', encoding='utf-8') as f:
            f.write(po_content)
        
        print(f"✓ 已创建 {po_file}")
    else:
        # 使用msgmerge更新现有的.po文件
        try:
            subprocess.run(['msgmerge', '-U', str(po_file), str(pot_file)], 
                         check=True, capture_output=True, text=True)
            print(f"✓ 已更新 {po_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ 更新 {po_file} 失败: {e}")
        except FileNotFoundError:
            print(f"⚠  msgmerge 命令未找到，请手动更新 {po_file}")


def get_simple_translation(text):
    """简单的英文到中文翻译（仅用于示例）。
    
    Args:
        text: 英文文本
        
    Returns:
        中文翻译
    """
    # 这是一个简单的翻译映射，实际项目中应该使用专业的翻译服务
    translations = {
        "Error": "错误",
        "Error processing document": "处理文档时出错",
        "Info": "信息",
        "Debug": "调试",
        "Warning": "警告",
        "Critical": "严重",
        "Exception": "异常",
        "File not found": "文件未找到",
        "Folder not found": "文件夹未找到",
        "Invalid": "无效",
        "not found": "未找到",
        "does not exist": "不存在",
        "Unsupported": "不支持",
        "Failed": "失败",
        "timed out": "超时",
        "unavailable": "不可用",
        "Missing": "缺少",
        "required": "必需",
        "initialized": "初始化",
        "not initialized": "未初始化",
        "must be": "必须是",
        "processing": "处理",
        "conversion": "转换",
        "dependencies": "依赖库",
        "installed": "已安装",
        "response": "响应",
        "fields": "字段",
        "content": "内容",
        "extracted": "提取",
        "subclasses": "子类",
        "implement": "实现",
        "method": "方法",
        "unknown": "未知",
        "configuration": "配置",
        "model": "模型",
        "request": "请求",
        "provider": "提供程序",
        "storage": "存储",
        "backend": "后端",
        "source": "源",
        "document": "文档",
        "metadata": "元数据",
        "encode": "编码",
        "image": "图片",
        "base64": "base64",
        "audio": "音频",
        "video": "视频",
        "PDF": "PDF",
        "HTML": "HTML",
        "Text": "文本",
        "Office": "Office",
        "API": "API",
        "key": "密钥",
        "context": "上下文",
        "window": "窗口",
        "tokens": "令牌",
        "timeout": "超时",
        "retries": "重试",
        "logging": "日志",
        "max_bytes": "最大字节数",
        "backup_count": "备份数量",
        "concurrent": "并发",
        "files": "文件",
        "parse": "解析",
        "parser": "解析器",
        "method": "方法",
        "mode": "模式",
        "path": "路径",
        "directory": "目录",
        "outside": "超出",
        "root": "根",
        "prefix": "前缀",
        "integration": "集成",
        "embeddings": "嵌入",
        "core": "核心",
        "LightRAG": "LightRAG",
        "RAGAnything": "RAGAnything",
        "OpenAI": "OpenAI",
        "Ollama": "Ollama",
        "LangChain": "LangChain",
        "get_i18n_logger": "get_i18n_logger",
        "RobustOllamaClient": "RobustOllamaClient",
        "pydub": "pydub",
        "funasr": "funasr",
    }
    
    # 简单的翻译逻辑
    for eng, chn in translations.items():
        if eng.lower() in text.lower():
            return text.replace(eng, chn)
    
    # 如果没有找到翻译，返回原文
    return text


def main():
    """主函数。"""
    print("提取可翻译字符串...")
    strings = extract_translatable_strings()
    print(f"找到 {len(strings)} 个可翻译字符串")
    
    print("\n更新.pot文件...")
    update_pot_file(strings)
    
    print("\n更新.po文件...")
    project_root = Path(__file__).parent.parent
    pot_file = project_root / 'locale' / 'raganything.pot'
    po_file = project_root / 'locale' / 'zh_CN' / 'LC_MESSAGES' / 'raganything.po'
    
    update_po_file(po_file, pot_file)
    
    print("\n编译.mo文件...")
    try:
        subprocess.run(['msgfmt', str(po_file), '-o', 
                       str(po_file.with_suffix('.mo'))], 
                      check=True, capture_output=True, text=True)
        print(f"✓ 已编译 {po_file.with_suffix('.mo')}")
    except subprocess.CalledProcessError as e:
        print(f"✗ 编译失败: {e}")
    except FileNotFoundError:
        print(f"⚠  msgfmt 命令未找到，请手动编译")
    
    print("\n完成!")


if __name__ == "__main__":
    main()