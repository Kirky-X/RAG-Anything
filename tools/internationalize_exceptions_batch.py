#!/usr/bin/env python3
"""
异常消息国际化批量处理工具

这个脚本自动将代码中的异常消息替换为国际化版本。
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def get_exception_translations() -> Dict[str, str]:
    """获取异常消息的翻译映射。
    
    Returns:
        英文到中文的翻译映射
    """
    return {
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
        "File not found": "文件未找到",
        "Folder not found": "文件夹未找到",
        "Document file does not exist": "文档文件不存在",
        "PDF file does not exist": "PDF文件不存在",
        "Image file does not exist": "图片文件不存在",
        "HTML file does not exist": "HTML文件不存在",
        "Text file does not exist": "文本文件不存在",
        "Office document does not exist": "Office文档不存在",
        "Unsupported text format": "不支持的文本格式",
        "Unsupported office format": "不支持的Office格式",
        "Unsupported HTML format": "不支持的HTML格式",
        "Unknown conversion method": "未知的转换方法",
        "Audio dependencies are not installed": "音频依赖库未安装",
        "Failed to convert audio file": "音频文件转换失败",
        "Failed to process image": "图片处理失败",
        "Could not open video file": "无法打开视频文件",
        "Request timed out": "请求超时",
        "Vision model configuration failed": "视觉模型配置失败",
        "LangChain OpenAI embeddings unavailable": "LangChain OpenAI嵌入不可用",
        "LangChain Ollama embeddings unavailable": "LangChain Ollama嵌入不可用",
        "LangChain core unavailable": "LangChain核心不可用",
        "OpenAI integration unavailable": "OpenAI集成不可用",
        "Ollama integration unavailable": "Ollama集成不可用",
        "Unsupported embedding provider": "不支持的嵌入提供程序",
        "Unsupported provider": "不支持的提供程序",
        "Unsupported storage backend": "不支持的存储后端",
        "Source file not found": "源文件未找到",
        "Source document not found": "源文档未找到",
        "Metadata not found for": "未找到元数据",
        "Failed to encode image to base64": "图片编码为base64失败",
    }


def internationalize_file(file_path: str) -> bool:
    """国际化单个文件中的异常消息。
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否进行了修改
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        translations = get_exception_translations()
        
        # 确保导入国际化函数
        if 'from raganything.i18n import _' not in content:
            # 找到第一个导入语句后添加
            import_pattern = r'^(from\s+\S+\s+import\s+.*|import\s+.*)$'
            matches = list(re.finditer(import_pattern, content, re.MULTILINE))
            if matches:
                last_import = matches[-1]
                insert_pos = last_import.end()
                content = content[:insert_pos] + '\nfrom raganything.i18n import _' + content[insert_pos:]
        
        # 替换简单的异常消息
        for eng_msg, chn_msg in translations.items():
            # 处理 raise Exception("message") 格式
            pattern1 = rf'raise\s+(\w+)\([f\'"]{re.escape(eng_msg)}[\'"\)]'
            replacement1 = r'raise \1(_("' + eng_msg + '"))'
            content = re.sub(pattern1, replacement1, content)
            
            # 处理 HTTPException detail="message" 格式
            pattern2 = rf'detail=[f\'"]{re.escape(eng_msg)}[\'"\)]'
            replacement2 = 'detail=_("' + eng_msg + '")'
            content = re.sub(pattern2, replacement2, content)
        
        # 特殊处理包含变量的消息
        variable_patterns = [
            (r'raise\s+FileNotFoundError\([f\'"]File not found: \{([^}]+)\}[\'"\)]', 
             r'raise FileNotFoundError(_("File not found: %s") % \1)'),
            (r'raise\s+FileNotFoundError\([f\'"]Folder not found: \{([^}]+)\}[\'"\)]', 
             r'raise FileNotFoundError(_("Folder not found: %s") % \1)'),
            (r'detail=[f\'"]File not found: \{([^}]+)\}[\'"\)]', 
             r'detail=_("File not found: %s") % \1'),
            (r'detail=[f\'"]Folder not found: \{([^}]+)\}[\'"\)]', 
             r'detail=_("Folder not found: %s") % \1'),
        ]
        
        for pattern, replacement in variable_patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def main():
    """主函数。"""
    project_root = Path(__file__).parent.parent
    raganything_dir = project_root / 'raganything'
    
    print("开始国际化异常消息...")
    
    # 获取需要处理的文件列表
    python_files = []
    for root, dirs, files in os.walk(str(raganything_dir)):
        # 跳过测试目录和工具目录
        dirs[:] = [d for d in dirs if d not in ['tests', '__pycache__', '.git']]
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_'):
                file_path = os.path.join(root, file)
                # 跳过已经国际化过的文件
                if 'i18n.py' in file_path or 'i18n_' in file_path:
                    continue
                python_files.append(file_path)
    
    modified_files = []
    
    for file_path in python_files:
        if internationalize_file(file_path):
            modified_files.append(file_path)
            print(f"✓ 已国际化: {os.path.relpath(file_path, str(project_root))}")
    
    print(f"\n完成! 共国际化了 {len(modified_files)} 个文件")
    
    if modified_files:
        print("\n更新翻译文件...")
        # 这里可以调用更新翻译文件的脚本
        print("请手动运行以下命令更新翻译文件:")
        print("python tools/update_translations.py")


if __name__ == "__main__":
    main()