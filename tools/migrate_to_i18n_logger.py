#!/usr/bin/env python3
"""
批量迁移脚本：将所有模块从旧logger迁移到国际化logger
"""

import os
import re
from pathlib import Path

def migrate_file(file_path: Path) -> bool:
    """迁移单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 替换导入语句
        content = re.sub(
            r'^from raganything\.logger import logger$',
            'from raganything.i18n_logger import get_i18n_logger',
            content,
            flags=re.MULTILINE
        )
        
        content = re.sub(
            r'^from raganything\.logger import logger as (\w+)$',
            r'from raganything.i18n_logger import get_i18n_logger\ndef \1(): return get_i18n_logger()',
            content,
            flags=re.MULTILINE
        )
        
        # 2. 替换logger变量定义
        content = re.sub(
            r'^logger = get_i18n_logger\(\)$',
            'logger = get_i18n_logger()',
            content,
            flags=re.MULTILINE
        )
        
        # 3. 替换logger = logger (这种情况在patches中)
        content = re.sub(
            r'^logger = logger$',
            'def logger(): return get_i18n_logger()',
            content,
            flags=re.MULTILINE
        )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
        
    except Exception as e:
        print(f"迁移 {file_path} 失败: {e}")
        return False

def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    raganything_dir = project_root / "raganything"
    
    # 需要迁移的文件列表
    target_files = [
        "parser/mineru_parser.py",
        "parser/base_parser.py", 
        "parser/video_parser.py",
        "parser/vlm_parser.py",
        "parser/docling_parser.py",
        "parser/audio_parser.py",
        "health/notifiers.py",
        "health/monitor.py",
        "modalprocessors/generic.py",
        "modalprocessors/equation.py",
        "modalprocessors/table.py",
        "modalprocessors/base.py",
        "modalprocessors/image.py",
        "models/manager.py",
        "models/device.py",
        "storage/core/factory.py",
        "storage/manager/storage_manager.py",
        "storage/backends/minio_backend.py",
        "storage/backends/local_backend.py",
        "llm/embedding.py",
        "llm/ollama_client.py",
        "llm/llm.py",
        "llm/agent.py",
        "raganything.py",
        "batch_parser.py",
        "server_config.py",
        "processor.py",
        "batch.py",
        "enhanced_markdown.py",
        "query.py",
        "config.py",
        "utils.py",
        "patches/lightrag_patch.py",
    ]
    
    migrated_count = 0
    for file_path in target_files:
        full_path = raganything_dir / file_path
        if full_path.exists():
            if migrate_file(full_path):
                print(f"✅ 已迁移: {file_path}")
                migrated_count += 1
            else:
                print(f"ℹ️  无需迁移: {file_path}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print(f"\n迁移完成: {migrated_count} 个文件已更新")

if __name__ == "__main__":
    main()