#!/usr/bin/env python3
"""
Fix syntax errors caused by extra parentheses in internationalized exception messages.
"""

import re
from pathlib import Path

def fix_syntax_errors_in_file(file_path: Path) -> bool:
    """Fix syntax errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match: _("...")))
        # Replace with: _("...")
        pattern = r'_\(("[^"]*"|\'[^\']*\')\)\)'
        replacement = r'_(\1)'
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✓ Fixed syntax errors in {file_path}")
            return True
        else:
            print(f"- No syntax errors found in {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Fix syntax errors in all Python files."""
    raganything_dir = Path("/home/project/RAG-Anything/raganything")
    
    # Find all Python files
    python_files = list(raganything_dir.rglob("*.py"))
    
    fixed_count = 0
    for file_path in python_files:
        if fix_syntax_errors_in_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed syntax errors in {fixed_count} files")

if __name__ == "__main__":
    main()