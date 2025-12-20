#!/usr/bin/env python3
"""
Compile .po files to .mo files using Python's msgfmt module.
"""

import os
import sys
from pathlib import Path
import polib


def compile_po_to_mo(po_file_path: str, mo_file_path: str = None) -> bool:
    """
    Compile a .po file to .mo format.
    
    Args:
        po_file_path: Path to the .po file
        mo_file_path: Path to the output .mo file (optional)
        
    Returns:
        True if compilation succeeded, False otherwise
    """
    po_path = Path(po_file_path)
    
    if not po_path.exists():
        print(f"Error: PO file not found: {po_file_path}")
        return False
    
    if mo_file_path is None:
        mo_file_path = po_path.with_suffix('.mo')
    
    try:
        # Load the PO file
        po = polib.pofile(po_file_path)
        
        # Save as MO file
        po.save_as_mofile(mo_file_path)
        
        print(f"Successfully compiled {po_file_path} -> {mo_file_path}")
        print(f"Compiled {len(po)} messages")
        return True
        
    except Exception as e:
        print(f"Error compiling {po_file_path}: {e}")
        return False


def compile_all_translations(locale_dir: str = "locale") -> None:
    """
    Compile all .po files in the locale directory to .mo files.
    
    Args:
        locale_dir: Path to the locale directory
    """
    locale_path = Path(locale_dir)
    
    if not locale_path.exists():
        print(f"Error: Locale directory not found: {locale_dir}")
        return
    
    compiled_count = 0
    
    # Find all .po files
    for po_file in locale_path.rglob("*.po"):
        # Generate corresponding .mo file path
        mo_file = po_file.with_suffix('.mo')
        
        print(f"Compiling {po_file}...")
        if compile_po_to_mo(str(po_file), str(mo_file)):
            compiled_count += 1
    
    print(f"\nCompilation complete: {compiled_count} files compiled")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        po_file = sys.argv[1]
        mo_file = sys.argv[2] if len(sys.argv) > 2 else None
        compile_po_to_mo(po_file, mo_file)
    else:
        compile_all_translations()