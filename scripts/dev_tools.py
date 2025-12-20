#!/usr/bin/env python3
"""
RAG-Anything 开发工具集

提供项目开发过程中常用的工具函数，包括代码检查、依赖管理等。
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional


class DevTools:
    """开发工具集"""
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """检查项目依赖是否安装"""
        required_packages = [
            'polib', 'requests', 'fastapi', 'uvicorn', 'pydantic',
            'sqlalchemy', 'alembic', 'redis', 'celery', 'pytest',
            'black', 'flake8', 'mypy'
        ]
        
        results = {}
        for package in required_packages:
            try:
                __import__(package)
                results[package] = True
            except ImportError:
                results[package] = False
        
        return results
    
    @staticmethod
    def run_code_quality_checks(project_root: Optional[str] = None) -> bool:
        """运行代码质量检查"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
        
        print("运行代码质量检查...")
        
        # 检查black格式化
        try:
            result = subprocess.run(
                ['black', '--check', '--diff', str(project_root / 'raganything')],
                capture_output=True, text=True, cwd=str(project_root)
            )
            if result.returncode != 0:
                print("❌ Black格式化检查失败")
                print(result.stdout)
                return False
            print("✅ Black格式化检查通过")
        except FileNotFoundError:
            print("⚠️  Black未安装，跳过格式化检查")
        
        # 检查flake8
        try:
            result = subprocess.run(
                ['flake8', str(project_root / 'raganything')],
                capture_output=True, text=True, cwd=str(project_root)
            )
            if result.returncode != 0:
                print("❌ Flake8检查失败")
                print(result.stdout)
                return False
            print("✅ Flake8检查通过")
        except FileNotFoundError:
            print("⚠️  Flake8未安装，跳过代码风格检查")
        
        # 检查mypy类型注解
        try:
            result = subprocess.run(
                ['mypy', str(project_root / 'raganything')],
                capture_output=True, text=True, cwd=str(project_root)
            )
            if result.returncode != 0:
                print("❌ MyPy类型检查失败")
                print(result.stdout)
                return False
            print("✅ MyPy类型检查通过")
        except FileNotFoundError:
            print("⚠️  MyPy未安装，跳过类型检查")
        
        return True
    
    @staticmethod
    def generate_requirements(project_root: Optional[str] = None) -> None:
        """生成requirements.txt文件"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
        
        try:
            # 使用pip freeze生成当前环境的依赖
            result = subprocess.run(
                ['pip', 'freeze'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                requirements_file = project_root / 'requirements.txt'
                with open(requirements_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                print(f"✅ 已生成requirements.txt: {requirements_file}")
            else:
                print("❌ 生成requirements.txt失败")
                
        except FileNotFoundError:
            print("⚠️  pip未找到，无法生成requirements.txt")
    
    @staticmethod
    def check_project_structure(project_root: Optional[str] = None) -> bool:
        """检查项目结构是否符合规范"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
        
        required_files = [
            'pyproject.toml',
            'requirements.txt',
            'README.md',
            'raganything/__init__.py',
            'tests/__init__.py'
        ]
        
        required_dirs = [
            'raganything',
            'tests',
            'docs',
            'locale'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file_path in required_files:
            if not (project_root / file_path).exists():
                missing_files.append(file_path)
        
        for dir_path in required_dirs:
            if not (project_root / dir_path).exists():
                missing_dirs.append(dir_path)
        
        if missing_files:
            print("❌ 缺少必需文件:")
            for file_path in missing_files:
                print(f"  - {file_path}")
        
        if missing_dirs:
            print("❌ 缺少必需目录:")
            for dir_path in missing_dirs:
                print(f"  - {dir_path}")
        
        if not missing_files and not missing_dirs:
            print("✅ 项目结构检查通过")
            return True
        
        return False
    
    @staticmethod
    def run_tests(project_root: Optional[str] = None, test_type: str = 'all') -> bool:
        """运行测试"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
        
        print(f"运行{test_type}测试...")
        
        try:
            if test_type == 'all':
                cmd = ['pytest', str(project_root / 'tests')]
            elif test_type == 'unit':
                cmd = ['pytest', str(project_root / 'tests/unit')]
            elif test_type == 'integration':
                cmd = ['pytest', str(project_root / 'tests/integration')]
            else:
                cmd = ['pytest', str(project_root / 'tests')]
            
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, cwd=str(project_root)
            )
            
            if result.returncode == 0:
                print("✅ 测试通过")
                return True
            else:
                print("❌ 测试失败")
                print(result.stdout)
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️  pytest未安装，跳过测试")
            return True
    
    @staticmethod
    def clean_project(project_root: Optional[str] = None) -> None:
        """清理项目中的临时文件和缓存"""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        else:
            project_root = Path(project_root)
        
        patterns_to_clean = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.pyd',
            '**/.pytest_cache',
            '**/.mypy_cache',
            '**/.coverage',
            '**/htmlcov',
            '**/*.egg-info',
            '**/.tox',
            '**/.cache',
            '**/dist',
            '**/build'
        ]
        
        cleaned_count = 0
        
        for pattern in patterns_to_clean:
            for file_path in project_root.rglob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"清理 {file_path} 时出错: {e}")
        
        print(f"✅ 清理完成，删除了 {cleaned_count} 个文件/目录")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python dev_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  check-deps          - 检查依赖")
        print("  check-structure     - 检查项目结构")
        print("  check-quality       - 运行代码质量检查")
        print("  generate-reqs       - 生成requirements.txt")
        print("  test [type]         - 运行测试 (all/unit/integration)")
        print("  clean               - 清理项目")
        print("  help                - 显示帮助信息")
        return
    
    command = sys.argv[1]
    
    if command == "check-deps":
        results = DevTools.check_dependencies()
        print("\n依赖检查结果:")
        for package, installed in results.items():
            status = "✅" if installed else "❌"
            print(f"  {status} {package}")
    
    elif command == "check-structure":
        DevTools.check_project_structure()
    
    elif command == "check-quality":
        DevTools.run_code_quality_checks()
    
    elif command == "generate-reqs":
        DevTools.generate_requirements()
    
    elif command == "test":
        test_type = sys.argv[2] if len(sys.argv) > 2 else 'all'
        DevTools.run_tests(test_type=test_type)
    
    elif command == "clean":
        DevTools.clean_project()
    
    elif command == "help":
        print("RAG-Anything 开发工具集")
        print("\n用法: python dev_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  check-deps          - 检查依赖")
        print("  check-structure     - 检查项目结构")
        print("  check-quality       - 运行代码质量检查")
        print("  generate-reqs       - 生成requirements.txt")
        print("  test [type]         - 运行测试 (all/unit/integration)")
        print("  clean               - 清理项目")
        print("  help                - 显示帮助信息")
    
    else:
        print(f"未知命令: {command}")
        print("使用 'python dev_tools.py help' 查看帮助")


if __name__ == "__main__":
    main()