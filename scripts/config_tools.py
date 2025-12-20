#!/usr/bin/env python3
"""
RAG-Anything 配置管理工具集

提供项目配置管理、依赖检查、环境设置等功能。
"""

import os
import sys
import subprocess
import tomllib
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

# 获取项目根目录（相对于脚本位置）
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))


try:
    import tomli_w
except ImportError:
    tomli_w = None
    print("警告: tomli_w 未安装，配置合并功能将不可用")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class ConfigManager:
    """配置管理器：处理项目配置相关任务"""
    
    def __init__(self):
        self.project_root = project_root
        self.config_file = self.project_root / "config.toml"
        self.env_example = self.project_root / "env.example"
    
    def check_dependencies(self, detailed: bool = False) -> Dict[str, Any]:
        """
        检查项目依赖
        
        Args:
            detailed: 是否显示详细信息
        
        Returns:
            依赖检查结果字典
        """
        print(f"Python 可执行文件: {sys.executable}")
        print(f"Python 版本: {sys.version}")
        print()
        
        # 核心依赖列表
        core_deps = {
            'pydub': '音频处理',
            'funasr': '语音识别',
            'tiktoken': 'Token化工具',
            'requests': 'HTTP请求',
            'fastapi': 'Web框架',
            'uvicorn': 'ASGI服务器',
            'pydantic': '数据验证',
            'sqlalchemy': 'ORM框架',
            'alembic': '数据库迁移',
            'redis': '缓存',
            'celery': '任务队列',
            'pytest': '测试框架',
            'black': '代码格式化',
            'flake8': '代码检查',
            'mypy': '类型检查',
            'numpy': '数值计算',
            'psutil': '系统监控',
            'soundfile': '音频文件处理'
        }
        
        # RAG-Anything特定依赖
        rag_deps = {
            'raganything.parser.audio_parser': '音频解析器',
            'raganything.parser.vlm_parser': '视觉语言模型解析器',
            'raganything.models.device': '设备管理器',
            'raganything.i18n_logger': '国际化日志器',
            'raganything.i18n': '国际化支持',
            'raganything.config': '配置管理'
        }
        
        results = {
            'core_dependencies': {},
            'rag_dependencies': {},
            'audio_deps_available': None
        }
        
        # 检查核心依赖
        print("=== 核心依赖检查 ===")
        for dep, description in core_deps.items():
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', '未知版本')
                file_path = getattr(module, '__file__', '未知路径')
                
                results['core_dependencies'][dep] = {
                    'installed': True,
                    'version': version,
                    'path': file_path,
                    'description': description
                }
                
                if detailed:
                    print(f"✅ {dep} ({description}) - 版本: {version}")
                    print(f"   路径: {file_path}")
                else:
                    print(f"✅ {dep} - {description}")
                    
            except ImportError:
                results['core_dependencies'][dep] = {
                    'installed': False,
                    'description': description
                }
                print(f"❌ {dep} - {description} - 未安装")
            except Exception as e:
                results['core_dependencies'][dep] = {
                    'installed': False,
                    'error': str(e),
                    'description': description
                }
                print(f"⚠️  {dep} - {description} - 错误: {e}")
        
        print()
        
        # 检查RAG-Anything特定依赖
        print("=== RAG-Anything 模块检查 ===")
        for dep, description in rag_deps.items():
            try:
                module = __import__(dep, fromlist=[''])
                file_path = getattr(module, '__file__', '未知路径')
                
                results['rag_dependencies'][dep] = {
                    'installed': True,
                    'path': file_path,
                    'description': description
                }
                
                if detailed:
                    print(f"✅ {dep} ({description})")
                    print(f"   路径: {file_path}")
                else:
                    print(f"✅ {dep} - {description}")
                    
            except ImportError:
                results['rag_dependencies'][dep] = {
                    'installed': False,
                    'description': description
                }
                print(f"❌ {dep} - {description} - 未安装")
            except Exception as e:
                results['rag_dependencies'][dep] = {
                    'installed': False,
                    'error': str(e),
                    'description': description
                }
                print(f"⚠️  {dep} - {description} - 错误: {e}")
        
        print()
        
        # 检查音频依赖可用性
        print("=== 音频依赖可用性 ===")
        try:
            from raganything.parser.audio_parser import AUDIO_DEPS_AVAILABLE
            results['audio_deps_available'] = AUDIO_DEPS_AVAILABLE
            print(f"音频依赖可用性: {AUDIO_DEPS_AVAILABLE}")
        except Exception as e:
            results['audio_deps_available'] = False
            print(f"检查音频依赖可用性时出错: {e}")
            traceback.print_exc()
        
        return results
    
    def setup_tiktoken_cache(self, cache_dir: Optional[str] = None) -> bool:
        """
        设置tiktoken缓存
        
        Args:
            cache_dir: 缓存目录路径，如果为None则使用默认路径
        
        Returns:
            成功返回True，失败返回False
        """
        if not TIKTOKEN_AVAILABLE:
            print("❌ tiktoken 未安装，无法设置缓存")
            return False
        
        # 确定缓存目录
        if cache_dir is None:
            try:
                from raganything.config import RAGAnythingConfig
                cfg = RAGAnythingConfig()
                cache_dir = cfg.tiktoken.cache_dir or "/tmp/tiktoken_cache"
            except Exception:
                cache_dir = "/tmp/tiktoken_cache"
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_path))
        
        print(f"正在下载和缓存tiktoken模型...")
        
        try:
            # 下载并缓存常用模型
            tiktoken.get_encoding("cl100k_base")
            print(f"✅ tiktoken模型已缓存到 '{cache_path}'")
            return True
        except Exception as e:
            print(f"❌ 缓存tiktoken模型失败: {e}")
            return False
    
    def merge_config_files(self, base_path: str = "env.example", 
                          override_path: str = "config.toml", 
                          output_path: str = "config.toml") -> bool:
        """
        合并配置文件
        
        Args:
            base_path: 基础配置文件路径
            override_path: 覆盖配置文件路径
            output_path: 输出文件路径
        
        Returns:
            成功返回True，失败返回False
        """
        if tomli_w is None:
            print("❌ tomli_w 未安装，无法合并配置")
            return False
        
        base_file = self.project_root / base_path
        override_file = self.project_root / override_path
        output_file = self.project_root / output_path
        
        if not base_file.exists():
            print(f"❌ 基础配置文件不存在: {base_file}")
            return False
        
        print(f"正在合并配置文件...")
        print(f"基础文件: {base_file}")
        print(f"覆盖文件: {override_file}")
        print(f"输出文件: {output_file}")
        
        try:
            # 加载基础配置
            with open(base_file, 'rb') as f:
                base_config = tomllib.load(f)
            
            # 如果覆盖文件存在，加载它
            if override_file.exists():
                with open(override_file, 'rb') as f:
                    override_config = tomllib.load(f)
            else:
                override_config = {}
            
            # 合并配置
            merged_config = self._merge_dicts(base_config, override_config)
            
            # 写入输出文件
            with open(output_file, 'wb') as f:
                tomli_w.dump(merged_config, f)
            
            print(f"✅ 配置文件合并成功: {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ 合并配置文件失败: {e}")
            return False
    
    def _merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并字典"""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    def check_project_setup(self) -> Dict[str, Any]:
        """
        检查项目设置
        
        Returns:
            项目设置检查结果
        """
        results = {
            'config_file_exists': False,
            'env_example_exists': False,
            'tiktoken_cache_setup': False,
            'missing_dependencies': [],
            'recommendations': []
        }
        
        print("=== 项目设置检查 ===")
        
        # 检查配置文件
        if self.config_file.exists():
            results['config_file_exists'] = True
            print(f"✅ 配置文件存在: {self.config_file}")
        else:
            print(f"❌ 配置文件不存在: {self.config_file}")
            results['recommendations'].append("创建 config.toml 配置文件")
        
        # 检查环境示例文件
        if self.env_example.exists():
            results['env_example_exists'] = True
            print(f"✅ 环境示例文件存在: {self.env_example}")
        else:
            print(f"❌ 环境示例文件不存在: {self.env_example}")
            results['recommendations'].append("创建 env.example 示例文件")
        
        # 检查tiktoken缓存设置
        cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
        if cache_dir and Path(cache_dir).exists():
            results['tiktoken_cache_setup'] = True
            print(f"✅ TikToken缓存已设置: {cache_dir}")
        else:
            print("⚠️  TikToken缓存未设置")
            results['recommendations'].append("设置TikToken缓存目录")
        
        # 检查依赖
        print()
        dep_results = self.check_dependencies(detailed=False)
        
        missing_core_deps = [name for name, info in dep_results['core_dependencies'].items() 
                           if not info['installed']]
        missing_rag_deps = [name for name, info in dep_results['rag_dependencies'].items() 
                           if not info['installed']]
        
        results['missing_dependencies'] = missing_core_deps + missing_rag_deps
        
        if missing_core_deps:
            print(f"❌ 缺少核心依赖: {', '.join(missing_core_deps)}")
            results['recommendations'].append(f"安装缺少的核心依赖: {', '.join(missing_core_deps[:3])}")
        
        if missing_rag_deps:
            print(f"❌ 缺少RAG-Anything模块: {', '.join(missing_rag_deps)}")
            results['recommendations'].append("确保RAG-Anything项目正确安装")
        
        # 总体建议
        if results['recommendations']:
            print("\n=== 建议操作 ===")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec}")
        else:
            print("✅ 项目设置检查通过")
        
        return results
    
    def generate_setup_script(self, output_path: str = "setup_env.py") -> bool:
        """
        生成环境设置脚本
        
        Args:
            output_path: 输出脚本路径
        
        Returns:
            成功返回True，失败返回False
        """
        setup_script = f'''#!/usr/bin/env python3
"""
RAG-Anything 环境设置脚本
自动生成于 {__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path("{self.project_root}")

# 需要安装的依赖包
MISSING_DEPS = {repr(self._get_missing_dependencies())}

def install_dependencies():
    """安装缺少的依赖"""
    if not MISSING_DEPS:
        print("✅ 所有依赖都已安装")
        return
    
    print(f"正在安装 {len(MISSING_DEPS)} 个缺少的依赖...")
    
    for dep in MISSING_DEPS:
        try:
            print(f"安装 {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError:
            print(f"❌ {dep} 安装失败")

def setup_tiktoken_cache():
    """设置TikToken缓存"""
    cache_dir = "/tmp/tiktoken_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ TikToken缓存目录: {{cache_dir}}")

def setup_environment():
    """设置环境变量"""
    # 设置Python路径
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    
    # 设置其他环境变量
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    
    print("✅ 环境变量设置完成")

def main():
    """主函数"""
    print("开始设置RAG-Anything环境...")
    
    install_dependencies()
    setup_tiktoken_cache()
    setup_environment()
    
    print("\\n🎉 环境设置完成！")
    print("现在您可以运行RAG-Anything项目了。")

if __name__ == "__main__":
    main()
'''
        
        try:
            output_file = self.project_root / output_path
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(setup_script)
            
            # 设置执行权限
            output_file.chmod(0o755)
            
            print(f"✅ 环境设置脚本已生成: {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ 生成设置脚本失败: {e}")
            return False
    
    def _get_missing_dependencies(self) -> List[str]:
        """获取缺少的依赖列表"""
        results = self.check_dependencies(detailed=False)
        missing = []
        
        for name, info in results['core_dependencies'].items():
            if not info['installed']:
                missing.append(name)
        
        return missing


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python config_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  check-deps [detailed]     - 检查依赖")
        print("  setup-tiktoken [缓存目录]  - 设置TikToken缓存")
        print("  merge-config [基础文件] [覆盖文件] [输出文件] - 合并配置")
        print("  check-setup               - 检查项目设置")
        print("  generate-setup [输出文件]  - 生成环境设置脚本")
        print("  help                      - 显示帮助信息")
        return
    
    manager = ConfigManager()
    command = sys.argv[1]
    
    if command == "check-deps":
        detailed = len(sys.argv) > 2 and sys.argv[2] == "detailed"
        manager.check_dependencies(detailed=detailed)
    
    elif command == "setup-tiktoken":
        cache_dir = sys.argv[2] if len(sys.argv) > 2 else None
        success = manager.setup_tiktoken_cache(cache_dir)
        if success:
            print("✅ TikToken缓存设置成功")
        else:
            print("❌ TikToken缓存设置失败")
    
    elif command == "merge-config":
        base_path = sys.argv[2] if len(sys.argv) > 2 else "env.example"
        override_path = sys.argv[3] if len(sys.argv) > 3 else "config.toml"
        output_path = sys.argv[4] if len(sys.argv) > 4 else "config.toml"
        
        success = manager.merge_config_files(base_path, override_path, output_path)
        if success:
            print("✅ 配置合并成功")
        else:
            print("❌ 配置合并失败")
    
    elif command == "check-setup":
        results = manager.check_project_setup()
        
        # 显示总体状态
        total_checks = len(results)
        passed_checks = sum(1 for v in results.values() if v is True)
        
        print(f"\n📊 项目设置检查总结:")
        print(f"  通过: {passed_checks}/{total_checks}")
        
        if results['missing_dependencies']:
            print(f"  缺少依赖: {len(results['missing_dependencies'])} 个")
        
        if results['recommendations']:
            print(f"  建议操作: {len(results['recommendations'])} 项")
    
    elif command == "generate-setup":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "setup_env.py"
        success = manager.generate_setup_script(output_path)
        
        if success:
            print(f"\n🎯 使用说明:")
            print(f"  1. 运行设置脚本: python {output_path}")
            print(f"  2. 或者执行: ./{output_path}")
            print(f"  3. 按照提示完成环境设置")
    
    elif command == "help":
        print("RAG-Anything 配置管理工具集")
        print("\n用法: python config_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  check-deps [detailed]     - 检查依赖")
        print("  setup-tiktoken [缓存目录]  - 设置TikToken缓存")
        print("  merge-config [基础文件] [覆盖文件] [输出文件] - 合并配置")
        print("  check-setup               - 检查项目设置")
        print("  generate-setup [输出文件]  - 生成环境设置脚本")
        print("\n示例:")
        print("  python config_tools.py check-deps detailed")
        print("  python config_tools.py setup-tiktoken /tmp/tiktoken_cache")
        print("  python config_tools.py merge-config")
        print("  python config_tools.py generate-setup setup_my_env.py")
    
    else:
        print(f"未知命令: {command}")
        print("使用 'python config_tools.py help' 查看帮助")


if __name__ == "__main__":
    main()