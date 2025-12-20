# RAG-Anything 开发工具集

这个目录包含了RAG-Anything项目开发过程中使用的各种工具脚本。

## 工具列表

### 1. i18n_tools.py - 国际化工具集

整合了所有国际化相关的功能，包括翻译提取、编译和语法修复。

#### 使用方法

```bash
# 提取翻译字符串
python scripts/i18n_tools.py extract

# 编译.po文件为.mo文件
python scripts/i18n_tools.py compile

# 修复语法错误
python scripts/i18n_tools.py fix-syntax

# 显示帮助信息
python scripts/i18n_tools.py help
```

#### 功能说明

- **翻译提取**: 从Python文件中提取 `_()` 函数调用的字符串，生成POT文件
- **编译翻译**: 将.po文件编译为.mo文件，供程序运行时加载
- **语法修复**: 修复翻译字符串中的语法错误，如多余的括号等

### 2. dev_tools.py - 开发辅助工具集

提供项目开发过程中的各种辅助功能。

#### 使用方法

```bash
# 检查项目依赖
python scripts/dev_tools.py check-deps

# 检查项目结构
python scripts/dev_tools.py check-structure

# 运行代码质量检查（black, flake8, mypy）
python scripts/dev_tools.py check-quality

# 生成requirements.txt
python scripts/dev_tools.py generate-reqs

# 运行测试
python scripts/dev_tools.py test [all|unit|integration]

# 清理项目（删除缓存文件等）
python scripts/dev_tools.py clean

# 显示帮助信息
python scripts/dev_tools.py help
```

#### 功能说明

- **依赖检查**: 检查项目所需的Python包是否已安装
- **结构检查**: 验证项目目录结构是否符合规范
- **代码质量**: 运行black格式化、flake8代码风格检查和mypy类型检查
- **依赖生成**: 基于当前环境生成requirements.txt文件
- **测试运行**: 运行项目的单元测试和集成测试
- **项目清理**: 清理__pycache__、缓存文件等临时文件

### 3. media_tools.py - 媒体处理工具集

提供音频、视频和图像处理功能，整合了之前的独立脚本。

#### 使用方法

```bash
# 从视频中提取音频
python scripts/media_tools.py extract-audio <视频文件> [输出文件] [格式] [质量]

# 从视频中提取帧
python scripts/media_tools.py extract-frames <视频文件> <输出目录> [数量] [格式]

# 基准测试音频解析性能
python scripts/media_tools.py benchmark-audio [音频文件] [迭代次数] [时长]

# 验证VLM（视觉语言模型）解析功能
python scripts/media_tools.py verify-vlm <视频文件> [帧数]

# 显示帮助信息
python scripts/media_tools.py help
```

#### 功能说明

- **音频提取**: 从视频中提取音频，支持WAV、MP3、M4A格式
- **帧提取**: 从视频中提取随机帧，支持JPG、PNG格式
- **性能基准**: 测试音频解析性能，提供详细统计信息
- **VLM验证**: 验证视觉语言模型对视频帧的解析能力

### 4. config_tools.py - 配置管理工具集

提供项目配置管理、依赖检查、环境设置等功能。

#### 使用方法

```bash
# 检查项目依赖（详细模式）
python scripts/config_tools.py check-deps detailed

# 设置TikToken缓存
python scripts/config_tools.py setup-tiktoken [缓存目录]

# 合并配置文件
python scripts/config_tools.py merge-config [基础文件] [覆盖文件] [输出文件]

# 检查项目设置
python scripts/config_tools.py check-setup

# 生成环境设置脚本
python scripts/config_tools.py generate-setup [输出文件]

# 显示帮助信息
python scripts/config_tools.py help
```

#### 功能说明

- **依赖检查**: 详细检查Python包和RAG-Anything模块的安装状态
- **TikToken缓存**: 设置和管理TikToken模型的缓存
- **配置合并**: 合并多个TOML配置文件
- **项目检查**: 全面检查项目设置和环境配置
- **环境脚本**: 自动生成环境设置脚本，简化项目初始化

## 安装依赖

在使用这些工具之前，请确保安装了所需的依赖：

```bash
pip install polib requests fastapi uvicorn pydantic sqlalchemy alembic redis celery pytest black flake8 mypy
```

## 项目结构

```
scripts/
├── i18n_tools.py      # 国际化工具集
├── dev_tools.py       # 开发辅助工具集
└── README.md         # 本文件
```

## 注意事项

1. 所有工具脚本都使用Python 3.8+语法
2. 工具脚本会检查项目根目录的默认位置，也可以通过参数指定
3. 建议在虚拟环境中运行这些工具
4. 使用前请确保项目配置文件（pyproject.toml等）存在且配置正确