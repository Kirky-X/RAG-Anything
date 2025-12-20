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