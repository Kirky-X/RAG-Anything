# RAG-Anything 国际化维护指南

## 概述

本文档描述了 RAG-Anything 项目中基于 gettext 的国际化(i18n)系统的维护流程，包括翻译更新、文件管理和最佳实践。

## 目录结构

```
RAG-Anything/
├── locale/                          # 翻译文件根目录
│   ├── zh_CN/LC_MESSAGES/         # 中文翻译
│   │   ├── raganything.po         # 中文翻译源文件
│   │   └── raganything.mo         # 中文编译翻译文件
│   ├── en_US/LC_MESSAGES/         # 英文翻译
│   │   ├── raganything.po         # 英文翻译源文件
│   │   └── raganything.mo         # 英文编译翻译文件
│   └── raganything.pot             # 翻译模板文件
├── raganything/
│   ├── i18n.py                     # 国际化核心模块
│   ├── i18n_logger.py             # 日志国际化包装器
│   └── ...                        # 其他项目文件
├── tools/
│   ├── extract_translations.py      # 字符串提取工具
│   └── compile_translations.py     # 翻译编译工具
└── tests/
    └── test_i18n.py                # 国际化测试用例
```

## 翻译更新流程

### 1. 提取新的翻译字符串

当项目中添加了新的日志消息或用户可见的字符串时，需要更新翻译模板：

```bash
# 使用自定义工具提取字符串（推荐）
python tools/extract_translations.py

# 或使用传统 gettext 工具（需要安装 gettext）
xgettext -d raganything -o locale/raganything.pot --from-code=UTF-8 -L Python \
  raganything/*.py tools/*.py
```

### 2. 更新现有翻译文件

将新的字符串合并到现有的 .po 文件中：

```bash
# 中文翻译
msgmerge -U locale/zh_CN/LC_MESSAGES/raganything.po locale/raganything.pot

# 英文翻译  
msgmerge -U locale/en_US/LC_MESSAGES/raganything.po locale/raganything.pot
```

### 3. 翻译新字符串

编辑 .po 文件完成翻译：

```bash
# 使用文本编辑器或 poedit 等工具
vim locale/zh_CN/LC_MESSAGES/raganything.po
vim locale/en_US/LC_MESSAGES/raganything.po
```

.po 文件格式示例：
```po
#: raganything/i18n_logger.py:15
msgid "Processing request"
msgstr "处理请求"

#: raganything/i18n_logger.py:20
msgid "Request completed successfully"
msgstr "请求成功完成"
```

### 4. 编译翻译文件

将 .po 文件编译为 .mo 二进制文件：

```bash
# 使用自定义编译工具（推荐，无需额外依赖）
python tools/compile_translations.py

# 或使用传统 msgfmt 工具
msgfmt locale/zh_CN/LC_MESSAGES/raganything.po -o locale/zh_CN/LC_MESSAGES/raganything.mo
msgfmt locale/en_US/LC_MESSAGES/raganything.po -o locale/en_US/LC_MESSAGES/raganything.mo
```

### 5. 验证翻译

运行测试确保翻译正确工作：

```bash
# 运行国际化测试
python -m pytest tests/test_i18n.py -v

# 测试特定语言环境
RAG_LANGUAGE=zh_CN python -m pytest tests/test_i18n.py -v
RAG_LANGUAGE=en_US python -m pytest tests/test_i18n.py -v
```

## 代码集成指南

### 在代码中使用翻译

所有用户可见的字符串都应该使用 `_()` 函数包装：

```python
from raganything.i18n import _

# 日志消息
logger.info(_("Processing request"))
logger.debug(_("Request parameters: {}").format(params))

# 错误消息
raise ValueError(_("Invalid configuration"))

# 用户界面文本
print(_("Welcome to RAG-Anything"))
```

### 日志级别翻译

日志级别会自动翻译，无需额外处理：

```python
# DEBUG, INFO, WARNING, ERROR, CRITICAL 会自动翻译
logger.info("This message will be translated")  # 级别和内容都会翻译
```

### 动态语言切换

支持运行时切换语言：

```python
from raganything.i18n import set_language

# 切换到中文
set_language('zh_CN')

# 切换到英文
set_language('en_US')
```

## 故障排除

### 常见问题

#### 1. 翻译不生效

**症状**：字符串显示为原始英文

**可能原因和解决方案**：
- 检查 .mo 文件是否存在且最新
- 确认 RAG_LANGUAGE 环境变量设置正确
- 验证 set_language() 被正确调用
- 检查 `_()` 函数是否正确导入

#### 2. 编码错误

**症状**：中文字符显示为乱码

**解决方案**：
- 确保 .po 文件使用 UTF-8 编码
- 使用自定义编译工具（自动处理编码）
- 检查系统 locale 设置

#### 3. 日志级别不翻译

**症状**：日志级别显示为英文

**解决方案**：
- 确认日志级别在 .po 文件中有对应翻译
- 检查 i18n_logger.py 是否正确配置
- 验证翻译文件已编译

#### 4. 函数引用缓存问题

**症状**：切换语言后部分字符串仍显示旧语言

**解决方案**：
- 使用 Translator 类管理翻译状态
- 避免缓存翻译函数引用
- 确保使用模块级 `_()` 函数

### 调试工具

项目中提供了调试工具帮助诊断问题：

```bash
# 检查翻译文件内容
python -c "
import polib
po = polib.pofile('locale/zh_CN/LC_MESSAGES/raganything.po')
for entry in po:
    print(f'{entry.msgid} -> {entry.msgid}')
"

# 验证函数绑定
python debug_translation_dict.py
python debug_function_binding.py
```

## 最佳实践

### 1. 字符串提取

- 定期运行字符串提取工具
- 在代码审查时检查新字符串是否已包装
- 使用一致的字符串格式

### 2. 翻译质量

- 保持翻译的一致性
- 考虑上下文和使用场景
- 避免过长的翻译影响界面布局

### 3. 性能考虑

- 翻译函数调用有轻微性能开销
- 避免在性能关键路径中频繁调用
- 考虑缓存翻译结果（如果字符串不经常变化）

### 4. 维护流程

- 建立定期翻译更新计划
- 使用版本控制管理翻译文件
- 记录翻译决策和约定

## 扩展支持

### 添加新语言

1. 创建新的语言目录：
```bash
mkdir -p locale/ja_JP/LC_MESSAGES  # 示例：日语
```

2. 复制并修改翻译模板：
```bash
cp locale/raganything.pot locale/ja_JP/LC_MESSAGES/raganything.po
```

3. 完成翻译并编译：
```bash
# 编辑 .po 文件
vim locale/ja_JP/LC_MESSAGES/raganything.po

# 编译
python tools/compile_translations.py
```

4. 更新语言支持列表：
在 `i18n.py` 中更新 `SUPPORTED_LANGUAGES` 常量。

## 相关文档

- [gettext 文档](https://www.gnu.org/software/gettext/manual/)
- [Python gettext 模块](https://docs.python.org/3/library/gettext.html)
- [polib 文档](https://polib.readthedocs.io/)
- [Loguru 文档](https://loguru.readthedocs.io/)

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue 到项目仓库
- 联系维护团队