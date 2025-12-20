# RAG-Anything 国际化快速参考

## 快速开始

### 1. 在代码中使用翻译

```python
from raganything.i18n import _

# 基本用法
translated_text = _("Hello World")

# 带参数
logger.info(_("Processing {} items").format(count))

# 日志级别自动翻译
logger.info(_("Request started"))  # "信息: 请求开始" (中文)
```

### 2. 切换语言

```python
from raganything.i18n import set_language

# 切换到中文
set_language('zh_CN')

# 切换到英文
set_language('en_US')
```

### 3. 环境变量设置

```bash
# 设置默认语言
export RAG_LANGUAGE=zh_CN

# 临时设置
RAG_LANGUAGE=en_US python your_script.py
```

## 常用命令

### 提取字符串
```bash
python tools/extract_translations.py
```

### 编译翻译
```bash
python tools/compile_translations.py
```

### 运行测试
```bash
python -m pytest tests/test_i18n.py -v
```

## 翻译文件位置

- 模板文件：`locale/raganything.pot`
- 中文翻译：`locale/zh_CN/LC_MESSAGES/raganything.po/mo`
- 英文翻译：`locale/en_US/LC_MESSAGES/raganything.po/mo`

## 常见问题

### Q: 翻译不生效
A: 检查 .mo 文件是否存在，确认 RAG_LANGUAGE 设置正确

### Q: 日志级别不翻译
A: 确保日志级别在 .po 文件中有对应翻译

### Q: 切换语言后部分字符串仍显示旧语言
A: 使用模块级 `_()` 函数，避免缓存翻译函数引用

## 最佳实践

1. **所有用户可见字符串**都要用 `_()` 包装
2. **避免在循环中**频繁调用翻译函数
3. **使用格式化字符串**而不是字符串拼接
4. **定期更新**翻译文件
5. **测试所有语言**环境下的显示效果