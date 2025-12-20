#!/usr/bin/env python3
"""
恢复中文翻译的脚本

这个脚本用于在翻译更新后恢复关键的中文翻译，确保测试能够通过。
"""

import polib
from pathlib import Path

def restore_chinese_translations():
    """恢复中文翻译。"""
    po_file = Path("locale/zh_CN/LC_MESSAGES/raganything.po")
    
    if not po_file.exists():
        print(f"错误: 找不到文件 {po_file}")
        return False
    
    # 关键翻译映射
    key_translations = {
        # 日志级别
        "DEBUG": "调试",
        "INFO": "信息", 
        "WARNING": "警告",
        "ERROR": "错误",
        "CRITICAL": "严重",
        "SUCCESS": "成功",
        "TRACE": "跟踪",
        
        # 常用日志消息
        "RAGAnything CLI logging initialized": "RAGAnything CLI日志已初始化",
        "RAGAnything initialized successfully": "RAGAnything初始化成功",
        "Processing document": "正在处理文档",
        "Document processed successfully": "文档处理成功",
        "Query completed": "查询完成",
        "Error processing document": "处理文档时出错",
        "Initializing RAGAnything": "正在初始化RAGAnything",
        "Loading configuration": "正在加载配置",
        "Configuration loaded": "配置已加载",
        "Starting server": "正在启动服务器",
        "Server started": "服务器已启动",
        "Shutting down": "正在关闭",
        "Shutdown complete": "关闭完成",
        
        # 通用消息
        "Success": "成功",
        "Failed": "失败",
        "Error": "错误",
        "Warning": "警告",
        "Info": "信息",
        "Debug": "调试",
        "Complete": "完成",
        "Processing": "处理中",
        "Initializing": "初始化中",
        "Loading": "加载中",
        "Saving": "保存中",
        "Validating": "验证中",
        "Completed": "已完成",
        "Started": "已开始",
        "Finished": "已完成",
        
        # 文件和路径
        "File": "文件",
        "Directory": "目录",
        "Path": "路径",
        "Document": "文档",
        "Configuration": "配置",
        "Settings": "设置",
        
        # 操作
        "Create": "创建",
        "Update": "更新",
        "Delete": "删除",
        "Insert": "插入",
        "Query": "查询",
        "Search": "搜索",
        "Process": "处理",
        "Analyze": "分析",
        "Convert": "转换",
        "Export": "导出",
        "Import": "导入",
        
        # 状态
        "Ready": "就绪",
        "Running": "运行中",
        "Stopped": "已停止",
        "Paused": "已暂停",
        "Active": "活动",
        "Inactive": "非活动",
        "Enabled": "已启用",
        "Disabled": "已禁用",
        
        # 错误类型
        "Invalid": "无效",
        "Missing": "缺失",
        "Not found": "未找到",
        "Access denied": "访问被拒绝",
        "Timeout": "超时",
        "Connection failed": "连接失败",
        "Parse error": "解析错误",
        "Validation error": "验证错误",
        
        # 确认消息
        "Are you sure?": "您确定吗？",
        "Confirm": "确认",
        "Cancel": "取消",
        "OK": "确定",
        "Yes": "是",
        "No": "否",
    }
    
    try:
        # 加载现有的PO文件
        po = polib.pofile(str(po_file))
        
        translated_count = 0
        
        # 添加或更新关键翻译
        for msgid, translation in key_translations.items():
            # 查找是否存在该消息
            entry = po.find(msgid)
            if entry:
                # 更新现有翻译
                if not entry.msgstr:  # 只有没有翻译时才更新
                    entry.msgstr = translation
                    translated_count += 1
            else:
                # 添加新条目
                entry = polib.POEntry(
                    msgid=msgid,
                    msgstr=translation,
                    comment="关键翻译 - 自动恢复"
                )
                po.append(entry)
                translated_count += 1
        
        # 保存文件
        po.save(str(po_file))
        
        print(f"✅ 恢复了 {translated_count} 个关键翻译")
        
        # 显示统计信息
        total = len(po)
        translated = len([e for e in po if e.msgstr])
        print(f"总计: {total} 个字符串，已翻译: {translated} 个")
        
        return True
        
    except Exception as e:
        print(f"❌ 恢复翻译时出错: {e}")
        return False

if __name__ == "__main__":
    print("🌍 恢复中文翻译...")
    if restore_chinese_translations():
        print("✅ 翻译恢复完成")
    else:
        print("❌ 翻译恢复失败")
        exit(1)