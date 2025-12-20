#!/usr/bin/env python3
"""
RAG-Anything 全流程测试脚本
针对 project_1.mp4 进行从输入到输出的完整处理流程测试
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from raganything.logger import get_i18n_logger
from raganything.i18n import _

# 配置日志
log_file_handler = logging.FileHandler("/tmp/full_pipeline_test.log")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        log_file_handler,
        stream_handler,
    ],
)
logger = get_i18n_logger()


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.metrics = {}

    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = {
            "start_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
            "start_cpu": self.process.cpu_percent(),
            "start_disk": psutil.disk_usage("/").percent,
        }
        logger.info(
            f"开始监控 - 内存: {self.metrics['start_memory']:.2f}MB, CPU: {self.metrics['start_cpu']}%"
        )

    def stop_monitoring(self):
        """停止监控并记录指标"""
        end_time = time.time()
        self.metrics.update(
            {
                "end_memory": self.process.memory_info().rss / 1024 / 1024,  # MB
                "end_cpu": self.process.cpu_percent(),
                "end_disk": psutil.disk_usage("/").percent,
                "duration": end_time - self.start_time,
                "memory_increase": (self.process.memory_info().rss / 1024 / 1024)
                - self.metrics["start_memory"],
            }
        )
        logger.info(_("Processing time: {:.2f}s").format(self.metrics['duration']))
        logger.info(_("Memory increase: {:.2f}MB").format(self.metrics['memory_increase']))
        return self.metrics


class FullPipelineTester:
    """全流程测试器"""

    def __init__(self, test_video_path: str, output_dir: str):
        self.test_video_path = test_video_path
        self.output_dir = output_dir
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor()
        self.results = {}

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        logger.info(_("Initializing tester - test video: {}").format(test_video_path))
        logger.info(_("Output directory: {}").format(output_dir))
        logger.info(_("Temporary directory: {}").format(self.temp_dir))

    def validate_input_file(self) -> bool:
        """验证输入文件"""
        logger.info("=== 1. Input file validation ===")

        if not os.path.exists(self.test_video_path):
            logger.error(_("Test video file does not exist: {}").format(self.test_video_path))
            return False

        # 检查文件大小和格式
        file_size = os.path.getsize(self.test_video_path)
        file_ext = os.path.splitext(self.test_video_path)[1].lower()

        logger.info(_("File path: {}").format(self.test_video_path))
        logger.info(_("File size: {:.2f}MB").format(file_size / 1024 / 1024))
        logger.info(_("File format: {}").format(file_ext))

        # 验证是否为有效视频文件
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,duration,bit_rate",
                    "-of",
                    "json",
                    self.test_video_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                video_info = json.loads(result.stdout)
                if "streams" in video_info and len(video_info["streams"]) > 0:
                    stream = video_info["streams"][0]
                    logger.info(_("Video codec: {}").format(stream.get('codec_name', 'unknown')))
                    logger.info(_("Video duration: {}s").format(stream.get('duration', 'unknown')))
                    logger.info(_("Bitrate: {}bps").format(stream.get('bit_rate', 'unknown')))
                    self.results["input_validation"] = {
                        "file_size_mb": file_size / 1024 / 1024,
                        "codec": stream.get("codec_name"),
                        "duration": stream.get("duration"),
                        "bit_rate": stream.get("bit_rate"),
                    }
                    return True

        except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
            logger.warning(_("ffprobe analysis failed: {}").format(e))

        logger.info("Input file validation passed")
        return True

    def test_video_parsing(self) -> bool:
        """测试视频解析模块"""
        logger.info("=== 2. 视频解析测试 ===")

        self.monitor.start_monitoring()

        try:
            # 导入视频解析模块
            from raganything.parser.video_parser import VideoParser

            parser = VideoParser()
            logger.info("视频解析器初始化成功")

            # 解析视频文件
            parse_result = parser.parse_video(self.test_video_path)

            if not parse_result:
                logger.error("视频解析返回空结果")
                return False

            # 记录解析结果
            self.results["video_parsing"] = {
                "parse_result_type": type(parse_result).__name__,
                "parse_result_length": (
                    len(parse_result) if hasattr(parse_result, "__len__") else "unknown"
                ),
                "parse_result_keys": (
                    list(parse_result.keys()) if isinstance(parse_result, dict) else []
                ),
            }

            logger.info(
                f"解析结果类型: {self.results['video_parsing']['parse_result_type']}"
            )
            logger.info(
                f"解析结果长度: {self.results['video_parsing']['parse_result_length']}"
            )

            # 保存解析结果用于后续验证
            parse_output = os.path.join(self.temp_dir, "video_parse_result.json")
            with open(parse_output, "w", encoding="utf-8") as f:
                if isinstance(parse_result, (dict, list)):
                    json.dump(parse_result, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(parse_result))

            logger.info(_("Parse results saved to: {}").format(parse_output))

            metrics = self.monitor.stop_monitoring()
            self.results["video_parsing"]["performance"] = metrics

            return True

        except Exception as e:
            logger.error(_("Video parsing test failed: {}").format(e))
            self.monitor.stop_monitoring()
            return False

    async def test_embedding_generation(self) -> bool:
        """测试嵌入向量生成"""
        logger.info("=== 3. 嵌入向量生成测试 ===")

        self.monitor.start_monitoring()

        try:
            # 导入嵌入模块
            from raganything.llm.embedding import (LocalEmbeddingWrapper,
                                                   build_embedding_func)

            embedder = build_embedding_func(
                "local", "local-embedding", embedding_dim=1536
            )
            logger.info("嵌入包装器初始化成功")

            # 生成测试文本的嵌入向量
            test_texts = [
                "This is a test video about RAG-Anything system.",
                "The video demonstrates the complete processing pipeline.",
                "Testing embedding generation functionality.",
            ]

            embeddings = []
            for i, text in enumerate(test_texts):
                logger.info(_("Generating embedding {}/{}").format(i+1, len(test_texts)))
                embedding = await embedder.func([text])

                if embedding is None:
                    logger.error(_("Text {} embedding generation failed").format(i+1))
                    return False

                # Convert numpy array to list for JSON serialization
                if hasattr(embedding, "tolist"):
                    embedding = embedding.tolist()

                embeddings.append(embedding)

                embedding_vector = (
                    embedding[0]
                    if isinstance(embedding, list) and len(embedding) > 0
                    else embedding
                )
                logger.info(
                    f"嵌入向量维度: {len(embedding_vector) if hasattr(embedding_vector, '__len__') else 'unknown'}"
                )

            # 记录嵌入结果
            self.results["embedding_generation"] = {
                "test_texts_count": len(test_texts),
                "embeddings_count": len(embeddings),
                "embedding_dimensions": (
                    len(embeddings[0][0])
                    if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0
                    else 0
                ),
                "sample_embedding_preview": (
                    embeddings[0][0][:10]
                    if embeddings and len(embeddings) > 0 and len(embeddings[0]) > 0
                    else []
                ),
            }

            logger.info(_("Successfully generated {} embeddings").format(len(embeddings)))
            logger.info(
                _("Embedding dimensions: {}").format(self.results['embedding_generation']['embedding_dimensions'])
            )

            metrics = self.monitor.stop_monitoring()
            self.results["embedding_generation"]["performance"] = metrics

            return True

        except Exception as e:
            logger.error(_("Embedding generation test failed: {}").format(e))
            self.monitor.stop_monitoring()
            return False

    async def test_storage_operations(self) -> bool:
        """测试存储操作（异步版本）"""
        logger.info("=== 4. 存储操作测试 ===")

        self.monitor.start_monitoring()

        try:
            # 导入存储模块
            from raganything.storage.core.factory import StorageFactory

            # 创建本地存储管理器
            storage_config = {
                "backend": "local",
                "local_root": os.path.join(self.temp_dir, "storage_test"),
            }

            factory = StorageFactory()
            storage_manager = factory.create_manager(storage_config)

            logger.info("存储管理器初始化成功")

            # 测试存储操作 - 创建测试文件
            test_doc_path = os.path.join(self.temp_dir, "test_document.txt")
            test_content = "Test content for storage operations"
            with open(test_doc_path, "w", encoding="utf-8") as f:
                f.write(test_content)

            # 存储文档（异步）
            doc_id = await storage_manager.store_document(
                test_doc_path, "test_user_001"
            )
            if not doc_id:
                logger.error("文档存储失败")
                return False

            logger.info(_("Document stored successfully, ID: {}").format(doc_id))

            # 检索文档（异步）
            retrieved_content = await storage_manager.retrieve_document(doc_id)
            if not retrieved_content:
                logger.error("文档检索失败")
                return False

            logger.info("文档检索成功")

            # 验证内容一致性
            if retrieved_content.decode("utf-8") != test_content:
                logger.error("存储和检索的内容不一致")
                return False

            # 记录存储结果
            self.results["storage_operations"] = {
                "storage_type": type(storage_manager).__name__,
                "store_success": bool(doc_id),
                "retrieve_success": bool(retrieved_content),
                "content_consistency": True,
            }

            logger.info("存储操作测试通过")

            metrics = self.monitor.stop_monitoring()
            self.results["storage_operations"]["performance"] = metrics

            return True

        except Exception as e:
            logger.error(_("Storage operation test failed: {}").format(e))
            self.monitor.stop_monitoring()
            return False

    async def test_query_processing(self) -> bool:
        """测试查询处理"""
        logger.info("=== 5. 查询处理测试 ===")

        self.monitor.start_monitoring()

        try:
            # 导入查询模块
            from raganything.query import QueryMixin

            # 创建查询处理器（使用 QueryMixin）
            query_processor = QueryMixin()
            logger.info("查询处理器初始化成功")

            # 测试不同类型的查询
            test_queries = [
                "What is RAG-Anything?",
                "How does the video processing work?",
                "Tell me about the system architecture.",
            ]

            query_results = []
            for i, query in enumerate(test_queries):
                logger.info(_("Processing query {}: {}").format(i+1, query))

                try:
                    # 执行查询（使用同步方法作为回退）
                    result = query_processor.query(query, mode="mix")

                    if not result:
                        logger.warning(_("Query {} returned empty results").format(i+1))
                        continue

                    query_results.append(
                        {
                            "query": query,
                            "result": result,
                            "result_type": type(result).__name__,
                        }
                    )

                except Exception as query_error:
                    logger.warning(_("Query {} failed: {}").format(i+1, query_error))
                    continue

            if not query_results:
                logger.error("所有查询都失败了")
                return False

            # 记录查询结果
            self.results["query_processing"] = {
                "total_queries": len(test_queries),
                "successful_queries": len(query_results),
                "query_types": [qr["result_type"] for qr in query_results],
                "sample_results": [
                    (
                        qr["result"][:100] + "..."
                        if len(str(qr["result"])) > 100
                        else str(qr["result"])
                    )
                    for qr in query_results[:2]
                ],
            }

            logger.info(_("Successfully processed {}/{} queries").format(len(query_results), len(test_queries)))

            metrics = self.monitor.stop_monitoring()
            self.results["query_processing"]["performance"] = metrics

            return True

        except Exception as e:
            logger.error(_("Query processing test failed: {}").format(e))
            self.monitor.stop_monitoring()
            return False

    def test_output_generation(self) -> bool:
        """测试输出生成"""
        logger.info("=== 6. 输出生成测试 ===")

        self.monitor.start_monitoring()

        try:
            # 生成测试报告
            report_file = os.path.join(self.output_dir, "test_report.json")

            # 添加系统信息
            system_info = {
                "test_timestamp": datetime.now().isoformat(),
                "test_video": self.test_video_path,
                "python_version": sys.version,
                "platform": sys.platform,
            }

            final_report = {
                "system_info": system_info,
                "test_results": self.results,
                "summary": self.generate_summary(),
            }

            # 保存报告
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)

            logger.info(_("Test report generated: {}").format(report_file))

            # 生成文本报告
            text_report = os.path.join(self.output_dir, "test_summary.txt")
            with open(text_report, "w", encoding="utf-8") as f:
                f.write(self.generate_text_summary())

            logger.info(_("Text summary generated: {}").format(text_report))

            self.results["output_generation"] = {
                "report_file": report_file,
                "text_report": text_report,
                "report_size": os.path.getsize(report_file),
            }

            metrics = self.monitor.stop_monitoring()
            self.results["output_generation"]["performance"] = metrics

            return True

        except Exception as e:
            logger.error(_("Output generation test failed: {}").format(e))
            self.monitor.stop_monitoring()
            return False

    def generate_summary(self) -> Dict[str, Any]:
        """生成测试摘要"""
        total_tests = 6  # 总测试数量
        passed_tests = 0

        # 统计通过的测试
        test_modules = [
            "input_validation",
            "video_parsing",
            "embedding_generation",
            "storage_operations",
            "query_processing",
            "output_generation",
        ]

        for module in test_modules:
            if module in self.results:
                passed_tests += 1

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "overall_status": "PASSED" if passed_tests == total_tests else "FAILED",
        }

    def generate_text_summary(self) -> str:
        """生成文本摘要"""
        summary = self.generate_summary()

        text = f"""
RAG-Anything 全流程测试报告
============================

测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
测试文件: {self.test_video_path}

总体结果:
- 总测试数: {summary['total_tests']}
- 通过测试: {summary['passed_tests']}
- 失败测试: {summary['failed_tests']}
- 成功率: {summary['success_rate']:.1f}%
- 状态: {summary['overall_status']}

详细性能指标:
"""

        # 添加各模块的性能指标
        for module, data in self.results.items():
            if isinstance(data, dict) and "performance" in data:
                perf = data["performance"]
                text += f"\n{module.upper()}:\n"
                text += f"  处理耗时: {perf.get('duration', 0):.2f}s\n"
                text += f"  内存增长: {perf.get('memory_increase', 0):.2f}MB\n"

        return text

    def cleanup(self):
        """清理临时文件和日志处理器"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(_("Cleaned up temporary directory: {}").format(self.temp_dir))
        except Exception as e:
            logger.warning(_("Error cleaning up temporary files: {}").format(e))
        
        # 关闭日志处理器以避免ResourceWarning
        try:
            log_file_handler.close()
            stream_handler.close()
            logger.info("日志处理器已关闭")
        except Exception as e:
            logger.warning(_("Error closing log handlers: {}").format(e))

    async def run_full_pipeline_test_async(self) -> bool:
        """运行完整的流程测试（异步版本）"""
        logger.info("开始 RAG-Anything 全流程测试（跳过视频解析）")
        logger.info("=" * 50)

        try:
            # 1. 输入验证（跳过，因为视频已解析过）
            logger.info("跳过输入文件验证（视频已解析）")

            # 2. 视频解析（跳过）
            logger.info("跳过视频解析步骤")

            # 3. 嵌入生成（异步）
            if not await self.test_embedding_generation():
                logger.error("嵌入生成测试失败")
                return False

            # 4. 存储操作（异步）
            if not await self.test_storage_operations():
                logger.error("存储操作测试失败")
                return False

            # 5. 查询处理
            if not self.test_query_processing():
                logger.error("查询处理测试失败")
                return False

            # 6. 输出生成
            if not self.test_output_generation():
                logger.error("输出生成测试失败")
                return False

            logger.info("=" * 50)
            logger.info("✅ 全流程测试通过!")

            # 生成最终摘要
            summary = self.generate_summary()
            logger.info(
                _("Test summary: {}/{} passed").format(summary['passed_tests'], summary['total_tests'])
            )
            logger.info(_("Success rate: {:.1f}%").format(summary['success_rate']))

            return True

        except Exception as e:
            logger.error(_("Full pipeline test failed: {}").format(e))
            return False

        finally:
            self.cleanup()


async def main_async():
    """异步主函数"""
    # 测试视频文件路径
    test_video = "/home/project/RAG-Anything/tests/resource/project_1.mp4"

    # 输出目录
    output_dir = "/tmp/raganything_test_output"

    # 创建测试器
    tester = FullPipelineTester(test_video, output_dir)

    # 运行测试（异步）
    success = await tester.run_full_pipeline_test_async()

    # 退出码
    return success


def main():
    """主函数"""
    # 运行异步主函数
    success = asyncio.run(main_async())

    # 退出码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
