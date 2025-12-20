#!/usr/bin/env python3
"""
RAG-Anything 媒体处理工具集

提供音频、视频和图像处理相关的工具函数，包括格式转换、帧提取、基准测试等。
"""

import asyncio
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import psutil
import soundfile as sf

# 获取项目根目录（相对于脚本位置）
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))

try:
    from raganything.parser.audio_parser import AudioParser
    from raganything.parser.vlm_parser import VlmParser
    from raganything.models.device import device_manager
    from raganything.logger import get_i18n_logger
    from raganything.i18n import _
except ImportError as e:
    print(f"导入RAG-Anything模块失败: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)


class MediaProcessor:
    """媒体处理器：提供音频、视频处理功能"""
    
    def __init__(self):
        self.logger = get_i18n_logger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志"""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # 清除现有的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler("/tmp/media_processing.log"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )
    
    def extract_audio(self, video_path: Path, output_path: Path, 
                     audio_format: str = "wav", quality: str = "high") -> bool:
        """
        从视频中提取音频
        
        Args:
            video_path: 输入视频文件路径
            output_path: 输出音频文件路径
            audio_format: 音频格式 (wav, mp3, m4a)
            quality: 音质等级 (high, medium, low)
        
        Returns:
            成功返回True，失败返回False
        """
        self.logger.info(_("开始提取音频: {} -> {}").format(video_path, output_path))
        
        if not video_path.exists():
            self.logger.error(_("视频文件不存在: {}").format(video_path))
            return False
        
        # 根据格式和质量设置编码参数
        codec_params = {
            "wav": {"codec": "pcm_s16le", "bitrate": "192k" if quality == "high" else "128k"},
            "mp3": {"codec": "libmp3lame", "bitrate": "320k" if quality == "high" else "192k"},
            "m4a": {"codec": "aac", "bitrate": "256k" if quality == "high" else "160k"},
        }
        
        params = codec_params.get(audio_format, codec_params["wav"])
        
        command = [
            "ffmpeg",
            "-y",  # 覆盖输出文件
            "-i", str(video_path),
            "-vn",  # 禁用视频
            "-acodec", params["codec"],
            "-ab", params["bitrate"],
            str(output_path),
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            self.logger.info(_("音频提取成功: {}").format(output_path))
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(_("音频提取失败: {}").format(e.stderr))
            return False
    
    def extract_frames(self, video_path: Path, output_dir: Path, 
                      count: int = 5, frame_format: str = "jpg") -> List[Path]:
        """
        从视频中提取随机帧
        
        Args:
            video_path: 输入视频文件路径
            output_dir: 输出目录
            count: 要提取的帧数
            frame_format: 帧格式 (jpg, png)
        
        Returns:
            提取的帧文件路径列表
        """
        if not video_path.exists():
            self.logger.error(_("视频文件不存在: {}").format(video_path))
            return []
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取视频时长
        duration = self._get_video_duration(video_path)
        if duration is None:
            return []
        
        # 生成随机时间点
        timestamps = sorted([random.uniform(0, duration) for _ in range(count)])
        
        frame_paths = []
        for i, timestamp in enumerate(timestamps):
            frame_path = output_dir / f"frame_{i+1:03d}.{frame_format}"
            
            command = [
                "ffmpeg",
                "-y",
                "-ss", str(timestamp),
                "-i", str(video_path),
                "-vframes", "1",
                "-q:v", "2",  # 高质量
                str(frame_path),
            ]
            
            try:
                subprocess.run(command, capture_output=True, text=True, check=True)
                frame_paths.append(frame_path)
                self.logger.info(_("提取帧 {} 成功").format(frame_path))
            except subprocess.CalledProcessError as e:
                self.logger.error(_("提取帧失败: {}").format(e.stderr))
        
        return frame_paths
    
    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """获取视频时长"""
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            self.logger.info(_("视频时长: {:.2f}秒").format(duration))
            return duration
        except (subprocess.CalledProcessError, ValueError) as e:
            self.logger.error(_("获取视频时长失败: {}").format(e))
            return None
    
    def benchmark_audio_parsing(self, audio_path: Optional[Path] = None, 
                               iterations: int = 5, duration: int = 5) -> Dict[str, Any]:
        """
        基准测试音频解析性能
        
        Args:
            audio_path: 测试音频文件路径，如果为None则生成虚拟音频
            iterations: 测试迭代次数
            duration: 虚拟音频时长（秒）
        
        Returns:
            基准测试结果字典
        """
        self.logger.info(_("开始音频解析基准测试"))
        
        # 如果没有提供音频文件，生成虚拟音频
        if audio_path is None:
            audio_path = self._generate_dummy_audio(duration)
            cleanup_audio = True
        else:
            cleanup_audio = False
        
        if not audio_path.exists():
            self.logger.error(_("音频文件不存在: {}").format(audio_path))
            return {}
        
        # 初始化解析器
        try:
            parser = AudioParser()
        except Exception as e:
            self.logger.error(_("初始化AudioParser失败: {}").format(e))
            return {}
        
        # 预热
        self.logger.info(_("预热模型..."))
        try:
            parser.parse_audio(audio_path)
        except Exception as e:
            self.logger.warning(_("预热失败: {}").format(e))
        
        # 运行基准测试
        self.logger.info(_("运行 {} 次迭代...").format(iterations))
        latencies = []
        
        for i in range(iterations):
            try:
                start_time = time.time()
                result = parser.parse_audio(audio_path)
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                self.logger.info(_("第{}次迭代: {:.4f}秒").format(i + 1, latency))
                
            except Exception as e:
                self.logger.error(_("第{}次迭代失败: {}").format(i + 1, e))
                continue
        
        # 清理虚拟音频文件
        if cleanup_audio and audio_path.exists():
            audio_path.unlink()
        
        if not latencies:
            self.logger.error(_("没有成功的迭代"))
            return {}
        
        # 计算统计信息
        stats = {
            "iterations": len(latencies),
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
        }
        
        self.logger.info(_("基准测试完成"))
        self.logger.info(_("平均延迟: {:.4f}秒 ± {:.4f}秒").format(
            stats["mean_latency"], stats["std_latency"]))
        self.logger.info(_("延迟分布 - P50: {:.4f}秒, P90: {:.4f}秒, P95: {:.4f}秒").format(
            stats["p50_latency"], stats["p90_latency"], stats["p95_latency"]))
        
        return stats
    
    def _generate_dummy_audio(self, duration_sec: int = 5, sample_rate: int = 16000) -> Path:
        """生成虚拟音频文件用于测试"""
        # 生成440Hz正弦波
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, sample_rate)
        temp_file.close()
        
        self.logger.info(_("生成虚拟音频文件: {}").format(temp_file.name))
        return Path(temp_file.name)
    
    def verify_vlm_parsing(self, video_path: Path, frame_count: int = 5) -> Dict[str, Any]:
        """
        验证VLM（视觉语言模型）解析功能
        
        Args:
            video_path: 输入视频文件路径
            frame_count: 要提取和验证的帧数
        
        Returns:
            验证结果字典
        """
        self.logger.info(_("开始VLM验证: {}").format(video_path))
        
        if not video_path.exists():
            self.logger.error(_("视频文件不存在: {}").format(video_path))
            return {"success": False, "error": "Video file not found"}
        
        try:
            # 初始化VLM解析器
            vlm_parser = VlmParser()
        except Exception as e:
            self.logger.error(_("初始化VlmParser失败: {}").format(e))
            return {"success": False, "error": f"Failed to initialize VLM parser: {e}"}
        
        # 创建临时目录存储帧
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_dir = Path(temp_dir)
            
            # 提取帧
            self.logger.info(_("提取 {} 帧用于验证").format(frame_count))
            frame_paths = self.extract_frames(video_path, frame_dir, frame_count, "jpg")
            
            if not frame_paths:
                return {"success": False, "error": "Failed to extract frames"}
            
            # 验证每帧的解析
            results = []
            for i, frame_path in enumerate(frame_paths):
                try:
                    self.logger.info(_("验证帧 {}: {}").format(i + 1, frame_path))
                    
                    start_time = time.time()
                    description = vlm_parser.parse_image(frame_path)
                    end_time = time.time()
                    
                    results.append({
                        "frame": str(frame_path),
                        "description": description,
                        "latency": end_time - start_time,
                        "success": True
                    })
                    
                    self.logger.info(_("帧 {} 描述: {}").format(i + 1, description[:100]))
                    
                except Exception as e:
                    self.logger.error(_("帧 {} 验证失败: {}").format(i + 1, e))
                    results.append({
                        "frame": str(frame_path),
                        "error": str(e),
                        "success": False
                    })
            
            # 统计结果
            successful = sum(1 for r in results if r["success"])
            total_latency = sum(r.get("latency", 0) for r in results if r["success"])
            
            summary = {
                "success": successful == len(results),
                "total_frames": len(results),
                "successful_frames": successful,
                "failed_frames": len(results) - successful,
                "average_latency": total_latency / successful if successful > 0 else 0,
                "results": results
            }
            
            self.logger.info(_("VLM验证完成 - 成功: {}/{}").format(successful, len(results)))
            return summary


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python media_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  extract-audio <视频文件> [输出文件] [格式] [质量]  - 提取音频")
        print("  extract-frames <视频文件> <输出目录> [数量] [格式]  - 提取帧")
        print("  benchmark-audio [音频文件] [迭代次数] [时长]  - 基准测试音频解析")
        print("  verify-vlm <视频文件> [帧数]  - 验证VLM解析")
        print("  help  - 显示帮助信息")
        return
    
    processor = MediaProcessor()
    command = sys.argv[1]
    
    if command == "extract-audio":
        if len(sys.argv) < 3:
            print("错误：需要指定视频文件")
            return
        
        video_path = Path(sys.argv[2])
        output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else video_path.with_suffix(".wav")
        audio_format = sys.argv[4] if len(sys.argv) > 4 else "wav"
        quality = sys.argv[5] if len(sys.argv) > 5 else "high"
        
        success = processor.extract_audio(video_path, output_path, audio_format, quality)
        if success:
            print(f"✅ 音频提取成功: {output_path}")
        else:
            print("❌ 音频提取失败")
    
    elif command == "extract-frames":
        if len(sys.argv) < 4:
            print("错误：需要指定视频文件和输出目录")
            return
        
        video_path = Path(sys.argv[2])
        output_dir = Path(sys.argv[3])
        count = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        frame_format = sys.argv[5] if len(sys.argv) > 5 else "jpg"
        
        frame_paths = processor.extract_frames(video_path, output_dir, count, frame_format)
        if frame_paths:
            print(f"✅ 成功提取 {len(frame_paths)} 帧")
            for frame_path in frame_paths:
                print(f"  - {frame_path}")
        else:
            print("❌ 帧提取失败")
    
    elif command == "benchmark-audio":
        audio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        duration = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        
        results = processor.benchmark_audio_parsing(audio_path, iterations, duration)
        if results:
            print("\n📊 基准测试结果:")
            print(f"  迭代次数: {results['iterations']}")
            print(f"  平均延迟: {results['mean_latency']:.4f}s ± {results['std_latency']:.4f}s")
            print(f"  延迟范围: {results['min_latency']:.4f}s - {results['max_latency']:.4f}s")
            print(f"  P50延迟: {results['p50_latency']:.4f}s")
            print(f"  P90延迟: {results['p90_latency']:.4f}s")
            print(f"  P95延迟: {results['p95_latency']:.4f}s")
        else:
            print("❌ 基准测试失败")
    
    elif command == "verify-vlm":
        if len(sys.argv) < 3:
            print("错误：需要指定视频文件")
            return
        
        video_path = Path(sys.argv[2])
        frame_count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        
        results = processor.verify_vlm_parsing(video_path, frame_count)
        if results["success"]:
            print(f"\n✅ VLM验证成功 - {results['successful_frames']}/{results['total_frames']} 帧")
            print(f"  平均延迟: {results['average_latency']:.4f}s")
            
            for i, result in enumerate(results["results"][:3]):  # 只显示前3个结果
                if result["success"]:
                    print(f"  帧{i+1}: {result['description'][:80]}...")
        else:
            print(f"\n❌ VLM验证失败 - {results['successful_frames']}/{results['total_frames']} 帧")
    
    elif command == "help":
        print("RAG-Anything 媒体处理工具集")
        print("\n用法: python media_tools.py <命令> [参数]")
        print("\n可用命令:")
        print("  extract-audio <视频文件> [输出文件] [格式] [质量]")
        print("    - 从视频中提取音频")
        print("    - 格式: wav, mp3, m4a (默认: wav)")
        print("    - 质量: high, medium, low (默认: high)")
        print()
        print("  extract-frames <视频文件> <输出目录> [数量] [格式]")
        print("    - 从视频中提取帧")
        print("    - 数量: 要提取的帧数 (默认: 5)")
        print("    - 格式: jpg, png (默认: jpg)")
        print()
        print("  benchmark-audio [音频文件] [迭代次数] [时长]")
        print("    - 基准测试音频解析性能")
        print("    - 迭代次数: 测试次数 (默认: 5)")
        print("    - 时长: 虚拟音频时长，秒 (默认: 5)")
        print()
        print("  verify-vlm <视频文件> [帧数]")
        print("    - 验证VLM（视觉语言模型）解析功能")
        print("    - 帧数: 要验证的帧数 (默认: 5)")
    
    else:
        print(f"未知命令: {command}")
        print("使用 'python media_tools.py help' 查看帮助")


if __name__ == "__main__":
    main()