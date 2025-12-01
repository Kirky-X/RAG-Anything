import os
import sys
import random
import subprocess
import json
import asyncio
import logging
from pathlib import Path
from typing import List

# Add project root to path to ensure imports work
project_root = Path("/home/project/RAG-Anything")
sys.path.append(str(project_root))

from raganything.parser.vlm_parser import VlmParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vlm_verification.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

def extract_frames(video_path: Path, output_dir: Path, count: int = 5) -> List[Path]:
    """Extract random frames from video using ffmpeg."""
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return []
        
    # Get duration
    duration = 60.0 # Default fallback
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            logger.warning("Could not parse duration from ffprobe output. Using default.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("ffprobe failed or not found. Using default duration/random sampling strategy.")
        # If we know the file is project_1.mp4, we know it's ~3412s long from previous context
        if "project_1.mp4" in str(video_path):
            duration = 3400.0
        
    frames = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting {count} frames from video (duration: {duration:.2f}s)...")
    
    for i in range(count):
        timestamp = random.uniform(0, duration)
        # Format: frame_<timestamp>.jpg (replace . with _ to avoid double extension confusion)
        ts_str = f"{timestamp:.2f}".replace(".", "_")
        output_file = output_dir / f"frame_{ts_str}.jpg"
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", str(output_file)
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            frames.append(output_file)
            logger.info(f"[{i+1}/{count}] Extracted frame: {output_file.name} at {timestamp:.2f}s")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract frame at {timestamp}s: {e}")
            
    return frames

async def run_verification():
    base_dir = Path("/home/project/RAG-Anything/tests/resource")
    video_file = base_dir / "project_1.mp4"
    
    # 1. Extract Frames
    logger.info("=== Step 1: Extracting Frames ===")
    frames = extract_frames(video_file, base_dir, count=5)
    
    if not frames:
        logger.error("No frames extracted. Aborting.")
        return
        
    # 2. Initialize Parser
    logger.info("=== Step 2: Initializing VlmParser ===")
    try:
        # Point to the config file explicitly
        config_path = project_root / "config.toml"
        parser = VlmParser(config_path=str(config_path))
        logger.info("VlmParser initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to init parser: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 3. Process Frames
    logger.info("=== Step 3: Processing Frames ===")
    results = []
    
    for i, frame in enumerate(frames):
        logger.info(f"[{i+1}/{len(frames)}] Processing {frame.name}...")
        
        # Using a timeout for the task to enforce the 10s requirement on the caller side as well
        try:
            # We requested 10s timeout in config, but let's be safe
            result = await asyncio.wait_for(
                parser.parse_image_async(
                    frame, 
                    prompt="Describe this image in detail.",
                    max_size=800 # Optimize for speed
                ),
                timeout=120.0 # Allow slightly more than internal timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing {frame.name}")
            result = {
                "filename": frame.name,
                "timestamp": "",
                "description": "",
                "error": "Timeout",
                "latency_seconds": 120.0,
                "confidence": 0.0
            }
            
        results.append(result)
        desc_preview = result.get('description', '')[:50].replace('\n', ' ')
        logger.info(f"Completed {frame.name}. Latency: {result.get('latency_seconds')}s. Desc: {desc_preview}...")
        
    # 4. Generate Report
    logger.info("=== Step 4: Generating Report ===")
    report_path = base_dir / "vlm_test_report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
    
    # Print Console Summary
    print("\n" + "="*50)
    print("VLM PARSER VERIFICATION SUMMARY")
    print("="*50)
    for res in results:
        status = "FAILED" if res.get("error") else "SUCCESS"
        error_msg = f" (Error: {res.get('error')})" if res.get("error") else ""
        print(f"File: {res.get('filename')}")
        print(f"Status: {status}{error_msg}")
        print(f"Time: {res.get('latency_seconds')}s")
        print(f"Confidence: {res.get('confidence')}")
        print(f"Content: {res.get('description', '')}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(run_verification())
