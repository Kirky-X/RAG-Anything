import os
import cv2
import time
import json
import logging
from pathlib import Path
from raganything.parser.video_parser import VideoParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerificationScript")

def verify_video_parser():
    """
    Verification script for VideoParser.
    1. Slices 2 minutes from project_1.mp4
    2. Runs VideoParser on the slice
    3. Saves results
    """
    
    # Config
    SOURCE_VIDEO = Path("/home/project/RAG-Anything/tests/resource/project_1.mp4")
    OUTPUT_DIR = Path("/home/project/RAG-Anything/tests/resource/verification_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SOURCE_VIDEO.exists():
        logger.error(f"Source video not found: {SOURCE_VIDEO}")
        return

    # 1. Use Full Video (No Slicing)
    logger.info("Step 1: Using full video for analysis...")
    target_path = SOURCE_VIDEO
    
    # 2. Run VideoParser
    logger.info("Step 2: Running VideoParser...")
    
    parser = VideoParser()
    
    try:
        result = parser.parse_video(
            file_path=target_path,
            output_dir=OUTPUT_DIR,
            fps=0.5, # 1 frame every 2 seconds
            cleanup_frames=True
        )
        
        # 3. Save Results
        logger.info("Step 3: Saving results...")
        result_json_path = OUTPUT_DIR / "analysis_result.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Verification complete. Results saved to {result_json_path}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_video_parser()
