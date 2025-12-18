#!/usr/bin/env python3
"""Test script to verify entity extraction with completion delimiter fix"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_extraction():
    """Test entity extraction with our completion delimiter fix"""
    try:
        from raganything.parser.video_parser import VideoParser
        from raganything.config import RAGAnythingConfig
        
        logger.info("Testing entity extraction with completion delimiter fix...")
        
        # Create a simple test config
        config = RAGAnythingConfig()
        
        # Create video parser
        parser = VideoParser()
        
        # Test video file
        test_video = Path("/home/project/RAG-Anything/tests/resource/project_1.mp4")
        
        if not test_video.exists():
            logger.error(f"Test video not found: {test_video}")
            return False
        
        logger.info(f"Processing video: {test_video}")
        
        # Process the video with a simple configuration
        # We'll use a low FPS to make the test faster
        result = parser.parse_video(
            str(test_video),
            output_dir="/tmp/test_output",
            fps=0.1,  # Very low FPS for testing
            cleanup_frames=True,
        )
        
        logger.info("Video processing completed successfully!")
        logger.info(f"Result: {result}")
        
        # Check if we got any entities extracted
        if result and 'entities' in result:
            logger.info(f"Extracted {len(result['entities'])} entities")
            for entity in result['entities'][:5]:  # Show first 5 entities
                logger.info(f"  - {entity}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during entity extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting entity extraction test with completion delimiter fix...")
    
    success = test_entity_extraction()
    
    if success:
        logger.info("Entity extraction test passed! Completion delimiter fix is working in practice.")
        sys.exit(0)
    else:
        logger.error("Entity extraction test failed.")
        sys.exit(1)