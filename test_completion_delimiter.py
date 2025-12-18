#!/usr/bin/env python3
"""Test script to verify completion delimiter fix"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_completion_delimiter_fix():
    """Test the completion delimiter fix function"""
    try:
        from raganything.patches.lightrag_patch import fix_completion_delimiter
        
        # Test cases
        test_cases = [
            # Case 1: Missing delimiter, ends with entity
            ("Found entities: person, location", "Found entities: person, location\n<|COMPLETE|>"),
            
            # Case 2: Missing delimiter, ends with relation  
            ("Found relations: works_at, lives_in", "Found relations: works_at, lives_in\n<|COMPLETE|>"),
            
            # Case 3: Already has delimiter
            ("Found entities: person\n<|COMPLETE|>", "Found entities: person\n<|COMPLETE|>"),
            
            # Case 4: Empty string
            ("", ""),
            
            # Case 5: Just whitespace
            ("   ", "   "),
            
            # Case 6: Improperly formatted, not ending with entity/relation
            ("Random text", "Random text\n<|COMPLETE|>"),
        ]
        
        logger.info("Testing completion delimiter fix function...")
        
        for i, (input_text, expected_output) in enumerate(test_cases, 1):
            result = fix_completion_delimiter(input_text)
            if result == expected_output:
                logger.info(f"✓ Test case {i} passed")
            else:
                logger.error(f"✗ Test case {i} failed")
                logger.error(f"  Input: {repr(input_text)}")
                logger.error(f"  Expected: {repr(expected_output)}")
                logger.error(f"  Got: {repr(result)}")
                return False
        
        logger.info("All completion delimiter tests passed!")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import fix_completion_delimiter: {e}")
        return False
    except Exception as e:
        logger.error(f"Error testing completion delimiter fix: {e}")
        return False

def test_operate_integration():
    """Test the integration in operate.py"""
    try:
        # Read the operate.py file to verify our patch is integrated
        operate_file = project_root / ".venv" / "lib" / "python3.12" / "site-packages" / "lightrag" / "operate.py"
        
        if not operate_file.exists():
            logger.error(f"operate.py not found at {operate_file}")
            return False
            
        content = operate_file.read_text()
        
        # Check if our fix is integrated
        if "fix_completion_delimiter" in content:
            logger.info("✓ fix_completion_delimiter is integrated in operate.py")
            return True
        else:
            logger.error("✗ fix_completion_delimiter is NOT integrated in operate.py")
            return False
            
    except Exception as e:
        logger.error(f"Error checking operate.py integration: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting completion delimiter fix verification...")
    
    success = True
    
    # Test the fix function
    success &= test_completion_delimiter_fix()
    
    # Test the integration
    success &= test_operate_integration()
    
    if success:
        logger.info("All tests passed! Completion delimiter fix is working correctly.")
        sys.exit(0)
    else:
        logger.error("Some tests failed. Please check the implementation.")
        sys.exit(1)