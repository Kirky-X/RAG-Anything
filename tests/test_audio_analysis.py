
import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from raganything.parser.audio_parser import AudioParser

class TestAudioParserAnalysis(unittest.TestCase):
    def setUp(self):
        self.parser = AudioParser()
        # Create a dummy WAV file for testing
        self.test_dir = tempfile.mkdtemp()
        self.test_audio_path = Path(self.test_dir) / "test_audio.wav"
        
        # Create a simple silent wav file
        try:
            from pydub import AudioSegment
            # 1 second of silence
            audio = AudioSegment.silent(duration=1000, frame_rate=16000)
            with open(self.test_audio_path, 'wb') as f:
                audio.export(f, format="wav")
        except ImportError:
            self.skipTest("pydub not installed")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_analyze_audio_metadata(self):
        """Test that analyze_audio extracts correct metadata."""
        result = self.parser.analyze_audio(self.test_audio_path)
        
        self.assertIn("metadata", result)
        self.assertIn("waveform", result)
        self.assertIn("file_info", result)
        
        metadata = result["metadata"]
        self.assertAlmostEqual(metadata["duration_seconds"], 1.0, places=1)
        self.assertEqual(metadata["channels"], 1)
        self.assertEqual(metadata["sample_rate"], 16000)
        self.assertEqual(metadata["sample_width"], 2) # 16-bit = 2 bytes

    def test_analyze_audio_waveform(self):
        """Test that analyze_audio extracts waveform statistics."""
        result = self.parser.analyze_audio(self.test_audio_path)
        waveform = result["waveform"]
        
        # Since it's silent audio
        self.assertEqual(waveform["max_amplitude"], 0)
        self.assertEqual(waveform["rms"], 0)
        self.assertEqual(waveform["dBFS"], -float('inf'))

    def test_analyze_missing_file(self):
        """Test that analyze_audio raises FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            self.parser.analyze_audio("non_existent.wav")

if __name__ == '__main__':
    unittest.main()
