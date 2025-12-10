import unittest
import tempfile
from pathlib import Path
from pydub import AudioSegment
from raganything.parser.audio_parser import AudioParser

class TestAudioParser(unittest.TestCase):
    def setUp(self):
        self.parser = AudioParser()

    def test_supported_formats(self):
        self.assertIn(".wav", self.parser.SUPPORTED_FORMATS)
        self.assertIn(".mp3", self.parser.SUPPORTED_FORMATS)
        self.assertIn(".mp4", self.parser.SUPPORTED_FORMATS)

    def test_convert_to_wav_16k_real(self):
        tone = AudioSegment.silent(duration=1000).set_frame_rate(22050).set_channels(2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            # Fix: pydub export takes a file-like object or filename, 
            # but using filename directly keeps it open in pydub's internal implementation sometimes?
            # Actually, using open file handle is safer.
            tone.export(tf, format="wav")
            src = Path(tf.name)
        out = self.parser._convert_to_wav_16k(src)
        self.assertTrue(out.exists())
        with open(str(out), 'rb') as f:
            AudioSegment.from_file(f)
        out.unlink()
        src.unlink()

    def test_parse_audio_real(self):
        tone = AudioSegment.silent(duration=500)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tone.export(tf, format="wav")
            src = Path(tf.name)
        try:
            results = self.parser.parse_audio(src)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["type"], "text")
            self.assertIsInstance(results[0]["text"], str)
        except ImportError:
            self.skipTest("Audio dependencies not installed")
        finally:
            src.unlink()

if __name__ == "__main__":
    unittest.main()
