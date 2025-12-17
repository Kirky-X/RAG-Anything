import tempfile
import unittest
from pathlib import Path

from PIL import Image

from raganything.parser.vlm_parser import VlmParser


class TestVlmParser(unittest.IsolatedAsyncioTestCase):
    async def test_parse_image_async_offline(self):
        tmpdir = tempfile.TemporaryDirectory()
        img_path = Path(tmpdir.name) / "test.jpg"
        Image.new("RGB", (120, 80), color=(200, 50, 50)).save(img_path, format="JPEG")
        cfg_path = Path(tmpdir.name) / "config.toml"
        cfg_path.write_text(
            """
            [raganything.vision]
            provider = "offline"
            model = "offline"
            timeout = 5
            max_retries = 0
            """
        )
        parser = VlmParser(config_path=str(cfg_path))
        result = await parser.parse_image_async(img_path)
        self.assertEqual(result["filename"], "test.jpg")
        self.assertTrue("Image size:" in result["description"])
        self.assertGreater(result["confidence"], 0.0)
        tmpdir.cleanup()

    async def test_parse_image_async_not_found(self):
        parser = VlmParser(config_path="config.toml")
        result = await parser.parse_image_async(Path("__not_exist__.jpg"))
        self.assertIn("File not found", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()
