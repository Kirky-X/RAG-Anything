import unittest
from raganything.config import RAGAnythingConfig


class TestRAGAnythingConfigCoverage(unittest.TestCase):
    def test_property_setters_and_getters(self):
        cfg = RAGAnythingConfig()

        # Directory
        self.assertIsInstance(cfg.working_dir, str)
        self.assertIsInstance(cfg.parser_output_dir, str)
        cfg.working_dir = "/tmp/wd"
        cfg.parser_output_dir = "/tmp/out"
        self.assertEqual(cfg.working_dir, "/tmp/wd")
        self.assertEqual(cfg.parser_output_dir, "/tmp/out")

        # Parsing
        cfg.parser = "docling"
        cfg.parse_method = "txt"
        cfg.display_content_stats = False
        self.assertEqual(cfg.parser, "docling")
        self.assertEqual(cfg.parse_method, "txt")
        self.assertFalse(cfg.display_content_stats)

        # Multimodal
        cfg.enable_image_processing = True
        cfg.enable_table_processing = False
        cfg.enable_equation_processing = True
        self.assertTrue(cfg.enable_image_processing)
        self.assertFalse(cfg.enable_table_processing)
        self.assertTrue(cfg.enable_equation_processing)

        # Batch
        cfg.max_concurrent_files = 2
        cfg.supported_file_extensions = [".pdf", ".png"]
        cfg.recursive_folder_processing = False
        self.assertEqual(cfg.max_concurrent_files, 2)
        self.assertEqual(cfg.supported_file_extensions, [".pdf", ".png"])
        self.assertFalse(cfg.recursive_folder_processing)

        # Context
        cfg.context_window = 3
        cfg.context_mode = "page"
        cfg.max_context_tokens = 1000
        cfg.include_headers = True
        cfg.include_captions = False
        cfg.context_filter_content_types = ["text", "table"]
        cfg.content_format = "minerU"
        self.assertEqual(cfg.context_window, 3)
        self.assertEqual(cfg.context_mode, "page")
        self.assertEqual(cfg.max_context_tokens, 1000)
        self.assertTrue(cfg.include_headers)
        self.assertFalse(cfg.include_captions)
        self.assertEqual(cfg.context_filter_content_types, ["text", "table"])
        self.assertEqual(cfg.content_format, "minerU")

        # LLM
        cfg.llm_provider = "openai"
        cfg.llm_model = "gpt-4o-mini"
        cfg.llm_api_base = "https://api.openai.com/v1"
        cfg.llm_api_key = "key"
        cfg.llm_timeout = 30
        cfg.llm_max_retries = 1
        self.assertEqual(cfg.llm_provider, "openai")
        self.assertEqual(cfg.llm_model, "gpt-4o-mini")
        self.assertEqual(cfg.llm_api_base, "https://api.openai.com/v1")
        self.assertEqual(cfg.llm_api_key, "key")
        self.assertEqual(cfg.llm_timeout, 30)
        self.assertEqual(cfg.llm_max_retries, 1)

        # Embedding
        cfg.embedding_provider = "openai"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_api_base = "https://api.openai.com/v1"
        cfg.embedding_api_key = "ekey"
        cfg.embedding_dim = 1024
        cfg.embedding_func_max_async = 8
        cfg.embedding_batch_num = 4
        self.assertEqual(cfg.embedding_provider, "openai")
        self.assertEqual(cfg.embedding_model, "text-embedding-3-small")
        self.assertEqual(cfg.embedding_api_base, "https://api.openai.com/v1")
        self.assertEqual(cfg.embedding_api_key, "ekey")
        self.assertEqual(cfg.embedding_dim, 1024)
        self.assertEqual(cfg.embedding_func_max_async, 8)
        self.assertEqual(cfg.embedding_batch_num, 4)

        # Vision
        cfg.vision_provider = "openai"
        cfg.vision_model = "gpt-4o-mini"
        cfg.vision_api_base = "https://api.openai.com/v1"
        cfg.vision_api_key = "vkey"
        cfg.vision_timeout = 40
        cfg.vision_max_retries = 2
        self.assertEqual(cfg.vision_provider, "openai")
        self.assertEqual(cfg.vision_model, "gpt-4o-mini")
        self.assertEqual(cfg.vision_api_base, "https://api.openai.com/v1")
        self.assertEqual(cfg.vision_api_key, "vkey")
        self.assertEqual(cfg.vision_timeout, 40)
        self.assertEqual(cfg.vision_max_retries, 2)


if __name__ == "__main__":
    unittest.main()

