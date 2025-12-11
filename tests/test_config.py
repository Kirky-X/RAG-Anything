import os
import pytest
import tempfile
from pathlib import Path
from raganything.config import RAGAnythingConfig

def _write_tmp_toml(content: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(path)

class TestRAGAnythingConfig:
    
    def test_property_setters_and_getters(self):
        """Test configuration properties getters and setters."""
        cfg = RAGAnythingConfig()

        # Directory
        cfg.working_dir = "/tmp/wd"
        cfg.parser_output_dir = "/tmp/out"
        assert cfg.working_dir == "/tmp/wd"
        assert cfg.parser_output_dir == "/tmp/out"

        # Parsing
        cfg.parser = "docling"
        cfg.parse_method = "txt"
        cfg.display_content_stats = False
        assert cfg.parser == "docling"
        assert cfg.parse_method == "txt"
        assert cfg.display_content_stats is False

        # Multimodal
        cfg.enable_image_processing = True
        cfg.enable_table_processing = False
        assert cfg.enable_image_processing is True
        assert cfg.enable_table_processing is False

        # Batch
        cfg.max_concurrent_files = 2
        cfg.supported_file_extensions = [".pdf", ".png"]
        assert cfg.max_concurrent_files == 2
        assert cfg.supported_file_extensions == [".pdf", ".png"]

        # Context
        cfg.context_window = 3
        cfg.max_context_tokens = 1000
        assert cfg.context_window == 3
        assert cfg.max_context_tokens == 1000

        # LLM
        cfg.llm_provider = "openai"
        cfg.llm_model = "gpt-4o-mini"
        cfg.llm_api_base = "https://api.openai.com/v1"
        assert cfg.llm_provider == "openai"
        assert cfg.llm_model == "gpt-4o-mini"
        assert cfg.llm_api_base == "https://api.openai.com/v1"

        # Embedding
        cfg.embedding_provider = "openai"
        cfg.embedding_dim = 1024
        assert cfg.embedding_provider == "openai"
        assert cfg.embedding_dim == 1024

    def test_env_fallback(self, monkeypatch):
        """Test environment variable fallback."""
        # Ensure no default config.toml is loaded
        fd, p = tempfile.mkstemp(suffix=".toml")
        os.close(fd)
        monkeypatch.setenv("CONFIG_TOML", p)
        
        monkeypatch.setenv("WORKING_DIR", "/tmp/rag_storage")
        monkeypatch.setenv("PARSER", "docling")
        
        cfg = RAGAnythingConfig()
        assert cfg.working_dir == "/tmp/rag_storage"
        assert cfg.parser == "docling"
        
        os.unlink(p)

    def test_toml_loading_overrides_env(self, monkeypatch):
        """Test that TOML configuration overrides environment variables."""
        monkeypatch.setenv("WORKING_DIR", "/env/storage")
        
        content = """
        [directory]
        working_dir = "/toml/storage"
        parser_output_dir = "/toml/output"

        [raganything.parsing]
        parser = "mineru"
        parse_method = "ocr"
        """
        path = _write_tmp_toml(content)
        try:
            monkeypatch.setenv("CONFIG_TOML", str(path))
            cfg = RAGAnythingConfig()
            assert cfg.working_dir == "/toml/storage"
            assert cfg.parser == "mineru"
        finally:
            path.unlink(missing_ok=True)

    def test_validation_invalid_parser(self, monkeypatch):
        """Test validation of invalid parser configuration."""
        content = """
        [raganything.parsing]
        parser = "unknown_parser_xyz"
        """
        path = _write_tmp_toml(content)
        try:
            monkeypatch.setenv("CONFIG_TOML", str(path))
            with pytest.raises(ValueError):
                RAGAnythingConfig()
        finally:
            path.unlink(missing_ok=True)
