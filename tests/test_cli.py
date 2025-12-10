import subprocess
import pytest
from raganything import __version__

def test_version():
    assert __version__ is not None

def test_cli_help():
    """Test that the CLI help command works."""
    result = subprocess.run(
        ["uv", "run", "raganything", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "RAGAnything CLI" in result.stdout
    assert "usage: raganything" in result.stdout
