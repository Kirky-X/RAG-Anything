import subprocess
import sys

from raganything import __version__


def test_version():
    assert __version__ is not None


def test_cli_help():
    """Test that the CLI help command works."""
    # Use sys.executable to run the module directly instead of relying on 'uv run'
    # which might not be available or configured in the test environment.
    # We use raganything.main since raganything.cli does not exist.
    result = subprocess.run(
        [sys.executable, "-m", "raganything.main", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "RAGAnything CLI" in result.stdout
    assert "usage:" in result.stdout
