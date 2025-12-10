
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import scripts.benchmark_audio as benchmark_audio

def test_generate_dummy_audio(tmp_path):
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.name = str(tmp_path / "test_audio.wav")
        path = benchmark_audio.generate_dummy_audio(duration_sec=0.1)
        assert isinstance(path, Path)
        assert path.exists()

def test_run_benchmark_pass():
    mock_parser = MagicMock()
    audio_path = Path("test.wav")
    
    result = benchmark_audio.run_benchmark_pass(mock_parser, audio_path, iterations=2)
    
    assert "min" in result
    assert "max" in result
    assert "avg" in result
    assert "total" in result
    assert mock_parser.parse_audio.call_count == 3  # 1 warmup + 2 iterations

@patch("scripts.benchmark_audio.device_manager")
@patch("scripts.benchmark_audio.psutil")
@patch("scripts.benchmark_audio.generate_dummy_audio")
@patch("scripts.benchmark_audio.AudioParser")
def test_benchmark_audio_parser(mock_parser_cls, mock_gen_audio, mock_psutil, mock_device_mgr):
    # Setup mocks
    mock_device_mgr.device = "cpu"
    mock_psutil.cpu_percent.return_value = 10.0
    mock_psutil.virtual_memory.return_value.available = 1024**3 * 8
    
    mock_audio_path = MagicMock(spec=Path)
    mock_audio_path.exists.return_value = True
    mock_gen_audio.return_value = mock_audio_path
    
    # Run benchmark
    benchmark_audio.benchmark_audio_parser()
    
    # Verify calls
    mock_parser_cls.return_value._load_model.assert_called_once()
    mock_parser_cls.return_value.parse_audio.assert_called()
    mock_audio_path.unlink.assert_called_once()
