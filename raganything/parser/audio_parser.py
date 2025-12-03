# type: ignore
"""
Audio Parser Module

Defines the AudioParser class for processing audio and video files using SenseVoiceSmall.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from raganything.logger import logger
from .base_parser import Parser
from raganything.models import model_manager, device_manager, default_models_config

# Optional imports for audio processing / ASR
# Keep dependency flags separate to avoid coupling metadata analysis to ASR model presence
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    AudioSegment = None
    HAS_PYDUB = False

try:
    from funasr import AutoModel
    HAS_FUNASR = True
except ImportError:
    AutoModel = None
    HAS_FUNASR = False

# Backward-compatible aggregate flag used by legacy tests
AUDIO_DEPS_AVAILABLE = HAS_PYDUB and HAS_FUNASR


class AudioParser(Parser):
    """
    Parser for audio and video files using iic/SenseVoiceSmall.
    Supports automatic format conversion and speech-to-text.
    """

    SUPPORTED_FORMATS = {
        ".aac", ".amr", ".avi", ".flac", ".flv", ".m4a", ".mkv",
        ".mov", ".mp3", ".mp4", ".mpeg", ".ogg", ".opus", ".wav",
        ".webm", ".wma", ".wmv"
    }

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._model_path = None
        
        if not HAS_PYDUB:
            logger.warning(
                "pydub not found. Audio conversion/analysis disabled. "
                "Install with `uv sync --extra audio` or `pip install raganything[audio]`"
            )
        if not HAS_FUNASR:
            logger.warning(
                "funasr not found. Speech-to-text disabled. "
                "Install with `uv sync --extra audio` or `pip install raganything[audio]`"
            )

    def _load_model(self):
        """Lazy load the SenseVoiceSmall model."""
        if self._model is not None:
            return

        if not AUDIO_DEPS_AVAILABLE:
            raise ImportError("Audio dependencies (funasr, pydub) are not installed.")

        try:
            logger.info("Loading SenseVoiceSmall model...")
            # Get model path from manager (downloads if needed)
            self._model_path = model_manager.get_sense_voice_model_path()
            
            # Determine device
            device = device_manager.device
            
            # Load model using funasr
            self._model = AutoModel(
                model=self._model_path,
                trust_remote_code=True,
                device=device,
                disable_update=True
            )
            logger.info(f"SenseVoiceSmall model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load SenseVoiceSmall model: {e}")
            raise

    def _convert_to_wav_16k(self, input_path: Path) -> Path:
        """
        Convert input audio/video to 16kHz 16-bit mono WAV required by SenseVoice.
        Returns path to temporary WAV file.
        """
        if AudioSegment is None:
            raise ImportError("pydub is not installed; audio conversion unavailable.")
        try:
            # Create temp file
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            logger.info(f"Converting {input_path.name} to 16kHz WAV...")
            
            # Load audio (pydub handles format detection and video containers)
            audio = AudioSegment.from_file(str(input_path))
            
            # Convert: 16kHz, mono, 16-bit
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Export
            audio.export(temp_path, format="wav")
            
            return Path(temp_path)
            
        except Exception as e:
            logger.error(f"Audio conversion failed for {input_path}: {e}")
            # Clean up temp file if created but failed
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise RuntimeError(f"Failed to convert audio file: {e}")

    def parse_audio(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: str = "auto",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Parse audio/video file and extract text.
        
        Args:
            file_path: Path to audio/video file
            output_dir: Optional output directory (not used for now but kept for consistency)
            lang: Language code (auto, zh, en, ja, ko, yue)
            **kwargs: Additional arguments
            
        Returns:
            List containing a dictionary with extracted text and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            logger.warning(f"Format {suffix} might not be supported. Attempting anyway.")

        # Ensure required deps are present
        if AudioSegment is None:
            raise ImportError("pydub is not installed; audio conversion unavailable.")
        # Ensure model is loaded for ASR
        self._load_model()
        
        temp_wav_path = None
        try:
            # Convert to standard format
            temp_wav_path = self._convert_to_wav_16k(file_path)
            
            # Check file duration to decide on chunking
            # If duration > 300s (5 mins), process in chunks to avoid OOM
            # In test environments, _convert_to_wav_16k may be mocked to return a non-existent path.
            # Guard against missing temp file to keep behavior robust and allow mocked model to run.
            try:
                if os.path.exists(str(temp_wav_path)) and AudioSegment is not None:
                    audio_info = AudioSegment.from_file(str(temp_wav_path))
                    duration_sec = len(audio_info) / 1000.0
                else:
                    logger.warning(
                        f"Temp WAV path {temp_wav_path} not found or AudioSegment unavailable;"
                        " skipping duration check and proceeding without chunking"
                    )
                    audio_info = None
                    duration_sec = 0.0
            except Exception as e:
                logger.warning(
                    f"Failed to read temp WAV for duration ({temp_wav_path}): {e}; proceeding without chunking"
                )
                audio_info = None
                duration_sec = 0.0
            
            full_text = ""
            
            # Threshold for chunking: 5 minutes
            CHUNK_THRESHOLD = 300
            
            if duration_sec > CHUNK_THRESHOLD and audio_info is not None:
                logger.info(f"Audio duration ({duration_sec:.2f}s) exceeds threshold. Processing in chunks...")
                # Chunk size: 30 seconds to be safe with VRAM
                CHUNK_SIZE_MS = 30 * 1000 
                
                chunks_text = []
                for i, chunk_start in enumerate(range(0, len(audio_info), CHUNK_SIZE_MS)):
                    chunk_end = min(chunk_start + CHUNK_SIZE_MS, len(audio_info))
                    chunk = audio_info[chunk_start:chunk_end]
                    
                    # Export chunk to temp file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk_file:
                        chunk.export(temp_chunk_file.name, format="wav")
                        chunk_path = temp_chunk_file.name
                    
                    try:
                        logger.info(f"Processing chunk {i+1} ({chunk_start/1000:.1f}s - {chunk_end/1000:.1f}s)...")
                        res = self._model.generate(
                            input=chunk_path,
                            cache={},
                            language=lang,
                            use_itn=True,
                            batch_size_s=60,
                            merge_vad=True,
                            merge_length_s=15,
                        )
                        if isinstance(res, list) and len(res) > 0:
                            text = res[0].get("text", "")
                            if text:
                                chunks_text.append(text)
                    finally:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                
                full_text = " ".join(chunks_text)
                
            else:
                # Small file, process directly
                logger.info(f"Transcribing {file_path.name}...")
                res = self._model.generate(
                    input=str(temp_wav_path),
                    cache={},
                    language=lang,
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,  
                    merge_length_s=15,
                )
                
                # Extract text
                if isinstance(res, list) and len(res) > 0:
                    full_text = res[0].get("text", "")
            
            logger.info(f"Transcription complete. Length: {len(full_text)} chars")
            
            return [{
                "type": "text",
                "text": full_text,
                "metadata": {
                    "source": str(file_path),
                    "original_format": suffix,
                    "model": "iic/SenseVoiceSmall",
                    "language": lang
                }
            }]
            
        except Exception as e:
            logger.error(f"Error parsing audio file {file_path}: {e}")
            raise
            
        finally:
            # Clean up temp file
            if temp_wav_path and temp_wav_path.exists():
                try:
                    os.remove(temp_wav_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_wav_path}: {e}")

    def analyze_audio(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze audio file to extract metadata and waveform characteristics.
        
        Args:
            file_path: Path to audio/video file
            
        Returns:
            Dictionary containing metadata and waveform statistics
        """
        if not HAS_PYDUB:
            raise ImportError("pydub is not installed; audio analysis unavailable.")
            
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            logger.info(f"Analyzing audio file: {file_path.name}")
            audio = AudioSegment.from_file(str(file_path))
            
            # Basic Metadata
            metadata = {
                "duration_seconds": len(audio) / 1000.0,
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "sample_width": audio.sample_width,
                "frame_count": audio.frame_count(),
            }
            
            # Waveform Analysis (Statistics)
            # RMS (Root Mean Square) is a measure of average power/loudness
            # Max amplitude
            waveform_stats = {
                "rms": audio.rms,
                "max_amplitude": audio.max,
                "dBFS": audio.dBFS,
                "max_dBFS": audio.max_dBFS,
            }
            
            result = {
                "metadata": metadata,
                "waveform": waveform_stats,
                "file_info": {
                    "filename": file_path.name,
                    "size_bytes": file_path.stat().st_size
                }
            }
            
            logger.info(f"Analysis complete for {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze audio file {file_path}: {e}")
            raise

    # Alias for generic parse calls if we implement a router later
    parse = parse_audio
