import json
import logging
import subprocess
import sys
from pathlib import Path

from raganything.parser.audio_parser import AudioParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/tmp/audio_extraction_process.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True  # Ensure we overwrite any existing config
)
logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Path) -> bool:
    """
    Extract audio from video using ffmpeg.
    """
    logger.info(f"Starting audio extraction: {video_path} -> {output_path}")

    command = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-i", str(video_path),
        "-vn",  # Disable video
        "-acodec", "copy",  # Copy audio stream (no transcoding) if compatible, else ffmpeg might default
        # User requested: "保持原始音频质量，不进行压缩或转码" (Keep original quality, no compression/transcoding)
        # However, simply copying (-acodec copy) might result in an m4a/aac file inside a .wav container if we force .wav extension, which is bad.
        # Or if we output to .mp3, we can't "copy" if source is aac.
        # Let's check the request again: "输出音频格式应为WAV或MP3".
        # If we want WAV, we usually need to decode to PCM.
        # If we want MP3, we need to re-encode if source is not MP3.
        # But user said "不进行压缩或转码" (no transcoding). This is contradictory if container changes.
        # "Keep original quality" -> usually implies -acodec copy.
        # But "Output WAV or MP3" -> implies format change.
        # Safe bet: Transcode to WAV (lossless) to satisfy "WAV" requirement and "Quality" (no lossy compression).
        # So we will use PCM WAV.
        str(output_path)
    ]

    # Wait, if I use -acodec copy, I should match the extension of the source audio stream.
    # But the user explicitly asked for "WAV or MP3".
    # Let's use WAV (pcm_s16le) as it preserves quality better than MP3 and is standard for processing.

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",  # Transcode to uncompressed PCM WAV
        str(output_path)
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Audio extraction successful.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        return False


def main():
    # Paths
    base_dir = Path("/home/project/RAG-Anything/tests/resource")
    video_file = base_dir / "project_1.mp4"
    # Output file name same as video but with .wav extension
    audio_output = base_dir / "project_1.wav"

    if not video_file.exists():
        logger.error(f"Video file not found: {video_file}")
        sys.exit(1)

    # 1. Audio Extraction
    logger.info("=== Step 1: Audio Extraction ===")
    if not extract_audio(video_file, audio_output):
        logger.error("Extraction failed. Aborting.")
        sys.exit(1)

    if not audio_output.exists():
        logger.error("Output file not found after extraction.")
        sys.exit(1)

    # 2. Audio Analysis
    logger.info("=== Step 2: Audio Analysis ===")
    parser = AudioParser()

    try:
        # Analyze
        analysis_result = parser.analyze_audio(audio_output)

        # Log detailed results
        logger.info("Analysis Result:")
        logger.info(json.dumps(analysis_result, indent=2))

        # 3. Verification
        logger.info("=== Step 3: Output Verification ===")

        # Verify file integrity (size > 0)
        if analysis_result["file_info"]["size_bytes"] == 0:
            logger.error("Extracted audio file is empty!")
            sys.exit(1)

        # Verify metadata reasonable values
        duration = analysis_result["metadata"]["duration_seconds"]
        if duration <= 0:
            logger.error(f"Invalid duration: {duration}")
            sys.exit(1)

        logger.info(f"Verified Duration: {duration:.2f}s")
        logger.info(f"Verified Channels: {analysis_result['metadata']['channels']}")
        logger.info(f"Verified Sample Rate: {analysis_result['metadata']['sample_rate']}")

        # 4. Transcription (User Request)
        logger.info("=== Step 4: Transcription ===")
        logger.info("Starting transcription (this may take a moment)...")

        # Set logger level to DEBUG to ensure we capture detailed model output if needed
        # But we will explicitly log the result

        transcription_results = parser.parse_audio(audio_output, lang="auto")

        if transcription_results and len(transcription_results) > 0:
            text_content = transcription_results[0].get("text", "")
            logger.info("Transcription Result:")
            logger.info("-" * 40)
            logger.info(text_content)
            logger.info("-" * 40)

            # Also log as debug as requested
            logger.debug(f"Full Transcription Output: {text_content}")
        else:
            logger.warning("No transcription result returned.")

        logger.info("=== Process Completed Successfully ===")

    except Exception as e:
        logger.error(f"Analysis/Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
