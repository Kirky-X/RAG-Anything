import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from raganything.models.device import device_manager
from raganything.parser.audio_parser import AudioParser

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_dummy_audio(duration_sec=5, sample_rate=16000):
    """Generate a dummy sine wave audio file."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate a 440 Hz sine wave
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio_data, sample_rate)
    temp_file.close()
    return Path(temp_file.name)


def run_benchmark_pass(
    parser: AudioParser, audio_path: Path, iterations: int = 5
) -> dict:
    """Run a single benchmark pass."""
    latencies = []

    # Warmup
    logger.info("Warming up model...")
    parser.parse_audio(audio_path)

    logger.info(f"Running {iterations} iterations...")
    for i in range(iterations):
        start_time = time.time()
        parser.parse_audio(audio_path)
        end_time = time.time()
        latencies.append(end_time - start_time)
        logger.info(f"Iteration {i + 1}: {latencies[-1]:.4f}s")

    return {
        "min": min(latencies),
        "max": max(latencies),
        "avg": sum(latencies) / len(latencies),
        "total": sum(latencies),
    }


def benchmark_audio_parser():
    print("\n" + "=" * 60)
    print("🎧 RAG-Anything Audio Parser Benchmark")
    print("=" * 60 + "\n")

    # 1. Setup & Environment Info
    device = device_manager.device
    cpu_usage = psutil.cpu_percent()
    memory = psutil.virtual_memory()

    print(f"Device: {device.upper()}")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"RAM Available: {memory.available / (1024 ** 3):.2f} GB")
    print("-" * 60)

    # 2. Generate Test Data
    audio_path = generate_dummy_audio(duration_sec=5)
    logger.info(f"Generated dummy audio (5s, 16kHz) at: {audio_path}")

    try:
        # 3. Initialize Parser
        logger.info("Initializing AudioParser...")
        start_init = time.time()
        parser = AudioParser()
        parser._load_model()  # Force load
        end_init = time.time()
        init_time = end_init - start_init
        logger.info(f"Model Initialization Time: {init_time:.4f}s")

        # 4. Run Benchmark
        results = run_benchmark_pass(parser, audio_path)

        # 5. Report
        print("\n" + "=" * 60)
        print("📊 Benchmark Results Summary")
        print("=" * 60)
        print(f"{'Metric':<20} | {'Value':<15}")
        print("-" * 40)
        print(f"{'Device':<20} | {device.upper()}")
        print(f"{'Init Time':<20} | {init_time:.4f}s")
        print(f"{'Avg Latency (5s)':<20} | {results['avg']:.4f}s")
        print(f"{'Min Latency':<20} | {results['min']:.4f}s")
        print(f"{'Max Latency':<20} | {results['max']:.4f}s")
        print(f"{'RTF (Real Time Factor)':<20} | {results['avg'] / 5.0:.4f}x")
        print("=" * 60 + "\n")

        if results["avg"] / 5.0 < 1.0:
            print("✅ Performance is faster than real-time!")
        else:
            print("⚠️ Performance is slower than real-time.")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if audio_path.exists():
            audio_path.unlink()


if __name__ == "__main__":
    benchmark_audio_parser()
