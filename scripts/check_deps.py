import sys
import traceback

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    print("Attempting to import pydub...")
    import pydub

    print(f"pydub imported successfully: {pydub.__file__}")
except ImportError:
    print("Failed to import pydub")
    traceback.print_exc()
except Exception:
    print("Error importing pydub")
    traceback.print_exc()

try:
    print("Attempting to import funasr...")
    import funasr

    print(f"funasr imported successfully: {funasr.__file__}")
except ImportError:
    print("Failed to import funasr")
    traceback.print_exc()
except Exception:
    print("Error importing funasr")
    traceback.print_exc()

try:
    from raganything.parser.audio_parser import AUDIO_DEPS_AVAILABLE

    print(f"AUDIO_DEPS_AVAILABLE: {AUDIO_DEPS_AVAILABLE}")
except Exception:
    print("Error importing AUDIO_DEPS_AVAILABLE")
    traceback.print_exc()
