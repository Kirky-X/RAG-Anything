# RAG-Anything Parsers & Models

This document describes the document parsing and model management capabilities of RAG-Anything.

## Document Parsers

RAG-Anything provides a robust parsing system for various file formats, including text, documents, images, and audio/video.

### Core Parsers

All parsers inherit from the base `Parser` class in `raganything.parser.base_parser`.

#### 1. MineruParser
- **Formats**: PDF, Images (png, jpg, etc.)
- **Functionality**: Uses MinerU (Magic-PDF) for high-quality layout analysis and OCR.
- **Features**:
  - Layout preservation
  - Formula and table extraction
  - Parallel processing support

#### 2. DoclingParser
- **Formats**: Office Documents (docx, pptx, xlsx), HTML
- **Functionality**: Uses Docling for structural parsing of Office documents.
- **Features**:
  - Native Office format parsing
  - HTML content extraction

#### 3. AudioParser
- **Formats**: Audio/Video (wav, mp3, mp4, m4a, flac, etc.)
- **Functionality**: Uses `iic/SenseVoiceSmall` for automatic speech recognition (ASR).
- **Features**:
  - Automatic format conversion to 16kHz WAV
  - Multilingual support (auto, zh, en, ja, ko, yue)
  - GPU acceleration support

## Model Management

RAG-Anything includes a dedicated `models` module for managing AI models and compute resources.

### Model Manager (`raganything.models.manager`)

Handles model downloading, caching, and loading using `modelscope`.

```python
from raganything.models import model_manager, default_models_config

# Download/Get path for SenseVoiceSmall
model_path = model_manager.get_sense_voice_model_path()
```

### Device Manager (`raganything.models.device`)

Automatically detects and manages compute resources (GPU/CPU).

- **Priority**: CUDA > MPS (Apple Silicon) > CPU
- **Features**:
  - Singleton instance for global resource management
  - Automatic fallback to CPU if GPU fails
  - Resource usage logging (RAM, GPU Memory)

```python
from raganything.models import device_manager

# Get current device
device = device_manager.device  # 'cuda', 'mps', or 'cpu'

# Check if GPU is available
if device_manager.is_gpu_available():
    print("Using GPU acceleration")
```

### Configuration (`raganything.models.config`)

Centralized configuration for model IDs and task types.

```python
from raganything.models import default_models_config

# Access model info
print(default_models_config.sense_voice_small.model_id)  # iic/SenseVoiceSmall
```

## Installation

To use the new audio parsing features, install the audio extras:

```bash
uv sync --extra audio
# or
pip install raganything[audio]
```
