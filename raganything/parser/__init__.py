# Copyright (c) 2025 Kirky.X
# All rights reserved.

"""
Document Parser Package

This package provides functionality for parsing various document formats (PDF, images, office documents, audio, video, etc.)
and converting them into structured data (markdown and JSON).
"""

from .audio_parser import AudioParser
from .base_parser import Parser
from .docling_parser import DoclingParser
from .mineru_parser import MineruExecutionError, MineruParser
from .video_parser import VideoParser
from .vlm_parser import VlmParser
from raganything.i18n import _

__all__ = [
    "Parser",
    "MineruParser",
    "MineruExecutionError",
    "DoclingParser",
    "AudioParser",
    "VideoParser",
    "VlmParser",
]
