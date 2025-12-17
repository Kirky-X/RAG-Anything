# Copyright (c) 2025 Kirky.X
# All rights reserved.

"""
Document Parser Package

This package provides functionality for parsing various document formats (PDF, images, office documents, audio, video, etc.)
and converting them into structured data (markdown and JSON).
"""

from .base_parser import Parser
from .mineru_parser import MineruParser, MineruExecutionError
from .docling_parser import DoclingParser
from .audio_parser import AudioParser
from .video_parser import VideoParser
from .vlm_parser import VlmParser

__all__ = [
    "Parser",
    "MineruParser",
    "MineruExecutionError",
    "DoclingParser",
    "AudioParser",
    "VideoParser",
    "VlmParser",
]
