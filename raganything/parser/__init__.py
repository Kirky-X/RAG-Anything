"""
Document Parser Package

This package provides functionality for parsing various document formats (PDF, images, office documents, audio, video, etc.)
and converting them into structured data (markdown and JSON).
"""

from .base_parser import Parser
from .mineru_parser import MineruParser, MineruExecutionError
from .docling_parser import DoclingParser
from .audio_parser import AudioParser

__all__ = [
    "Parser",
    "MineruParser",
    "MineruExecutionError",
    "DoclingParser",
    "AudioParser",
]
