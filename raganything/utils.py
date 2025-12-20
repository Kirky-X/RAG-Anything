"""
Utility functions for RAGAnything

Contains helper functions for content separation, text insertion, and other utilities
"""

import base64
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from raganything.i18n_logger import get_i18n_logger
from raganything.i18n import _

T = TypeVar("T")


def get_env_value(key: str, default: T, expected_type: Type[T]) -> T:
    """Get environment variable with type conversion"""
    value = os.getenv(key)
    if value is None:
        return default

    try:
        if expected_type is bool:
            return value.lower() in ("true", "1", "yes", "on")  # type: ignore
        return expected_type(value)  # type: ignore
    except Exception:
        return default


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute MD5 hash ID for content

    Args:
        content: The content to hash
        prefix: Optional prefix for the ID

    Returns:
        str: MD5 hash ID with prefix
    """
    return prefix + hashlib.md5(content.encode("utf-8")).hexdigest()


def separate_content(
    content_list: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Separate text content and multimodal content

    Args:
        content_list: Content list from MinerU parsing

    Returns:
        (text_content, multimodal_items): Pure text content and multimodal items list
    """
    text_parts = []
    multimodal_items = []

    for item in content_list:
        content_type = item.get("type", "text")

        if content_type == "text":
            # Text content
            text = item.get("text", "")
            if text.strip():
                text_parts.append(text)
        else:
            # Multimodal content (image, table, equation, etc.)
            multimodal_items.append(item)

    # Merge all text content
    text_content = "\n\n".join(text_parts)

    logger.info("Content separation complete:")
    logger.info(_("  - Text content length: {} characters").format(len(text_content)))
    logger.info(_("  - Multimodal items count: {}").format(len(multimodal_items)))

    # Count multimodal types
    modal_types = {}
    for item in multimodal_items:
        modal_type = item.get("type", "unknown")
        modal_types[modal_type] = modal_types.get(modal_type, 0) + 1

    if modal_types:
        logger.info(_("  - Multimodal type distribution: {}").format(modal_types))

    return text_content, multimodal_items


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        str: Base64 encoded string, empty string if encoding fails
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        logger.error(_("Failed to encode image {}: {}").format(image_path, e))
        return ""


def validate_image_file(image_path: str, max_size_mb: int = 50) -> bool:
    """
    Validate if a file is a valid image file

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum file size in MB

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        path = Path(image_path)

        logger.debug(_("Validating image path: {}").format(image_path))
        logger.debug(_("Resolved path object: {}").format(path))
        logger.debug(_("Path exists check: {}").format(path.exists()))

        # Check if file exists
        if not path.exists():
            logger.warning(_("Image file not found: {}").format(image_path))
            return False

        # Check file extension
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        ]

        path_lower = str(path).lower()
        has_valid_extension = any(path_lower.endswith(ext) for ext in image_extensions)
        logger.debug(
            f"File extension check - path: {path_lower}, valid: {has_valid_extension}"
        )

        if not has_valid_extension:
            logger.warning(_("File does not appear to be an image: {}").format(image_path))
            return False

        # Check file size
        file_size = path.stat().st_size
        max_size = max_size_mb * 1024 * 1024
        logger.debug(
            f"File size check - size: {file_size} bytes, max: {max_size} bytes"
        )

        if file_size > max_size:
            logger.warning(_("Image file too large ({} bytes): {}").format(file_size, image_path))
            return False

        logger.debug(_("Image validation successful: {}").format(image_path))
        return True

    except Exception as e:
        logger.error(_("Error validating image file {}: {}").format(image_path, e))
        return False


async def insert_text_content(
    lightrag,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    wait_for_processing: bool = True,
    max_wait_time: int = 60,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        wait_for_processing: if True, wait for document processing to complete before returning
        max_wait_time: maximum time to wait for processing completion (in seconds)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    import asyncio
    import time

    from raganything.base import DocStatus

    insert_func = getattr(lightrag, "ainsert", None)
    if insert_func is None:
        return None

    # Determine document IDs for tracking
    doc_ids = ids if ids else []
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]

    kwargs = {
        "input": input,
        "file_paths": file_paths,
        "split_by_character": split_by_character,
        "split_by_character_only": split_by_character_only,
        "ids": ids,
    }

    if asyncio.iscoroutinefunction(insert_func):
        track_id = await insert_func(**kwargs)
    else:
        loop = asyncio.get_running_loop()
        track_id = await loop.run_in_executor(None, lambda: insert_func(**kwargs))

    logger.info(_("Text content insertion initiated with track_id: {}").format(track_id))

    if wait_for_processing and doc_ids:
        logger.info(_("Waiting for document processing to complete for IDs: {}").format(doc_ids))
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # Check if all documents are processed
            all_processed = True
            for doc_id in doc_ids:
                try:
                    doc_status = await lightrag.doc_status.get_by_id(doc_id)
                    if not doc_status:
                        all_processed = False
                        break

                    status = doc_status.get("status", DocStatus.PENDING)
                    if status in [DocStatus.PENDING, DocStatus.PROCESSING]:
                        all_processed = False
                        logger.info(_("Document {} status: {}").format(doc_id, status))
                        break
                    elif status == DocStatus.FAILED:
                        logger.error(_("Document {} processing failed").format(doc_id))
                        # Don't break here, check other documents
                    elif status == DocStatus.PROCESSED:
                        logger.info(_("Document {} processing completed").format(doc_id))

                except Exception as e:
                    logger.warning(_("Error checking status for document {}: {}").format(doc_id, e))
                    all_processed = False
                    break

            if all_processed:
                logger.info("All documents processed successfully")
                break

            # Wait before checking again
            await asyncio.sleep(2.0)
        else:
            logger.warning(_("Processing wait timeout after {} seconds").format(max_wait_time))

    logger.info("Text content insertion complete")
    return track_id


async def insert_text_content_with_multimodal_content(
    lightrag,
    input: str | list[str],
    multimodal_content: list[dict[str, any]] | None = None,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    scheme_name: str | None = None,
):
    """
    Insert pure text content into LightRAG

    Args:
        lightrag: LightRAG instance
        input: Single document string or list of document strings
        multimodal_content: Multimodal content list (optional)
        split_by_character: if split_by_character is not None, split the string by character, if chunk longer than
        chunk_token_size, it will be split again by token size.
        split_by_character_only: if split_by_character_only is True, split the string by character only, when
        split_by_character is None, this parameter is ignored.
        ids: single string of the document ID or list of unique document IDs, if not provided, MD5 hash IDs will be generated
        file_paths: single string of the file path or list of file paths, used for citation
        scheme_name: scheme name (optional)
    """
    logger.info("Starting text content insertion into LightRAG...")

    # Use LightRAG's insert method with all parameters
    try:
        await lightrag.ainsert(
            input=input,
            # multimodal_content=multimodal_content, # LightRAG's insert method might not support this yet, need to verify signature
            file_paths=file_paths,
            split_by_character=split_by_character,
            split_by_character_only=split_by_character_only,
            ids=ids,
            # scheme_name=scheme_name, # LightRAG's insert method might not support this yet
        )
        if multimodal_content:
            logger.warning(
                "Multimodal content passed to insert_text_content_with_multimodal_content but currently ignored in fallback insert call."
            )

    except Exception as e:
        logger.info(_("Error: {}").format(e))
        logger.info(
            "If the error is caused by the insert function not having a multimodal content parameter, please update the raganything branch of lightrag"
        )

    logger.info("Text content insertion complete")


def get_processor_for_type(modal_processors: Dict[str, Any], content_type: str):
    """
    Get appropriate processor based on content type

    Args:
        modal_processors: Dictionary of available processors
        content_type: Content type

    Returns:
        Corresponding processor instance
    """
    # Direct mapping to corresponding processor
    if content_type == "image":
        return modal_processors.get("image")
    elif content_type == "table":
        return modal_processors.get("table")
    elif content_type == "equation":
        return modal_processors.get("equation")
    elif content_type == "video":
        # Currently route video-derived items via generic processor;
        # visual frames are mapped to image type upstream.
        return modal_processors.get("generic")
    else:
        # For other types, use generic processor
        return modal_processors.get("generic")


def get_processor_supports(proc_type: str) -> List[str]:
    """Get processor supported features"""
    supports_map = {
        "image": [
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table": [
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation": [
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic": [
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
    }
    return supports_map.get(proc_type, ["Basic processing"])
