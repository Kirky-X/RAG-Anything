"""
Document processing functionality for RAGAnything

Contains methods for parsing documents and processing multimodal content
"""

import os
import time
import hashlib
import json
import uuid
import aiofiles
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from raganything.base import DocStatus
from raganything.parser import MineruParser, DoclingParser, MineruExecutionError
from raganything.parser import VideoParser
from raganything.utils import (
    separate_content,
    insert_text_content,
    insert_text_content_with_multimodal_content,
    get_processor_for_type,
)
import asyncio
from lightrag.utils import compute_mdhash_id


class ProcessorMixin:
    """ProcessorMixin class containing document processing functionality for RAGAnything"""

    async def _save_upload_file(
            self, file: Any, doc_id: str | None = None
        ) -> tuple[str, str]:
            """
            Save uploaded file to disk
            
            Args:
                file: Uploaded file object
                doc_id: Optional document ID
                
            Returns:
                tuple[str, str]: (file_path, doc_id)
            """
            self.logger.info(f"Starting to save uploaded file: {file.filename}")
            if doc_id is None:
                doc_id = f"doc-{uuid.uuid4()}"
                self.logger.info(f"Generated new doc_id: {doc_id}")

            file_name = file.filename

            working_dir = getattr(self, "working_dir", None)
            if not working_dir and getattr(self, "config", None) is not None:
                working_dir = getattr(self.config, "working_dir", None)
            if not working_dir:
                working_dir = "./rag_storage"

            working_dir = os.path.abspath(working_dir)
            self.logger.info(f"Resolved working directory for upload: {working_dir}")

            try:
                os.makedirs(working_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(
                    f"Failed to create working directory {working_dir}: {e}",
                    exc_info=True,
                )
                raise

            self.working_dir = working_dir

            file_path = os.path.join(working_dir, file_name)
            self.logger.info(f"Target file path for upload: {file_path}")

            # Save file to disk
            try:
                self.logger.info("Opening file for writing...")
                async with aiofiles.open(file_path, "wb") as out_file:
                    self.logger.info("Starting to read file chunks...")
                    chunk_index = 0
                    while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                        self.logger.info(
                            f"Read chunk {chunk_index} of size {len(content)}"
                        )
                        await out_file.write(content)
                        self.logger.info(f"Wrote chunk {chunk_index}")
                        chunk_index += 1
                self.logger.info("Finished writing all file chunks.")
            except FileNotFoundError:
                fallback_dir = "/tmp/rag_anything_uploads"
                self.logger.warning(
                    f"Primary upload path not available: {file_path}, "
                    f"falling back to {fallback_dir}"
                )
                try:
                    os.makedirs(fallback_dir, exist_ok=True)
                    fallback_path = os.path.join(fallback_dir, file_name)
                    async with aiofiles.open(fallback_path, "wb") as out_file:
                        self.logger.info(
                            "Starting to read file chunks (fallback path)..."
                        )
                        chunk_index = 0
                        while content := await file.read(1024 * 1024):
                            self.logger.info(
                                f"Read chunk {chunk_index} of size {len(content)}"
                            )
                            await out_file.write(content)
                            self.logger.info(f"Wrote chunk {chunk_index}")
                            chunk_index += 1
                    self.logger.info(
                        f"Successfully saved uploaded file to fallback path: {fallback_path}"
                    )
                    file_path = fallback_path
                except Exception as e:
                    self.logger.error(
                        f"Error saving file to fallback path {fallback_dir}: {e}",
                        exc_info=True,
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Error saving file {file_path}: {e}", exc_info=True)
                raise

            self.logger.info(f"Successfully saved uploaded file to: {file_path}")
            return file_path, doc_id
    def _generate_cache_key(
        self, file_path: Path, parse_method: str = None, **kwargs
    ) -> str:
        """
        Generate cache key based on file path and parsing configuration

        Args:
            file_path: Path to the file
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            str: Cache key for the file and configuration
        """

        # Get file modification time
        mtime = file_path.stat().st_mtime

        # Create configuration dict for cache key
        config_dict = {
            "file_path": str(file_path.absolute()),
            "mtime": mtime,
            "parser": self.config.parser,
            "parse_method": parse_method or self.config.parse_method,
        }

        # Add relevant kwargs to config
        relevant_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "lang",
                "device",
                "start_page",
                "end_page",
                "formula",
                "table",
                "backend",
                "source",
            ]
        }
        config_dict.update(relevant_kwargs)

        # Generate hash from config
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()

        return cache_key

    def _generate_content_based_doc_id(self, content_list: List[Dict[str, Any]]) -> str:
        """
        Generate doc_id based on document content

        Args:
            content_list: Parsed content list

        Returns:
            str: Content-based document ID with doc- prefix
        """
        from lightrag.utils import compute_mdhash_id

        # Extract key content for ID generation
        content_hash_data = []

        for item in content_list:
            if isinstance(item, dict):
                # For text content, use the text
                if item.get("type") == "text" and item.get("text"):
                    content_hash_data.append(item["text"].strip())
                # For other content types, use key identifiers
                elif item.get("type") == "image" and item.get("img_path"):
                    content_hash_data.append(f"image:{item['img_path']}")
                elif item.get("type") == "table" and item.get("table_body"):
                    content_hash_data.append(f"table:{item['table_body']}")
                elif item.get("type") == "equation" and item.get("text"):
                    content_hash_data.append(f"equation:{item['text']}")
                else:
                    # For other types, use string representation
                    content_hash_data.append(str(item))

        # Create a content signature
        content_signature = "\n".join(content_hash_data)

        # Generate doc_id from content
        doc_id = compute_mdhash_id(content_signature, prefix="doc-")

        return doc_id

    async def _get_cached_result(
        self, cache_key: str, file_path: Path, parse_method: str = None, **kwargs
    ) -> tuple[List[Dict[str, Any]], str] | None:
        """
        Get cached parsing result if available and valid

        Args:
            cache_key: Cache key to look up
            file_path: Path to the file for mtime check
            parse_method: Parse method used
            **kwargs: Additional parser parameters

        Returns:
            tuple[List[Dict[str, Any]], str] | None: (content_list, doc_id) or None if not found/invalid
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return None

        try:
            cached_data = await self.parse_cache.get_by_id(cache_key)
            if not cached_data:
                return None

            # Check file modification time
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cached_data.get("mtime", 0)

            if current_mtime != cached_mtime:
                self.logger.debug(f"Cache invalid - file modified: {cache_key}")
                return None

            # Check parsing configuration
            cached_config = cached_data.get("parse_config", {})
            current_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to current config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            current_config.update(relevant_kwargs)

            if cached_config != current_config:
                self.logger.debug(f"Cache invalid - config changed: {cache_key}")
                return None

            content_list = cached_data.get("content_list", [])
            doc_id = cached_data.get("doc_id")

            if content_list and doc_id:
                self.logger.debug(
                    f"Found valid cached parsing result for key: {cache_key}"
                )
                return content_list, doc_id
            else:
                self.logger.debug(
                    f"Cache incomplete - missing content or doc_id: {cache_key}"
                )
                return None

        except Exception as e:
            self.logger.warning(f"Error accessing parse cache: {e}")

        return None

    async def _store_cached_result(
        self,
        cache_key: str,
        content_list: List[Dict[str, Any]],
        doc_id: str,
        file_path: Path,
        parse_method: str = None,
        **kwargs,
    ) -> None:
        """
        Store parsing result in cache

        Args:
            cache_key: Cache key to store under
            content_list: Content list to cache
            doc_id: Content-based document ID
            file_path: Path to the file for mtime storage
            parse_method: Parse method used
            **kwargs: Additional parser parameters
        """
        if not hasattr(self, "parse_cache") or self.parse_cache is None:
            return

        try:
            # Get file modification time
            file_mtime = file_path.stat().st_mtime

            # Create parsing configuration
            parse_config = {
                "parser": self.config.parser,
                "parse_method": parse_method or self.config.parse_method,
            }

            # Add relevant kwargs to config
            relevant_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                in [
                    "lang",
                    "device",
                    "start_page",
                    "end_page",
                    "formula",
                    "table",
                    "backend",
                    "source",
                ]
            }
            parse_config.update(relevant_kwargs)

            cache_data = {
                cache_key: {
                    "content_list": content_list,
                    "doc_id": doc_id,
                    "mtime": file_mtime,
                    "parse_config": parse_config,
                    "cached_at": time.time(),
                    "cache_version": "1.0",
                }
            }
            await self.parse_cache.upsert(cache_data)
            # Ensure data is persisted to disk
            await self.parse_cache.index_done_callback()
            self.logger.info(f"Stored parsing result in cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error storing to parse cache: {e}")

    async def parse_document(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        **kwargs,
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Parse document with caching support

        Args:
            file_path: Path to the file to parse
            output_dir: Output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)

        Returns:
            tuple[List[Dict[str, Any]], str]: (content_list, doc_id)
        """
        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting document parsing: {file_path}")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate cache key based on file and configuration
        cache_key = self._generate_cache_key(file_path, parse_method, **kwargs)

        # Check cache first
        if not kwargs.get("force_parse", False):
            cached_result = await self._get_cached_result(
                cache_key, file_path, parse_method, **kwargs
            )
            if cached_result is not None:
                content_list, doc_id = cached_result
                self.logger.info(f"Using cached parsing result for: {file_path}")
                if display_stats:
                    self.logger.info(
                        f"* Total blocks in cached content_list: {len(content_list)}"
                    )
                return content_list, doc_id

        # Choose appropriate parsing method based on file extension
        ext = file_path.suffix.lower()

        try:
            doc_parser = (
                DoclingParser() if self.config.parser == "docling" else MineruParser()
            )

            # Log parser and method information
            self.logger.info(
                f"Using {self.config.parser} parser with method: {parse_method}"
            )

            if ext in [".pdf"]:
                self.logger.info("Detected PDF file, using parser for PDF...")
                content_list = await asyncio.to_thread(
                    doc_parser.parse_pdf,
                    pdf_path=file_path,
                    output_dir=output_dir,
                    method=parse_method,
                    **kwargs,
                )
            elif ext in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tiff",
                ".tif",
                ".gif",
                ".webp",
            ]:
                self.logger.info("Detected image file, using parser for images...")
                # Use the selected parser's image parsing capability
                if hasattr(doc_parser, "parse_image"):
                    content_list = await asyncio.to_thread(
                        doc_parser.parse_image,
                        image_path=file_path,
                        output_dir=output_dir,
                        **kwargs,
                    )
                else:
                    # Fallback to MinerU for image parsing if current parser doesn't support it
                    self.logger.warning(
                        f"{self.config.parser} parser doesn't support image parsing, falling back to MinerU"
                    )
                    content_list = MineruParser().parse_image(
                        image_path=file_path, output_dir=output_dir, **kwargs
                    )
            elif ext in [
                ".doc",
                ".docx",
                ".ppt",
                ".pptx",
                ".xls",
                ".xlsx",
                ".html",
                ".htm",
                ".xhtml",
            ]:
                self.logger.info(
                    "Detected Office or HTML document, using parser for Office/HTML..."
                )
                content_list = await asyncio.to_thread(
                    doc_parser.parse_office_doc,
                    doc_path=file_path,
                    output_dir=output_dir,
                    **kwargs,
                )
            elif ext in [".txt", ".md"]:
                self.logger.info(f"Detected text/markdown file: {ext}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    content_list = [{"type": "text", "text": text_content, "page_idx": 0}]
                except Exception as e:
                    self.logger.error(f"Error reading text file {file_path}: {e}")
                    raise e
            elif ext in [
                ".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".mpeg"
            ]:
                self.logger.info("Detected video file, using VideoParser for AV + visual timeline...")
                video_parser = VideoParser()
                self.logger.info("Calling video_parser.parse_video")
                video_result = await asyncio.to_thread(
                    video_parser.parse_video,
                    file_path=str(file_path),
                    output_dir=output_dir,
                    fps=kwargs.get("video_fps", 0.5),
                    cleanup_frames=kwargs.get("cleanup_frames", False),
                )
                self.logger.info("video_parser.parse_video returned")
                # Convert video timeline to content_list for multimodal processing
                timeline = video_result.get("content", [])
                content_list = []
                def _fmt_ts(v: float) -> str:
                    m = int(v // 60)
                    s = int(v % 60)
                    return f"{m:02d}:{s:02d}"

                if timeline:
                    for item in timeline:
                        if item.get("type") == "audio":
                            # audio content now may include fine-grained segments
                            text = item.get("content", "")
                            start = float(item.get("timestamp", 0.0))
                            end = float(item.get("end", start))
                            speaker = item.get("speaker")
                            segments = item.get("segments")

                            if isinstance(segments, list) and segments:
                                for seg in segments:
                                    seg_text = seg.get("text", "")
                                    seg_start = float(seg.get("start", start))
                                    seg_end = float(seg.get("end", seg_start))
                                    seg_speaker = seg.get("speaker", speaker)
                                    if seg_text:
                                        content_list.append(
                                            {
                                                "text": f"[{_fmt_ts(seg_start)}-{_fmt_ts(seg_end)}] "
                                                + (
                                                    f"[{seg_speaker}] "
                                                    if seg_speaker
                                                    else ""
                                                )
                                                + seg_text,
                                                "page_idx": 0,
                                                "source_type": "video_audio",
                                                "start": seg_start,
                                                "end": seg_end,
                                                "speaker": seg_speaker,
                                            }
                                        )
                            elif text:
                                content_list.append(
                                    {
                                        "text": f"[{_fmt_ts(start)}-{_fmt_ts(end)}] "
                                        + (f"[{speaker}] " if speaker else "")
                                        + text,
                                        "page_idx": 0,
                                        "source_type": "video_audio",
                                        "start": start,
                                        "end": end,
                                        "speaker": speaker,
                                    }
                                )
                        elif item.get("type") == "image":
                            # Handle visual content (frames)
                            img_path = item.get("path")
                            timestamp = float(item.get("timestamp", 0.0))
                            if img_path:
                                content_list.append({
                                    "type": "image",
                                    "img_path": img_path,
                                    "page_idx": 0,
                                    "source_type": "video_frame",
                                    "timestamp": timestamp,
                                    "text": f"Frame at {_fmt_ts(timestamp)}"
                                })
                        elif item.get("type") == "visual":
                            desc = item.get("content", "")
                            frame_path = item.get("frame_path", "")
                            # Map to image multimodal item to reuse image processors
                            content_list.append({
                                "type": "image",
                                "img_path": frame_path,
                                "image_caption": [desc] if desc else [],
                                "image_footnote": [],
                                "page_idx": 0
                            })

        except MineruExecutionError as e:
            self.logger.error(f"Mineru command failed: {e}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error during parsing with {self.config.parser} parser: {str(e)}"
            )
            raise e

        msg = f"Parsing {file_path} complete! Extracted {len(content_list)} content blocks"
        self.logger.info(msg)

        if len(content_list) == 0:
            raise ValueError("Parsing failed: No content was extracted")

        # Generate doc_id based on content
        doc_id = self._generate_content_based_doc_id(content_list)

        # Store result in cache
        await self._store_cached_result(
            cache_key, content_list, doc_id, file_path, parse_method, **kwargs
        )

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        return content_list, doc_id

    async def _process_multimodal_content(
        self,
        multimodal_items: List[Dict[str, Any]],
        file_path: str,
        doc_id: str,
        pipeline_status: Optional[Any] = None,
        pipeline_status_lock: Optional[Any] = None,
    ):
        """
        Process multimodal content (using specialized processors)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
            pipeline_status: Pipeline status object
            pipeline_status_lock: Pipeline status lock
        """
        self.logger.info(f"Starting _process_multimodal_content for {doc_id} with {len(multimodal_items)} items")

        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Check multimodal processing status - handle LightRAG's early DocStatus.PROCESSED marking
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if existing_doc_status:
                # Check if multimodal content is already processed
                multimodal_processed = existing_doc_status.get(
                    "multimodal_processed", False
                )

                if multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} multimodal content is already processed"
                    )
                    return

                # Even if status is DocStatus.PROCESSED (text processing done),
                # we still need to process multimodal content if not yet done
                doc_status = existing_doc_status.get("status", "")
                if doc_status == DocStatus.PROCESSED and not multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} text processing is complete, but multimodal content still needs processing"
                    )
                    # Continue with multimodal processing
                elif doc_status == DocStatus.PROCESSED and multimodal_processed:
                    self.logger.info(
                        f"Document {doc_id} is fully processed (text + multimodal)"
                    )
                    return

        except Exception as e:
            self.logger.debug(f"Error checking document status for {doc_id}: {e}")
            # Continue with processing if cache check fails

        # Use ProcessorMixin's own batch processing that can handle multiple content types
        log_message = "Starting multimodal content processing..."
        self.logger.info(log_message)
        if pipeline_status_lock and pipeline_status:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        if hasattr(self.lightrag, "chunks_vdb") and hasattr(self.lightrag.chunks_vdb, "embedding_func"):
            self.logger.debug(f"Processor LightRAG chunks_vdb embedding_func type: {type(self.lightrag.chunks_vdb.embedding_func)}")

        try:
            # Ensure LightRAG is initialized
            await self._ensure_lightrag_initialized()

            self.logger.info(f"Calling _process_multimodal_content_batch_type_aware for {doc_id}")
            await self._process_multimodal_content_batch_type_aware(
                multimodal_items=multimodal_items, file_path=file_path, doc_id=doc_id
            )
            self.logger.info(f"Returned from _process_multimodal_content_batch_type_aware for {doc_id}")

            # Mark multimodal content as processed and update final status
            await self._mark_multimodal_processing_complete(doc_id)

            log_message = "Multimodal content processing complete"
            self.logger.info(log_message)
            if pipeline_status_lock and pipeline_status:
                async with pipeline_status_lock:
                    pipeline_status["latest_message"] = log_message
                    pipeline_status["history_messages"].append(log_message)

        except Exception as e:
            self.logger.error(f"Error in multimodal processing: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            # Fallback to individual processing if batch processing fails
            self.logger.warning("Falling back to individual multimodal processing")
            try:
                await self._process_multimodal_content_individual(
                    multimodal_items, file_path, doc_id
                )
            except Exception as individual_err:
                self.logger.error(f"Individual processing also failed: {individual_err}")
                import traceback
                self.logger.error(traceback.format_exc())

            # Mark multimodal content as processed even after fallback
            # We must ensure this is called even if individual processing fails
            # otherwise the document will be stuck in 'handling' state forever
            try:
                await self._mark_multimodal_processing_complete(doc_id)
            except Exception as mark_err:
                self.logger.error(f"Failed to mark multimodal processing as complete: {mark_err}")

    async def _process_multimodal_content_individual(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Process multimodal content individually (fallback method)

        Args:
            multimodal_items: List of multimodal items
            file_path: File path (for reference)
            doc_id: Document ID for proper chunk association
        """
        file_name = os.path.basename(file_path)

        # Collect all chunk results for batch processing (similar to text content processing)
        all_chunk_results = []
        multimodal_chunk_ids = []

        # Get current text chunks count to set proper order indexes for multimodal chunks
        existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
        existing_chunks_count = (
            existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
        )

        for i, item in enumerate(multimodal_items):
            try:
                content_type = item.get("type", "unknown")
                self.logger.info(
                    f"Processing item {i+1}/{len(multimodal_items)}: {content_type} content"
                )

                # Select appropriate processor
                processor = get_processor_for_type(self.modal_processors, content_type)

                if processor:
                    # Prepare item info for context extraction
                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": i,
                        "type": content_type,
                    }

                    # Process content and get chunk results instead of immediately merging
                    process_result = await processor.process_multimodal_content(
                        modal_content=item,
                        content_type=content_type,
                        file_path=file_name,
                        item_info=item_info,  # Pass item info for context extraction
                        batch_mode=True,
                        doc_id=doc_id,  # Pass doc_id for proper association
                        chunk_order_index=existing_chunks_count
                        + i,  # Proper order index
                    )
                    
                    # Unpack result based on length to handle both 3-value and 4-value returns
                    if len(process_result) == 3:
                        enhanced_caption, entity_info, chunk_results = process_result
                    elif len(process_result) == 4:
                        enhanced_caption, entity_info, chunk_results, _ = process_result
                    else:
                        raise ValueError(f"Unexpected return value count from process_multimodal_content: {len(process_result)}")

                    # Collect chunk results for batch processing
                    all_chunk_results.extend(chunk_results)

                    # Extract chunk ID from the entity_info (actual chunk_id created by processor)
                    if entity_info and "chunk_id" in entity_info:
                        chunk_id = entity_info["chunk_id"]
                        multimodal_chunk_ids.append(chunk_id)

                    self.logger.info(
                        f"{content_type} processing complete: {entity_info.get('entity_name', 'Unknown')}"
                    )
                else:
                    self.logger.warning(
                        f"No suitable processor found for {content_type} type content"
                    )

            except Exception as e:
                self.logger.error(f"Error processing multimodal content: {str(e)}")
                self.logger.debug("Exception details:", exc_info=True)
                continue

        # Update doc_status to include multimodal chunks in the standard chunks_list
        if multimodal_chunk_ids:
            try:
                # Get current document status
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

                if current_doc_status:
                    existing_chunks_list = current_doc_status.get("chunks_list", [])
                    existing_chunks_count = current_doc_status.get("chunks_count", 0)

                    # Add multimodal chunks to the standard chunks_list
                    updated_chunks_list = existing_chunks_list + multimodal_chunk_ids
                    updated_chunks_count = existing_chunks_count + len(
                        multimodal_chunk_ids
                    )

                    # Update document status with integrated chunk list
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,  # Keep existing fields
                                "chunks_list": updated_chunks_list,  # Integrated chunks list
                                "chunks_count": updated_chunks_count,  # Updated total count
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )

                    # Ensure doc_status update is persisted to disk
                    await self.lightrag.doc_status.index_done_callback()

                    self.logger.info(
                        f"Updated doc_status with {len(multimodal_chunk_ids)} multimodal chunks integrated into chunks_list"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error updating doc_status with multimodal chunks: {e}"
                )

        # Batch merge all multimodal content results (similar to text content processing)
        if all_chunk_results:
            from lightrag.operate import merge_nodes_and_edges
            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            # Get pipeline status and lock from shared storage
            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            await merge_nodes_and_edges(
                chunk_results=all_chunk_results,
                knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
                entity_vdb=self.lightrag.entities_vdb,
                relationships_vdb=self.lightrag.relationships_vdb,
                global_config=self.lightrag.__dict__,
                full_entities_storage=self.lightrag.full_entities,
                full_relations_storage=self.lightrag.full_relations,
                doc_id=doc_id,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.lightrag.llm_response_cache,
                current_file_number=1,
                total_files=1,
                file_path=file_name,
            )

            await self.lightrag._insert_done()

        self.logger.info("Individual multimodal content processing complete")

        # Mark multimodal content as processed
        await self._mark_multimodal_processing_complete(doc_id)

    async def _process_multimodal_content_batch_type_aware(
        self, multimodal_items: List[Dict[str, Any]], file_path: str, doc_id: str
    ):
        """
        Type-aware batch processing that selects correct processors based on content type.
        This is the corrected implementation that handles different modality types properly.

        Args:
            multimodal_items: List of multimodal items with different types
            file_path: File path for citation
            doc_id: Document ID for proper association
        """
        self.logger.info(f"Entered _process_multimodal_content_batch_type_aware for {doc_id}")
        if not multimodal_items:
            self.logger.debug("No multimodal content to process")
            return

        # Get existing chunks count for proper order indexing
        try:
            existing_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            existing_chunks_count = (
                existing_doc_status.get("chunks_count", 0) if existing_doc_status else 0
            )
            self.logger.info(f"Existing chunks count: {existing_chunks_count}")
        except Exception as e:
            self.logger.warning(f"Error getting existing chunks count: {e}")
            existing_chunks_count = 0

        # Use LightRAG's concurrency control
        semaphore_value = getattr(self.lightrag, "max_parallel_insert", 2)
        self.logger.info(f"Using semaphore with value: {semaphore_value}")
        semaphore = asyncio.Semaphore(semaphore_value)

        # Progress tracking variables
        total_items = len(multimodal_items)
        completed_count = 0
        progress_lock = asyncio.Lock()

        # Log processing start
        self.logger.info(f"Starting to process {total_items} multimodal content items")

        # Stage 1: Concurrent generation of descriptions using correct processors for each type
        async def process_single_item_with_correct_processor(
            item: Dict[str, Any], index: int, file_path: str
        ):
            """Process single item using the correct processor for its type"""
            nonlocal completed_count
            self.logger.info(f"Waiting for semaphore to process item {index}")
            async with semaphore:
                try:
                    content_type = item.get("type", "unknown")
                    self.logger.info(f"Processing item {index} (type: {content_type})")

                    # Select the correct processor based on content type
                    processor = get_processor_for_type(
                        self.modal_processors, content_type
                    )

                    if not processor:
                        self.logger.warning(
                            f"No processor found for type: {content_type}"
                        )
                        return None

                    item_info = {
                        "page_idx": item.get("page_idx", 0),
                        "index": index,
                        "type": content_type,
                    }

                    self.logger.info(f"Calling generate_description_only for item {index} of type {content_type}")

                    # Call the correct processor's description generation method
                    (
                        description,
                        entity_info,
                    ) = await processor.generate_description_only(
                        modal_content=item,
                        content_type=content_type,
                        item_info=item_info,
                        entity_name=None,  # Let LLM auto-generate
                    )

                    self.logger.info(f"Finished processing item {index} of type {content_type}")

                    # Update progress (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    return {
                        "index": index,
                        "content_type": content_type,
                        "description": description,
                        "entity_info": entity_info,
                        "original_item": item,
                        "item_info": item_info,
                        "chunk_order_index": existing_chunks_count + index,
                        "processor": processor,  # Keep reference to the processor used
                        "file_path": file_path,  # Add file_path to the result
                    }

                except Exception as e:
                    # Update progress even on error (non-blocking)
                    async with progress_lock:
                        completed_count += 1
                        if (
                            completed_count % max(1, total_items // 10) == 0
                            or completed_count == total_items
                        ):
                            progress_percent = (completed_count / total_items) * 100
                            self.logger.info(
                                f"Multimodal chunk generation progress: {completed_count}/{total_items} ({progress_percent:.1f}%)"
                            )

                    self.logger.error(
                        f"Error generating description for {content_type} item {index}: {e}"
                    )
                    return None

        # Process all items concurrently with correct processors
        tasks = [
            asyncio.create_task(
                process_single_item_with_correct_processor(item, i, file_path)
            )
            for i, item in enumerate(multimodal_items)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        multimodal_data_list = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Task failed: {result}")
                continue
            if result is not None:
                multimodal_data_list.append(result)

        if not multimodal_data_list:
            self.logger.warning("No valid multimodal descriptions generated")
            return

        self.logger.info(
            f"Generated descriptions for {len(multimodal_data_list)}/{len(multimodal_items)} multimodal items using correct processors"
        )

        # Stage 2: Convert to LightRAG chunks format
        self.logger.info("Stage 2: Converting to LightRAG chunks format...")
        lightrag_chunks = self._convert_to_lightrag_chunks_type_aware(
            multimodal_data_list, file_path, doc_id
        )
        self.logger.info(f"Stage 2 complete. Generated {len(lightrag_chunks)} chunks.")

        # Stage 3: Store chunks to LightRAG storage
        self.logger.info("Stage 3: Storing chunks to LightRAG storage...")
        await self._store_chunks_to_lightrag_storage_type_aware(lightrag_chunks)
        self.logger.info("Stage 3 complete.")

        # Stage 3.5: Store multimodal main entities to entities_vdb and full_entities
        self.logger.info("Stage 3.5: Storing multimodal main entities...")
        await self._store_multimodal_main_entities(
            multimodal_data_list, lightrag_chunks, file_path, doc_id
        )
        self.logger.info("Stage 3.5 complete.")

        # Track chunk IDs for doc_status update
        chunk_ids = list(lightrag_chunks.keys())

        # Stage 4: Use LightRAG's batch entity relation extraction
        self.logger.info("Stage 4: Batch entity relation extraction (LightRAG style)...")
        chunk_results = await self._batch_extract_entities_lightrag_style_type_aware(
            lightrag_chunks
        )
        self.logger.info(f"Stage 4 complete. Extracted results for {len(chunk_results)} chunks.")
        self.logger.debug(f"Chunk results after Stage 4: {chunk_results}")

        # Stage 5: Add belongs_to relations (multimodal-specific)
        self.logger.info("Stage 5: Adding belongs_to relations...")
        enhanced_chunk_results = await self._batch_add_belongs_to_relations_type_aware(
            chunk_results, multimodal_data_list
        )
        self.logger.info("Stage 5 complete.")

        # Stage 6: Use LightRAG's batch merge
        self.logger.info("Stage 6: Batch merging LightRAG style...")
        await self._batch_merge_lightrag_style_type_aware(
            enhanced_chunk_results, file_path, doc_id
        )
        self.logger.info("Stage 6 complete.")

        # Stage 7: Update doc_status with integrated chunks_list
        await self._update_doc_status_with_chunks_type_aware(doc_id, chunk_ids)

    def _convert_to_lightrag_chunks_type_aware(
        self, multimodal_data_list: List[Dict[str, Any]], file_path: str, doc_id: str
    ) -> Dict[str, Any]:
        """Convert multimodal data to LightRAG standard chunks format"""

        chunks = {}

        for data in multimodal_data_list:
            description = data["description"]
            entity_info = data["entity_info"]
            chunk_order_index = data["chunk_order_index"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the appropriate chunk template based on content type
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Calculate tokens
            tokens = len(self.lightrag.tokenizer.encode(formatted_chunk_content))

            # Build LightRAG standard chunk format
            chunks[chunk_id] = {
                "content": formatted_chunk_content,  # Now uses the templated content
                "tokens": tokens,
                "full_doc_id": doc_id,
                "chunk_order_index": chunk_order_index,
                "file_path": os.path.basename(file_path),
                "llm_cache_list": [],  # LightRAG will populate this field
                # Multimodal-specific metadata
                "is_multimodal": True,
                "modal_entity_name": entity_info["entity_name"],
                "original_type": data["content_type"],
                "page_idx": data["item_info"].get("page_idx", 0),
            }

        self.logger.debug(
            f"Converted {len(chunks)} multimodal items to multimodal chunks format"
        )
        return chunks

    def _apply_chunk_template(
        self, content_type: str, original_item: Dict[str, Any], description: str
    ) -> str:
        """
        Apply the appropriate chunk template based on content type

        Args:
            content_type: Type of content (image, table, equation, generic)
            original_item: Original multimodal item data
            description: Enhanced description generated by the processor

        Returns:
            Formatted chunk content using the appropriate template
        """
        from raganything.prompt import PROMPTS

        try:
            if content_type == "image":
                image_path = original_item.get("img_path", "")
                captions = original_item.get(
                    "image_caption", original_item.get("img_caption", [])
                )
                footnotes = original_item.get(
                    "image_footnote", original_item.get("img_footnote", [])
                )

                return PROMPTS["image_chunk"].format(
                    image_path=image_path,
                    captions=", ".join(captions) if captions else "None",
                    footnotes=", ".join(footnotes) if footnotes else "None",
                    enhanced_caption=description,
                )

            elif content_type == "table":
                table_img_path = original_item.get("img_path", "")
                table_caption = original_item.get("table_caption", [])
                table_body = original_item.get("table_body", "")
                table_footnote = original_item.get("table_footnote", [])

                return PROMPTS["table_chunk"].format(
                    table_img_path=table_img_path,
                    table_caption=", ".join(table_caption) if table_caption else "None",
                    table_body=table_body,
                    table_footnote=", ".join(table_footnote)
                    if table_footnote
                    else "None",
                    enhanced_caption=description,
                )

            elif content_type == "equation":
                equation_text = original_item.get("text", "")
                equation_format = original_item.get("text_format", "")

                return PROMPTS["equation_chunk"].format(
                    equation_text=equation_text,
                    equation_format=equation_format,
                    enhanced_caption=description,
                )

            else:  # generic or unknown types
                content = str(original_item.get("content", original_item))

                return PROMPTS["generic_chunk"].format(
                    content_type=content_type.title(),
                    content=content,
                    enhanced_caption=description,
                )

        except Exception as e:
            self.logger.warning(
                f"Error applying chunk template for {content_type}: {e}"
            )
            # Fallback to just the description if template fails
            return description

    async def _store_chunks_to_lightrag_storage_type_aware(
        self, chunks: Dict[str, Any]
    ):
        """Store chunks to storage"""
        try:
            # Store in text_chunks storage (required for extract_entities)
            self.logger.info("DEBUG: Upserting to text_chunks...")
            await self.lightrag.text_chunks.upsert(chunks)
            self.logger.info("DEBUG: text_chunks upsert complete.")

            # Store in chunks vector database for retrieval
            self.logger.info("DEBUG: Upserting to chunks_vdb (this involves embedding generation)...")
            self.logger.info(f"DEBUG: Chunks to upsert: {len(chunks)}")
            if len(chunks) > 0:
                first_chunk_key = next(iter(chunks))
                self.logger.info(f"DEBUG: First chunk content length: {len(chunks[first_chunk_key]['content'])}")
                
                # Check for unhashable types in values
                first_val = chunks[first_chunk_key]
                self.logger.info(f"DEBUG: First chunk keys: {list(first_val.keys())}")
                for k, v in first_val.items():
                    self.logger.info(f"DEBUG: Key '{k}' has type {type(v)}")
                    if isinstance(v, list):
                        self.logger.warning(f"DEBUG: Key '{k}' is a list! Content: {v[:50] if len(v) > 50 else v}")

            # Inspect embedding_func before call
            if hasattr(self.lightrag.chunks_vdb, "embedding_func"):
                ef = self.lightrag.chunks_vdb.embedding_func
                self.logger.info(f"DEBUG: chunks_vdb.embedding_func type: {type(ef)}")
                self.logger.info(f"DEBUG: chunks_vdb.embedding_func repr: {repr(ef)}")
                if hasattr(ef, "func"):
                    self.logger.info(f"DEBUG: chunks_vdb.embedding_func.func type: {type(ef.func)}")
                    self.logger.info(f"DEBUG: chunks_vdb.embedding_func.func repr: {repr(ef.func)}")

            # Add timeout to upsert to detect hang
            # Increase timeout to 300s (5 minutes) for embedding generation
            self.logger.debug(f"Calling upsert for chunks: {list(chunks.keys())}")
            # Ensure embedding_func is called and logged inside LightRAG if possible
            # But we can't easily modify LightRAG internals here.
            # We rely on our wrapper logging.
            
            try:
                await asyncio.wait_for(self.lightrag.chunks_vdb.upsert(chunks), timeout=1800)
                self.logger.info("DEBUG: chunks_vdb upsert complete.")
            except asyncio.TimeoutError:
                self.logger.error("CRITICAL: chunks_vdb.upsert timed out after 300s!")
                # Try to diagnose by checking embedding function directly
                if hasattr(self.lightrag.chunks_vdb, "embedding_func"):
                    self.logger.info("Diagnostics: Testing embedding function directly...")
                    test_text = ["test embedding"]
                    try:
                        # LightRAG stores EmbeddingFunc in .embedding_func
                        # EmbeddingFunc calls .func(texts)
                        # Our wrapper is in .func
                        # Check if we need to access .func directly or if EmbeddingFunc is callable
                        ef = self.lightrag.chunks_vdb.embedding_func
                        self.logger.info(f"Diagnostics: embedding_func type: {type(ef)}")
                        if hasattr(ef, "__call__"):
                            # If ef is our wrapper or EmbeddingFunc (which is callable)
                            # EmbeddingFunc.__call__ is typically async in LightRAG 
                            # But let's check if it returns a coroutine
                            try:
                                res = ef(test_text)
                                if asyncio.iscoroutine(res):
                                    # Add timeout for diagnostics to prevent infinite hang
                                    emb = await asyncio.wait_for(res, timeout=30)
                                else:
                                    emb = res
                                self.logger.info(f"Diagnostics: Embedding function works. Result len: {len(emb)}")
                            except asyncio.TimeoutError:
                                self.logger.error("Diagnostics: Embedding function timed out during test (30s)")
                            except Exception as e:
                                self.logger.error(f"Diagnostics: Embedding function failed during execution: {e}")
                        else:
                            self.logger.error(f"Diagnostics: embedding_func {type(ef)} is not callable")

                    except Exception as e:
                        self.logger.error(f"Diagnostics: Embedding function failed: {e}")
                raise


            self.logger.debug(f"Stored {len(chunks)} multimodal chunks to storage")

        except Exception as e:
            self.logger.error(f"Error storing chunks to storage: {e}")
            raise

    async def _store_multimodal_main_entities(
        self,
        multimodal_data_list: List[Dict[str, Any]],
        lightrag_chunks: Dict[str, Any],
        file_path: str,
        doc_id: str = None,
    ):
        """
        Store multimodal main entities to entities_vdb and full_entities.
        This ensures that entities like "TableName (table)" are properly indexed.

        Args:
            multimodal_data_list: List of processed multimodal data with entity info
            lightrag_chunks: Chunks in LightRAG format (already formatted with templates)
            file_path: File path for the entities
            doc_id: Document ID for full_entities storage
        """
        if not multimodal_data_list:
            return

        # Create entities_vdb entries for all multimodal main entities
        entities_to_store = {}

        for data in multimodal_data_list:
            entity_info = data["entity_info"]
            entity_name = entity_info["entity_name"]
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Apply the same chunk template to get the formatted content
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )

            # Generate chunk_id using the formatted content (same as in _convert_to_lightrag_chunks)
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            # Generate entity_id using LightRAG's standard format
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")

            # Create entity data in LightRAG format
            entity_data = {
                "entity_name": entity_name,
                "entity_type": entity_info.get("entity_type", content_type),
                "content": entity_info.get("summary", description),
                "source_id": chunk_id,
                "file_path": os.path.basename(file_path),
            }

            entities_to_store[entity_id] = entity_data

        if entities_to_store:
            try:
                # Store entities in knowledge graph
                for entity_id, entity_data in entities_to_store.items():
                    entity_name = entity_data["entity_name"]

                    # Create node data for knowledge graph
                    node_data = {
                        "entity_id": entity_name,
                        "entity_type": entity_data["entity_type"],
                        "description": entity_data["content"],
                        "source_id": entity_data["source_id"],
                        "file_path": entity_data["file_path"],
                        "created_at": int(time.time()),
                    }

                    # Store in knowledge graph
                    await self.lightrag.chunk_entity_relation_graph.upsert_node(
                        entity_name, node_data
                    )

                # Store in entities_vdb
                await self.lightrag.entities_vdb.upsert(entities_to_store)
                await self.lightrag.entities_vdb.index_done_callback()

                # NEW: Store multimodal main entities in full_entities storage
                if doc_id and self.lightrag.full_entities:
                    await self._store_multimodal_entities_to_full_entities(
                        entities_to_store, doc_id
                    )

                self.logger.debug(
                    f"Stored {len(entities_to_store)} multimodal main entities to knowledge graph, entities_vdb, and full_entities"
                )

            except Exception as e:
                self.logger.error(f"Error storing multimodal main entities: {e}")
                raise

    async def _store_multimodal_entities_to_full_entities(
        self, entities_to_store: Dict[str, Any], doc_id: str
    ):
        """
        Store multimodal main entities to full_entities storage.

        Args:
            entities_to_store: Dictionary of entities to store
            doc_id: Document ID for grouping entities
        """
        try:
            # Get current full_entities data for this document
            current_doc_entities = await self.lightrag.full_entities.get_by_id(doc_id)

            if current_doc_entities is None:
                # Create new document entry
                entity_names = list(
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                )
                doc_entities_data = {
                    "entity_names": entity_names,
                    "count": len(entity_names),
                    "update_time": int(time.time()),
                }
            else:
                # Update existing document entry
                existing_entity_names = set(
                    current_doc_entities.get("entity_names", [])
                )
                new_entity_names = [
                    entity_data["entity_name"]
                    for entity_data in entities_to_store.values()
                ]

                # Add new multimodal entities to the list (avoid duplicates)
                for entity_name in new_entity_names:
                    existing_entity_names.add(entity_name)

                doc_entities_data = {
                    "entity_names": list(existing_entity_names),
                    "count": len(existing_entity_names),
                    "update_time": int(time.time()),
                }

            # Store updated data
            await self.lightrag.full_entities.upsert({doc_id: doc_entities_data})
            await self.lightrag.full_entities.index_done_callback()

            self.logger.debug(
                f"Added {len(entities_to_store)} multimodal main entities to full_entities for doc {doc_id}"
            )

        except Exception as e:
            self.logger.error(
                f"Error storing multimodal entities to full_entities: {e}"
            )
            raise

    async def _batch_extract_entities_lightrag_style_type_aware(
        self, lightrag_chunks: Dict[str, Any]
    ) -> List[Tuple]:
        """Use LightRAG's extract_entities for batch entity relation extraction"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import extract_entities

        # Get pipeline status (consistent with LightRAG)
        pipeline_status = await get_namespace_data("pipeline_status")

        # In offline mode, skip the expensive entity extraction step
        if os.environ.get("LLM_PROVIDER") == "offline":
            # Return a list of tuples with the correct structure, but empty data.
            # The structure is [(nodes, edges), ...].
            # We need to provide a minimal `nodes` structure so the next stage can extract the chunk_id.
            results = []
            for chunk_id in lightrag_chunks.keys():
                # Mimic the structure of a single node entry to provide the source_id
                # The key of the outer dict is the entity name, which we don't have, so we use a placeholder.
                # The important part is that `nodes_dict[0]['source_id']` can be accessed.
                nodes = {f"placeholder_entity_{chunk_id}": [{"source_id": chunk_id}]}
                edges = {}
                results.append((nodes, edges))
            return results
        pipeline_status_lock = get_pipeline_status_lock()

        # Directly use LightRAG's extract_entities
        chunk_results = await extract_entities(
            chunks=lightrag_chunks,
            global_config=self.lightrag.__dict__,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            text_chunks_storage=self.lightrag.text_chunks,
        )

        self.logger.info(
            f"Extracted entities from {len(lightrag_chunks)} multimodal chunks"
        )
        return chunk_results

    async def _batch_add_belongs_to_relations_type_aware(
        self, chunk_results: List[Tuple], multimodal_data_list: List[Dict[str, Any]]
    ) -> List[Tuple]:
        """Add belongs_to relations for multimodal entities"""
        # Create mapping from chunk_id to modal_entity_name
        chunk_to_modal_entity = {}
        chunk_to_file_path = {}

        for data in multimodal_data_list:
            description = data["description"]
            content_type = data["content_type"]
            original_item = data["original_item"]

            # Use the same template formatting as in _convert_to_lightrag_chunks_type_aware
            formatted_chunk_content = self._apply_chunk_template(
                content_type, original_item, description
            )
            chunk_id = compute_mdhash_id(formatted_chunk_content, prefix="chunk-")

            chunk_to_modal_entity[chunk_id] = data["entity_info"]["entity_name"]
            chunk_to_file_path[chunk_id] = data.get("file_path", "multimodal_content")

        enhanced_chunk_results = []
        belongs_to_count = 0

        for maybe_nodes, maybe_edges in chunk_results:
            # Find corresponding modal_entity_name for this chunk
            chunk_id = None
            for nodes_dict in maybe_nodes.values():
                if nodes_dict:
                    chunk_id = nodes_dict[0].get("source_id")
                    break

            if chunk_id and chunk_id in chunk_to_modal_entity:
                modal_entity_name = chunk_to_modal_entity[chunk_id]
                file_path = chunk_to_file_path.get(chunk_id, "multimodal_content")

                # Add belongs_to relations for all extracted entities
                for entity_name in maybe_nodes.keys():
                    if entity_name != modal_entity_name:  # Avoid self-relation
                        belongs_to_relation = {
                            "src_id": entity_name,
                            "tgt_id": modal_entity_name,
                            "description": f"Entity {entity_name} belongs to {modal_entity_name}",
                            "keywords": "belongs_to,part_of,contained_in",
                            "source_id": chunk_id,
                            "weight": 10.0,
                            "file_path": file_path,
                        }

                        # Add to maybe_edges
                        edge_key = (entity_name, modal_entity_name)
                        if edge_key not in maybe_edges:
                            maybe_edges[edge_key] = []
                        maybe_edges[edge_key].append(belongs_to_relation)
                        belongs_to_count += 1

            enhanced_chunk_results.append((maybe_nodes, maybe_edges))

        self.logger.info(
            f"Added {belongs_to_count} belongs_to relations for multimodal entities"
        )
        return enhanced_chunk_results

    async def _batch_merge_lightrag_style_type_aware(
        self, enhanced_chunk_results: List[Tuple], file_path: str, doc_id: str = None
    ):
        """Use LightRAG's merge_nodes_and_edges for batch merge"""
        from lightrag.kg.shared_storage import (
            get_namespace_data,
            get_pipeline_status_lock,
        )
        from lightrag.operate import merge_nodes_and_edges

        pipeline_status = await get_namespace_data("pipeline_status")
        pipeline_status_lock = get_pipeline_status_lock()

        await merge_nodes_and_edges(
            chunk_results=enhanced_chunk_results,
            knowledge_graph_inst=self.lightrag.chunk_entity_relation_graph,
            entity_vdb=self.lightrag.entities_vdb,
            relationships_vdb=self.lightrag.relationships_vdb,
            global_config=self.lightrag.__dict__,
            full_entities_storage=self.lightrag.full_entities,
            full_relations_storage=self.lightrag.full_relations,
            doc_id=doc_id,
            pipeline_status=pipeline_status,
            pipeline_status_lock=pipeline_status_lock,
            llm_response_cache=self.lightrag.llm_response_cache,
            current_file_number=1,
            total_files=1,
            file_path=os.path.basename(file_path),
        )

        await self.lightrag._insert_done()

    async def _update_doc_status_with_chunks_type_aware(
        self, doc_id: str, chunk_ids: List[str]
    ):
        """Update document status with multimodal chunks"""
        try:
            # Get current document status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)

            if current_doc_status:
                existing_chunks_list = current_doc_status.get("chunks_list", [])
                existing_chunks_count = current_doc_status.get("chunks_count", 0)

                # Add multimodal chunks to the standard chunks_list
                updated_chunks_list = existing_chunks_list + chunk_ids
                updated_chunks_count = existing_chunks_count + len(chunk_ids)

                # Update document status with integrated chunk list
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,  # Keep existing fields
                            "chunks_list": updated_chunks_list,  # Integrated chunks list
                            "chunks_count": updated_chunks_count,  # Updated total count
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )

                # Ensure doc_status update is persisted to disk
                await self.lightrag.doc_status.index_done_callback()

                self.logger.info(
                    f"Updated doc_status: added {len(chunk_ids)} multimodal chunks to standard chunks_list "
                    f"(total chunks: {updated_chunks_count})"
                )

        except Exception as e:
            self.logger.warning(
                f"Error updating doc_status with multimodal chunks: {e}"
            )

    async def _mark_multimodal_processing_complete(self, doc_id: str):
        """Mark multimodal content processing as complete in the document status."""
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            **current_doc_status,
                            "multimodal_processed": True,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
                self.logger.debug(
                    f"Marked multimodal content processing as complete for document {doc_id}"
                )
        except Exception as e:
            self.logger.warning(
                f"Error marking multimodal processing as complete for document {doc_id}: {e}"
            )

    async def is_document_fully_processed(self, doc_id: str) -> bool:
        """
        Check if a document is fully processed (both text and multimodal content).

        Args:
            doc_id: Document ID to check

        Returns:
            bool: True if both text and multimodal content are processed
        """
        try:
            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return False

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)

            return text_processed and multimodal_processed

        except Exception as e:
            self.logger.error(
                f"Error checking document processing status for {doc_id}: {e}"
            )
            return False

    async def get_document_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """
        Get detailed processing status for a document.

        Args:
            doc_id: Document ID to check

        Returns:
            Dict with processing status details
        """
        try:
            init_result = await self._ensure_lightrag_initialized()
            if not init_result["success"]:
                error_msg = init_result.get("error", "LightRAG initialization failed")
                self.logger.error(
                    f"LightRAG not initialized when checking document status for {doc_id}: {error_msg}"
                )
                return {
                    "exists": False,
                    "error": error_msg,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not doc_status:
                return {
                    "exists": False,
                    "text_processed": False,
                    "multimodal_processed": False,
                    "fully_processed": False,
                    "chunks_count": 0,
                }

            text_processed = doc_status.get("status") == DocStatus.PROCESSED
            multimodal_processed = doc_status.get("multimodal_processed", False)
            fully_processed = text_processed and multimodal_processed

            return {
                "exists": True,
                "text_processed": text_processed,
                "multimodal_processed": multimodal_processed,
                "fully_processed": fully_processed,
                "chunks_count": doc_status.get("chunks_count", 0),
                "chunks_list": doc_status.get("chunks_list", []),
                "status": doc_status.get("status", ""),
                "updated_at": doc_status.get("updated_at", ""),
                "raw_status": doc_status,
            }

        except Exception as e:
            self.logger.error(
                f"Error getting document processing status for {doc_id}: {e}"
            )
            return {
                "exists": False,
                "error": str(e),
                "text_processed": False,
                "multimodal_processed": False,
                "fully_processed": False,
                "chunks_count": 0,
            }

    async def process_document_complete(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        file_name: str | None = None,
        **kwargs,
    ):
        self.logger.info(f"Starting process_document_complete for doc_id: {doc_id}")
        """
        Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        # Ensure LightRAG is initialized
        result = await self._ensure_lightrag_initialized()
        if not result["success"]:
            self.logger.error(f"LightRAG initialization failed: {result.get('error')}")
            return

        # Use config defaults if not provided
        if output_dir is None:
            output_dir = self.config.parser_output_dir
        if parse_method is None:
            parse_method = self.config.parse_method
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(f"Starting complete document processing: {file_path}")

        # Step 1: Parse document
        self.logger.info(f"Step 1: Parsing document {file_path} with method {parse_method}")
        content_list, content_based_doc_id = await self.parse_document(
            file_path, output_dir, parse_method, display_stats, **kwargs
        )
        self.logger.info(f"Step 1: Parsing complete. Content items: {len(content_list) if content_list else 0}")

        # Use provided doc_id or fall back to content-based doc_id
        if doc_id is None:
            doc_id = content_based_doc_id

        # Ensure a doc_status record exists early for this doc_id
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not current_doc_status:
                self.logger.info(f"Creating initial doc_status for {doc_id} in process_document_complete")
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.HANDLING,
                            "chunks_count": 0,
                            "multimodal_processed": False,
                            "updated_at": time.strftime(
                                "%Y-%m-%dT%H:%M:%S+00:00"
                            ),
                            "file_path": os.path.basename(file_path),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
            else:
                self.logger.info(f"Existing doc_status found for {doc_id}: {current_doc_status}")
        except Exception as e:
            self.logger.error(f"Error creating initial doc_status for {doc_id}: {e}")

        # Step 2: Separate text and multimodal content
        self.logger.info("Step 2: Separating text and multimodal content")
        text_content, multimodal_items = separate_content(content_list)
        self.logger.info(f"Step 2: Separation complete. Text length: {len(text_content)}, Multimodal items: {len(multimodal_items)}")

        # Step 2.5: Set content source for context extraction in multimodal processing
        if hasattr(self, "set_content_source_for_context") and multimodal_items:
            self.logger.info(
                "Setting content source for context-aware multimodal processing..."
            )
            self.set_content_source_for_context(
                content_list, self.config.content_format
            )

        # Auto split by segment when video audio segments exist and user did not specify splitting
        auto_split_char = split_by_character
        auto_split_only = split_by_character_only
        if auto_split_char is None:
            for it in content_list:
                if isinstance(it, dict) and it.get("source_type") == "video_audio":
                    auto_split_char = "\n\n"
                    auto_split_only = True
                    break

        # Step 3: Insert pure text content with all parameters
        if text_content.strip():
            self.logger.info(f"Step 3: Inserting text content for {doc_id}")
            if file_name is None:
                file_name = os.path.basename(file_path)

            # Use direct LightRAG insert
            insert_func = getattr(self.lightrag, "insert", None)
            if not insert_func:
                self.logger.error("LightRAG instance has no 'insert' method")
                return

            # Prepare arguments for insert
            insert_kwargs = {
                "input": text_content,
                "split_by_character": auto_split_char,
                "split_by_character_only": auto_split_only,
                "ids": doc_id,
                "file_paths": file_name,
            }

            # CRITICAL FIX: Clear pre-existing status to prevent LightRAG from skipping
            # because we created a placeholder status earlier which LightRAG interprets as "already processed"
            try:
                if await self.lightrag.doc_status.get_by_id(doc_id):
                    self.logger.info(f"Clearing pre-existing status for {doc_id} to force processing")
                    await self.lightrag.doc_status.delete([doc_id])
            except Exception as e:
                self.logger.warning(f"Failed to clear status for {doc_id}: {e}")

            try:
                if asyncio.iscoroutinefunction(insert_func):
                    await insert_func(**insert_kwargs)
                else:
                    # If it's sync, run it in executor to avoid blocking loop
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: insert_func(**insert_kwargs))
                self.logger.info(f"Step 3: Text insertion complete for {doc_id}")
            except Exception as e:
                self.logger.error(f"Step 3: Error during text insertion: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

            # Update doc_status to PROCESSED for text part
            try:
                self.logger.info(f"Step 3.5: Updating doc_status to PROCESSED for text part")
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                if current_doc_status:
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,
                                "status": DocStatus.PROCESSED,
                                "text_processed": True,
                                "updated_at": time.strftime(
                                    "%Y-%m-%dT%H:%M:%S+00:00"
                                ),
                            }
                        }
                    )
                    await self.lightrag.doc_status.index_done_callback()
                self.logger.info(f"Step 3.5: doc_status updated successfully")
            except Exception as e:
                self.logger.error(
                    f"Error updating doc_status after text insert for {doc_id}: {e}"
                )
        else:
            self.logger.info("Step 3: No text content to insert")
            # Mark text_processed=True even if empty, so we don't stall
            try:
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                if current_doc_status:
                     await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,
                                "text_processed": True,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )
                     await self.lightrag.doc_status.index_done_callback()
            except Exception as e:
                self.logger.error(f"Error marking empty text as processed: {e}")

        # Step 4: Process multimodal content (using specialized processors)
        self.logger.info(f"Step 4: Processing multimodal content. Items: {len(multimodal_items)}")
        if multimodal_items:
            await self._process_multimodal_content(multimodal_items, file_path, doc_id)
        else:
            # If no multimodal content, mark multimodal processing as complete
            # This ensures the document status properly reflects completion of all processing
            await self._mark_multimodal_processing_complete(doc_id)
            self.logger.debug(
                f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
            )

        self.logger.info(f"Document {file_path} processing complete!")
        return {
            "doc_id": doc_id,
            "content_based_doc_id": content_based_doc_id,
        }

    async def process_document_complete_lightrag_api(
        self,
        file_path: str,
        output_dir: str = None,
        parse_method: str = None,
        display_stats: bool = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        scheme_name: str | None = None,
        parser: str | None = None,
        **kwargs,
    ):
        """
        API exclusively for LightRAG calls: Complete document processing workflow

        Args:
            file_path: Path to the file to process
            output_dir: output directory (defaults to config.parser_output_dir)
            parse_method: Parse method (defaults to config.parse_method)
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            **kwargs: Additional parameters for parser (e.g., lang, device, start_page, end_page, formula, table, backend, source)
        """
        file_name = os.path.basename(file_path)
        doc_pre_id = f"doc-pre-{file_name}"
        pipeline_status = None
        pipeline_status_lock = None

        if parser:
            self.config.parser = parser

        current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)

        try:
            # Ensure LightRAG is initialized
            result = await self._ensure_lightrag_initialized()
            if not result["success"]:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": result["error"],
                        }
                    }
                )
                return False

            # Use config defaults if not provided
            if output_dir is None:
                output_dir = self.config.parser_output_dir
            if parse_method is None:
                parse_method = self.config.parse_method
            if display_stats is None:
                display_stats = self.config.display_content_stats

            self.logger.info(f"Starting complete document processing: {file_path}")

            # Initialize doc status
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_pre_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            "status": DocStatus.READY,
                            "content": "",
                            "error_msg": "",
                            "content_summary": "",
                            "multimodal_content": [],
                            "scheme_name": scheme_name,
                            "content_length": 0,
                            "created_at": "",
                            "updated_at": "",
                            "file_path": file_name,
                        }
                    }
                )
                current_doc_status = await self.lightrag.doc_status.get_by_id(
                    doc_pre_id
                )

            from lightrag.kg.shared_storage import (
                get_namespace_data,
                get_pipeline_status_lock,
            )

            pipeline_status = await get_namespace_data("pipeline_status")
            pipeline_status_lock = get_pipeline_status_lock()

            # Set processing status
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": True})
                pipeline_status["history_messages"].append("Now is not allowed to scan")

            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.HANDLING,
                        "error_msg": "",
                    }
                }
            )

            content_list = []
            content_based_doc_id = ""

            try:
                # Step 1: Parse document
                content_list, content_based_doc_id = await self.parse_document(
                    file_path, output_dir, parse_method, display_stats, **kwargs
                )
            except MineruExecutionError as e:
                error_message = e.error_msg
                if isinstance(e.error_msg, list):
                    error_message = "\n".join(e.error_msg)
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": error_message,
                        }
                    }
                )
                self.logger.info(
                    f"Error processing document {file_path}: MineruExecutionError"
                )
                return False
            except Exception as e:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_pre_id: {
                            **current_doc_status,
                            "status": DocStatus.FAILED,
                            "error_msg": str(e),
                        }
                    }
                )
                self.logger.info(f"Error processing document {file_path}: {str(e)}")
                return False

            # Use provided doc_id or fall back to content-based doc_id
            if doc_id is None:
                doc_id = content_based_doc_id

            # Step 2: Separate text and multimodal content
            text_content, multimodal_items = separate_content(content_list)

            # Step 2.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            # Auto split when video audio segments are present
            auto_split_char = split_by_character
            auto_split_only = split_by_character_only
            if auto_split_char is None:
                for it in content_list:
                    if isinstance(it, dict) and it.get("source_type") == "video_audio":
                        auto_split_char = "\n\n"
                        auto_split_only = True
                        break

            if text_content.strip():
                # Add debug logging before insert call
                self.logger.debug(f"Before insert call - LightRAG instance: {self.lightrag}, Type: {type(self.lightrag)}")
                if self.lightrag is None:
                    self.logger.error("CRITICAL: LightRAG instance is None before insert call!")
                    return False

                # Use direct LightRAG insert
                insert_func = getattr(self.lightrag, "insert", None)
                if not insert_func:
                    self.logger.error("LightRAG instance has no 'insert' method")
                    return False
    
                # Prepare arguments for insert
                insert_kwargs = {
                    "input": text_content,
                    "split_by_character": auto_split_char,
                    "split_by_character_only": auto_split_only,
                    "ids": doc_id,
                    "file_paths": file_name,
                }
                
                if asyncio.iscoroutinefunction(insert_func):
                    await insert_func(**insert_kwargs)
                else:
                    # If it's sync, run it in executor to avoid blocking loop
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, lambda: insert_func(**insert_kwargs))

            self.logger.info(f"Document {file_path} processing completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {str(e)}")
            self.logger.debug("Exception details:", exc_info=True)

            # Update doc status to Failed
            await self.lightrag.doc_status.upsert(
                {
                    doc_pre_id: {
                        **current_doc_status,
                        "status": DocStatus.FAILED,
                        "error_msg": str(e),
                    }
                }
            )
            await self.lightrag.doc_status.index_done_callback()

            # Update pipeline status
            if pipeline_status_lock and pipeline_status:
                try:
                    async with pipeline_status_lock:
                        pipeline_status.update({"scan_disabled": False})
                        error_msg = (
                            f"RAGAnything processing failed for {file_name}: {str(e)}"
                        )
                        pipeline_status["latest_message"] = error_msg
                        pipeline_status["history_messages"].append(error_msg)
                        pipeline_status["history_messages"].append(
                            "Now is allowed to scan"
                        )
                except Exception as pipeline_update_error:
                    self.logger.error(
                        f"Failed to update pipeline status: {pipeline_update_error}"
                    )

            return False

        finally:
            async with pipeline_status_lock:
                pipeline_status.update({"scan_disabled": False})
                pipeline_status["latest_message"] = (
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append(
                    f"RAGAnything processing completed for {file_name}"
                )
                pipeline_status["history_messages"].append("Now is allowed to scan")

    async def insert_content_list(
        self,
        content_list: List[Dict[str, Any]],
        file_path: str = "unknown_document",
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        doc_id: str | None = None,
        display_stats: bool = None,
    ):
        """
        Insert content list directly without document parsing

        Args:
            content_list: Pre-parsed content list containing text and multimodal items.
                         Each item should be a dictionary with the following structure:
                         - Text: {"type": "text", "text": "content", "page_idx": 0}
                         - Image: {"type": "image", "img_path": "/absolute/path/to/image.jpg",
                                  "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}
                         - Table: {"type": "table", "table_body": "markdown table",
                                  "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}
                         - Equation: {"type": "equation", "latex": "LaTeX formula",
                                     "text": "description", "page_idx": 3}
                         - Generic: {"type": "custom_type", "content": "any content", "page_idx": 4}
            file_path: Reference file path/name for citation (defaults to "unknown_document")
            split_by_character: Optional character to split the text by
            split_by_character_only: If True, split only by the specified character
            doc_id: Optional document ID, if not provided will be generated from content
            display_stats: Whether to display content statistics (defaults to config.display_content_stats)

        Note:
            - img_path must be an absolute path to the image file
            - page_idx represents the page number where the content appears (0-based indexing)
            - Items are processed in the order they appear in the list
        """
        # Ensure LightRAG is initialized
        await self._ensure_lightrag_initialized()

        # Use config defaults if not provided
        if display_stats is None:
            display_stats = self.config.display_content_stats

        self.logger.info(
            f"Starting direct content list insertion for: {file_path} ({len(content_list)} items)"
        )

        # Generate doc_id based on content if not provided
        if doc_id is None:
            doc_id = self._generate_content_based_doc_id(content_list)

        # Display content statistics if requested
        if display_stats:
            self.logger.info("\nContent Information:")
            self.logger.info(f"* Total blocks in content_list: {len(content_list)}")

            # Count elements by type
            block_types: Dict[str, int] = {}
            for block in content_list:
                if isinstance(block, dict):
                    block_type = block.get("type", "unknown")
                    if isinstance(block_type, str):
                        block_types[block_type] = block_types.get(block_type, 0) + 1

            self.logger.info("* Content block types:")
            for block_type, count in block_types.items():
                self.logger.info(f"  - {block_type}: {count}")

        # Step 1: Separate text and multimodal content
        text_content, multimodal_items = separate_content(content_list)

        # Ensure a doc_status record exists early
        try:
            current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
            if not current_doc_status:
                await self.lightrag.doc_status.upsert(
                    {
                        doc_id: {
                            "status": DocStatus.HANDLING,
                            "chunks_count": 0,
                            "multimodal_processed": False,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            "file_path": os.path.basename(file_path),
                        }
                    }
                )
                await self.lightrag.doc_status.index_done_callback()
        except Exception as e:
            self.logger.error(f"Error creating initial doc_status for {doc_id}: {e}")

        try:
            # Step 1.5: Set content source for context extraction in multimodal processing
            if hasattr(self, "set_content_source_for_context") and multimodal_items:
                self.logger.info(
                    "Setting content source for context-aware multimodal processing..."
                )
                self.set_content_source_for_context(
                    content_list, self.config.content_format
                )

            self.logger.info(f"Step 2: Inserting text content for doc_id: {doc_id}")
            # Step 2: Insert pure text content with all parameters
            if text_content.strip():
                file_name = os.path.basename(file_path)
                await insert_text_content(
                    self.lightrag,
                    input=text_content,
                    file_paths=file_name,
                    split_by_character=split_by_character,
                    split_by_character_only=split_by_character_only,
                    ids=doc_id,
                )
            self.logger.info(f"Step 2: Text content insertion complete for doc_id: {doc_id}")

            # Update status to PROCESSED for text part
            try:
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                if current_doc_status:
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,
                                "status": DocStatus.PROCESSED,
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )
                    await self.lightrag.doc_status.index_done_callback()
            except Exception as e:
                self.logger.error(f"Error updating doc_status after text insert: {e}")

            self.logger.info(f"Step 3: Processing multimodal content for doc_id: {doc_id}")
            # Step 3: Process multimodal content (using specialized processors)
            if multimodal_items:
                await self._process_multimodal_content(multimodal_items, file_path, doc_id)
            else:
                # If no multimodal content, mark multimodal processing as complete
                # This ensures the document status properly reflects completion of all processing
                await self._mark_multimodal_processing_complete(doc_id)
                self.logger.debug(
                    f"No multimodal content found in document {doc_id}, marked multimodal processing as complete"
                )
            self.logger.info(f"Step 3: Multimodal content processing complete for doc_id: {doc_id}")

        except Exception as e:
            self.logger.error(f"Error processing content list for {doc_id}: {e}")
            # Mark as FAILED
            try:
                current_doc_status = await self.lightrag.doc_status.get_by_id(doc_id)
                if current_doc_status:
                    await self.lightrag.doc_status.upsert(
                        {
                            doc_id: {
                                **current_doc_status,
                                "status": DocStatus.FAILED,
                                "error_msg": str(e),
                                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                            }
                        }
                    )
                    await self.lightrag.doc_status.index_done_callback()
            except Exception as update_err:
                self.logger.error(f"Failed to update doc_status to FAILED: {update_err}")
            raise e

        self.logger.info(f"Content list insertion complete for: {file_path}")
