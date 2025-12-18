#!/usr/bin/env python
"""
RAGAnything CLI Entry Point
"""

import argparse
import asyncio
import os
import re
# Add project root directory to Python path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Use Ollama for LLM instead of OpenAI
from lightrag.llm.ollama import _ollama_model_if_cache as ollama_complete_if_cache
from lightrag.utils import set_verbose_debug
import lightrag.operate

# Monkey patch handle_cache in lightrag.operate to fix argument error
if hasattr(lightrag.operate, 'handle_cache'):
    original_handle_cache = lightrag.operate.handle_cache


    async def patched_handle_cache(hashing_kv, args_hash, prompt, mode="default", cache_type="unknown"):
        # Call the original function with the correct signature
        return await original_handle_cache(hashing_kv, args_hash, prompt, mode, cache_type)


    lightrag.operate.handle_cache = patched_handle_cache

# Monkey patch fix_tuple_delimiter_corruption in lightrag.operate
if hasattr(lightrag.operate, 'fix_tuple_delimiter_corruption'):
    def patched_fix_tuple_delimiter_corruption(record, delimiter_core, tuple_delimiter):
        if not record or not delimiter_core or not tuple_delimiter:
            return record

        # Escape the delimiter core for regex use
        escaped_delimiter_core = re.escape(delimiter_core)

        # Apply fixes (simplified version of the one in utils.py)
        # Fix: <|##|> -> <|#|>, etc.
        record = re.sub(
            rf"<\|{escaped_delimiter_core}\|*?{escaped_delimiter_core}\|>",
            tuple_delimiter,
            record,
        )
        return record


    lightrag.operate.fix_tuple_delimiter_corruption = patched_fix_tuple_delimiter_corruption

from raganything import RAGAnything, RAGAnythingConfig
from raganything.config import DirectoryConfig, ParsingConfig, MultimodalConfig
from raganything.logger import logger

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


def configure_logging():
    """Configure logging for the application using local logger"""
    # Use the local logger that's already configured in raganything.logger
    # The logger is automatically initialized when imported
    logger.info("RAGAnything CLI logging initialized")
    
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


async def process_with_rag(
    file_path: str,
    output_dir: str,
    working_dir: str = None,
    parser: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        working_dir: Working directory for RAG storage
        parser: Parser to use (mineru or docling)
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            directory=DirectoryConfig(
                working_dir=working_dir or "./rag_storage",
                parser_output_dir=output_dir,
            ),
            parsing=ParsingConfig(
                parser=parser,  # Parser selection: mineru or docling
                parse_method="auto",  # Parse method: auto, ocr, or txt
                display_content_stats=True,
            ),
            multimodal=MultimodalConfig(
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
                enable_audio_processing=True,
                enable_video_processing=True,
            ),
        )

        # Define LLM model function using Ollama
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            # Debug logging to trace parameter types
            print(f"=== DEBUG llm_model_func called ===")
            print(f"prompt type: {type(prompt)}, value: {repr(prompt)[:200]}")
            print(f"system_prompt type: {type(system_prompt)}, value: {repr(system_prompt)[:200]}")
            print(f"history_messages type: {type(history_messages)}, length: {len(history_messages)}")
            print(f"kwargs keys: {list(kwargs.keys())}")
            
            # Check if prompt is a dict (which would cause the encode error)
            if isinstance(prompt, dict):
                print(f"ERROR: prompt is a dict instead of string!")
                print(f"prompt content: {prompt}")
                # Try to extract text content from dict
                if "query" in prompt:
                    prompt = prompt["query"]
                    print(f"Extracted query from dict: {prompt}")
                elif "content" in prompt:
                    prompt = prompt["content"]
                    print(f"Extracted content from dict: {prompt}")
                else:
                    # Convert dict to string as fallback
                    prompt = str(prompt)
                    print(f"Converted dict to string: {prompt}")
            
            # Handle keyword_extraction for Ollama
            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["format"] = "json"

            return ollama_complete_if_cache(
                model=os.getenv("LLM_MODEL", "qwen3-vl:2b"),
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                host=os.getenv("OLLAMA_BASE_URL", "http://172.24.160.1:11434"),
                **kwargs,
            )

        # Define vision model function for image processing using Ollama
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # Handle keyword_extraction for Ollama
            keyword_extraction = kwargs.pop("keyword_extraction", None)
            if keyword_extraction:
                kwargs["format"] = "json"

            # If messages format is provided (for multimodal VLM enhanced query), use it directly
            if messages:
                # Extract the prompt content from messages if prompt is empty
                if not prompt and messages:
                    # Find the last user message or use the first message content
                    for msg in reversed(messages):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                # Handle multimodal content
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        prompt = item.get("text", "")
                                        break
                            else:
                                prompt = content
                            break
                
                return ollama_complete_if_cache(
                    model=os.getenv("LLM_MODEL", "qwen3-vl:2b"),
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    host=os.getenv("OLLAMA_BASE_URL", "http://172.24.160.1:11434"),
                    **kwargs,
                )
            # Traditional single image format
            elif image_data:
                return ollama_complete_if_cache(
                    model=os.getenv("LLM_MODEL", "qwen3-vl:2b"),
                    prompt="",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt}
                        if system_prompt
                        else None,
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        }
                        if image_data
                        else {"role": "user", "content": prompt},
                    ],
                    host=os.getenv("OLLAMA_BASE_URL", "http://172.24.160.1:11434"),
                    **kwargs,
                )
            # Pure text format
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function using Ollama instead of OpenAI
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
        embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
        ollama_host = os.getenv("OLLAMA_BASE_URL", "http://172.24.160.1:11434")

        # Use the build_embedding_func from raganything.llm.embedding
        from raganything.llm.embedding import build_embedding_func

        embedding_func = build_embedding_func(
            provider="ollama",
            model=embedding_model,
            api_base=ollama_host,
            embedding_dim=embedding_dim,
            max_token_size=8192,
        )

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto", video_fps=0.05
        )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        logger.info(
            "\n[Multimodal Query]: Analyzing performance data in context of document"
        )
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this performance data with any similar results mentioned in the document",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Method,Accuracy,Processing_Time
                                RAGAnything,95.2%,120ms
                                Traditional_RAG,87.3%,180ms
                                Baseline,82.1%,200ms""",
                    "table_caption": "Performance comparison results",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {multimodal_result}")

        # 3. Another multimodal query with equation content
        logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        equation_result = await rag.aquery_with_multimodal(
            "Explain this formula and relate it to any mathematical concepts in the document",
            multimodal_content=[
                {
                    "type": "equation",
                    "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
                    "equation_caption": "F1-score calculation formula",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"Answer: {equation_result}")

    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="RAGAnything CLI")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--parser",
        default=os.getenv("PARSER", "mineru"),
        help="Parser to use (defaults to PARSER env var or 'mineru')",
    )

    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path,
            args.output,
            args.working_dir,
            args.parser,
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything CLI")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()
