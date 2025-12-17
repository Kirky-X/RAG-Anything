# Copyright (c) 2025 Kirky.X
# All rights reserved.

"""
Image modal processor
"""

import json
import base64
import asyncio
from typing import Dict, Any, Tuple
from pathlib import Path

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)
from lightrag.lightrag import LightRAG

# Import prompt templates
from raganything.prompt import PROMPTS

from .base import BaseModalProcessor, ContextExtractor


class ImageModalProcessor(BaseModalProcessor):
    """Processor specialized for image content"""

    def __init__(
        self,
        lightrag: LightRAG,
        modal_caption_func,
        context_extractor: ContextExtractor = None,
    ):
        """Initialize image processor

        Args:
            lightrag: LightRAG instance
            modal_caption_func: Function for generating descriptions (supporting image understanding)
            context_extractor: Context extractor instance
        """
        super().__init__(lightrag, modal_caption_func, context_extractor)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate image description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Image content to process
            content_type: Type of modal content ("image")
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        try:
            # Parse image content (reuse existing logic)
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_path = content_data.get("img_path")
            captions = content_data.get(
                "image_caption", content_data.get("img_caption", [])
            )
            footnotes = content_data.get(
                "image_footnote", content_data.get("img_footnote", [])
            )

            # Validate image path
            if not image_path:
                raise ValueError(
                    f"No image path provided in modal_content: {modal_content}"
                )

            # Convert to Path object and check if it exists
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Extract context for current item
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # Build detailed visual analysis prompt with context
            if context:
                vision_prompt = PROMPTS.get(
                    "vision_prompt_with_context", PROMPTS["vision_prompt"]
                ).format(
                    context=context,
                    entity_name=entity_name
                    if entity_name
                    else "unique descriptive name for this image",
                    image_path=image_path,
                    captions=captions if captions else "None",
                    footnotes=footnotes if footnotes else "None",
                )
            else:
                vision_prompt = PROMPTS["vision_prompt"].format(
                    entity_name=entity_name
                    if entity_name
                    else "unique descriptive name for this image",
                    image_path=image_path,
                    captions=captions if captions else "None",
                    footnotes=footnotes if footnotes else "None",
                )

            # Encode image to base64
            image_base64 = self._encode_image_to_base64(image_path)
            if not image_base64:
                raise RuntimeError(f"Failed to encode image to base64: {image_path}")

            # Call vision model with encoded image
            logger.info(f"Calling VLM for image analysis: {image_path}")
            try:
                # Remove system_prompt as it's not supported by all LLM functions
                # Some LLM functions (like those using use_llm_func_with_cache) don't accept system_prompt
                # We prepend it to the prompt instead
                full_prompt = f"{PROMPTS['IMAGE_ANALYSIS_SYSTEM']}\n\n{vision_prompt}"
                
                # Use a simpler invocation method if the wrapped function is a LangChainLLM
                # to avoid potential issues with argument handling in async contexts
                response = await asyncio.wait_for(
                    self.modal_caption_func(
                        full_prompt,
                        image_data=image_base64,
                    ),
                    timeout=300,  # 5 minutes timeout
                )
                logger.info(f"VLM response received for image: {image_path}")
            except (asyncio.TimeoutError, Exception) as call_err:
                error_msg = str(call_err)
                logger.error(f"VLM call failed or timed out for image {image_path}: {error_msg}")
                # Check if it's a connection error
                if "Connection" in error_msg or "connect" in error_msg or "ClientConnectorError" in error_msg:
                     logger.warning("Connection error detected. Returning fallback response.")
                
                # Return a fallback response instead of raising to allow pipeline to continue
                response = json.dumps({
                    "detailed_description": f"Image analysis failed or timed out for {image_path}. Error: {error_msg}",
                    "entity_info": {
                        "entity_name": entity_name if entity_name else f"image_{compute_mdhash_id(str(modal_content))}",
                        "entity_type": "image",
                        "summary": "Analysis failed or timed out"
                    }
                })

            # Parse response (reuse existing logic)
            enhanced_caption, entity_info = self._parse_response(response, entity_name)

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity

    async def process_multimodal_content(
        self,
        modal_content,
        content_type: str,
        file_path: str = "manual_creation",
        entity_name: str = None,
        item_info: Dict[str, Any] = None,
        batch_mode: bool = False,
        doc_id: str = None,
        chunk_order_index: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        """Process image content with context support"""
        try:
            # Generate description and entity info
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # Build complete image content
            if isinstance(modal_content, str):
                try:
                    content_data = json.loads(modal_content)
                except json.JSONDecodeError:
                    content_data = {"description": modal_content}
            else:
                content_data = modal_content

            image_path = content_data.get("img_path", "")
            captions = content_data.get(
                "image_caption", content_data.get("img_caption", [])
            )
            footnotes = content_data.get(
                "image_footnote", content_data.get("img_footnote", [])
            )

            modal_chunk = PROMPTS["image_chunk"].format(
                image_path=image_path,
                captions=", ".join(captions) if captions else "None",
                footnotes=", ".join(footnotes) if footnotes else "None",
                enhanced_caption=enhanced_caption,
            )

            return await self._create_entity_and_chunk(
                modal_chunk,
                entity_info,
                file_path,
                batch_mode,
                doc_id,
                chunk_order_index,
            )

        except Exception as e:
            logger.error(f"Error processing image content: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(str(modal_content))}",
                "entity_type": "image",
                "summary": f"Image content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity, [], []

    def _parse_response(
        self, response: str, entity_name: str = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse model response"""
        try:
            # Check if response is empty
            if not response or not response.strip():
                raise ValueError("Empty response received from model")

            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            # Try to recover if entity_info is missing but description is present
            if description and not entity_data:
                # If we have a description but no entity info, try to construct a basic entity info
                # This often happens when models forget the JSON structure but provide good text
                
                # Try to extract a name from the description (first few words)
                name_candidate = description.split('.')[0][:50]
                
                entity_data = {
                    "entity_name": entity_name if entity_name else f"entity_{compute_mdhash_id(description)}",
                    "entity_type": "image" if isinstance(self, ImageModalProcessor) else "content",
                    "summary": description[:200]
                }
                logger.warning("Recovered from missing entity_info using description")

            if not description or not entity_data:
                # If robust parsing returned empty or malformed data
                # Check if the raw response itself is the description (common with some models)
                if isinstance(response, str) and len(response) > 0 and not response.strip().startswith('{'):
                    description = response
                    entity_data = {
                        "entity_name": entity_name if entity_name else f"entity_{compute_mdhash_id(description)}",
                        "entity_type": "image" if isinstance(self, ImageModalProcessor) else "content",
                        "summary": description[:200]
                    }
                    logger.warning("Used raw response as description (model failed to output JSON)")
                else:
                    raise ValueError("Missing required fields in response and could not recover")

            # Validate entity data fields and fill defaults if missing
            if "entity_name" not in entity_data:
                entity_data["entity_name"] = entity_name if entity_name else f"entity_{compute_mdhash_id(description)}"
            
            if "entity_type" not in entity_data:
                entity_data["entity_type"] = "image" if isinstance(self, ImageModalProcessor) else "content"
                
            if "summary" not in entity_data:
                entity_data["summary"] = description[:200]

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing image analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"image_{compute_mdhash_id(response)}",
                "entity_type": "image",
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity
