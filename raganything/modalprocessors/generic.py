# Copyright (c) 2025 Kirky.X
# All rights reserved.

"""
Generic modal processor
"""

import json
from typing import Dict, Any, Tuple

from lightrag.utils import (
    logger,
    compute_mdhash_id,
)

# Import prompt templates
from raganything.prompt import PROMPTS

from .base import BaseModalProcessor


class GenericModalProcessor(BaseModalProcessor):
    """Generic processor for other types of modal content"""

    async def generate_description_only(
        self,
        modal_content,
        content_type: str,
        item_info: Dict[str, Any] = None,
        entity_name: str = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate generic modal description and entity info only, without entity relation extraction.
        Used for batch processing stage 1.

        Args:
            modal_content: Generic modal content to process
            content_type: Type of modal content
            item_info: Item information for context extraction
            entity_name: Optional predefined entity name

        Returns:
            Tuple of (enhanced_caption, entity_info)
        """
        try:
            # Extract context for current item
            context = ""
            if item_info:
                context = self._get_context_for_item(item_info)

            # Build generic analysis prompt with context
            if context:
                generic_prompt = PROMPTS.get(
                    "generic_prompt_with_context", PROMPTS["generic_prompt"]
                ).format(
                    context=context,
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )
            else:
                generic_prompt = PROMPTS["generic_prompt"].format(
                    content_type=content_type,
                    entity_name=entity_name
                    if entity_name
                    else f"descriptive name for this {content_type}",
                    content=str(modal_content),
                )

            # Call LLM for generic analysis
            system_prompt = PROMPTS["GENERIC_ANALYSIS_SYSTEM"].format(
                content_type=content_type
            )
            # Prepend system prompt to the main prompt
            full_prompt = f"{system_prompt}\n\n{generic_prompt}"
            
            response = await self.modal_caption_func(
                full_prompt
            )

            # Parse response (reuse existing logic)
            enhanced_caption, entity_info = self._parse_generic_response(
                response, entity_name, content_type
            )

            return enhanced_caption, entity_info

        except Exception as e:
            logger.error(f"Error generating {content_type} description: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
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
        """Process generic modal content with context support"""
        try:
            # Generate description and entity info
            enhanced_caption, entity_info = await self.generate_description_only(
                modal_content, content_type, item_info, entity_name
            )

            # Build complete content
            modal_chunk = PROMPTS["generic_chunk"].format(
                content_type=content_type.title(),
                content=str(modal_content),
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
            logger.error(f"Error processing {content_type} content: {e}")
            # Fallback processing
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(str(modal_content))}",
                "entity_type": content_type,
                "summary": f"{content_type} content: {str(modal_content)[:100]}",
            }
            return str(modal_content), fallback_entity, [], []

    def _parse_generic_response(
        self, response: str, entity_name: str = None, content_type: str = "content"
    ) -> Tuple[str, Dict[str, Any]]:
        """Parse generic analysis response"""
        try:
            response_data = self._robust_json_parse(response)

            description = response_data.get("detailed_description", "")
            entity_data = response_data.get("entity_info", {})

            if not description or not entity_data:
                raise ValueError("Missing required fields in response")

            if not all(
                key in entity_data for key in ["entity_name", "entity_type", "summary"]
            ):
                raise ValueError("Missing required fields in entity_info")

            entity_data["entity_name"] = (
                entity_data["entity_name"] + f" ({entity_data['entity_type']})"
            )
            if entity_name:
                entity_data["entity_name"] = entity_name

            return description, entity_data

        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            logger.error(f"Error parsing {content_type} analysis response: {e}")
            logger.debug(f"Raw response: {response}")
            fallback_entity = {
                "entity_name": entity_name
                if entity_name
                else f"{content_type}_{compute_mdhash_id(response)}",
                "entity_type": content_type,
                "summary": response[:100] + "..." if len(response) > 100 else response,
            }
            return response, fallback_entity
