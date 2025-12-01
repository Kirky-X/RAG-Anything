import asyncio
import logging
from typing import Any, List, Optional, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult

logger = logging.getLogger(__name__)

class RobustOllamaClient:
    """
    A wrapper around ChatOllama that provides connection management,
    retries, and error handling.
    """
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize the RobustOllamaClient.

        Args:
            model (str): The name of the Ollama model to use.
            base_url (str): The base URL of the Ollama server.
            timeout (float): Request timeout in seconds. Defaults to 30.0.
            max_retries (int): Maximum number of retry attempts. Defaults to 3.
            **kwargs: Additional arguments passed to ChatOllama.
        """
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize the underlying ChatOllama client
        # We pass other kwargs to ChatOllama (e.g. temperature)
        self.client = ChatOllama(
            model=model,
            base_url=base_url,
            timeout=timeout,
            **kwargs
        )
        
    async def ainvoke(self, messages: List[BaseMessage], **kwargs) -> Any:
        """
        Invoke the model asynchronously with retry logic.

        Args:
            messages (List[BaseMessage]): List of messages to send to the model.
            **kwargs: Additional arguments passed to the underlying client's ainvoke method.

        Returns:
            Any: The response from the model.

        Raises:
            Exception: The last exception encountered if all retries fail.
            RuntimeError: If an unknown error occurs.
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Attempt to invoke the client
                return await self.client.ainvoke(messages, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Ollama request failed (attempt {attempt + 1}/{self.max_retries + 1}). "
                    f"Error: {str(e)}"
                )
                
                # Determine if we should retry
                if attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s...
                    sleep_time = 1 * (2 ** attempt)
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for Ollama request. Last error: {str(e)}")
        
        # If we exhaust retries, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Unknown error in RobustOllamaClient")

    # Proxy other methods if needed, but ainvoke is the main one used by LLM class
