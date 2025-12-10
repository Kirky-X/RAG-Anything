import base64
import time
import logging
import io
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Optional imports handling
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import tomllib as toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        toml = None

from raganything.parser.base_parser import Parser
from raganything.llm import build_llm, LLMProviderConfig

logger = logging.getLogger(__name__)

class VlmParser(Parser):
    """
    VlmParser class for understanding and describing image content.
    Uses a VLM (Visual Language Model) via the defined LLM interface.
    """
    
    def __init__(self, config_path: str = "config.toml"):
        super().__init__()
        if not Image:
            raise ImportError("Pillow is required for VlmParser. Install with `pip install Pillow`.")
            
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.llm = self._init_llm()
        
    def _load_config(self, path: Path) -> Dict[str, Any]:
        """Load configuration from TOML file."""
        if not toml:
             logger.warning("tomli or tomllib not found. Cannot load config.toml. Using defaults.")
             return {}
             
        if not path.exists():
             # Try absolute path if relative failed
             abs_path = Path("/home/project/RAG-Anything") / path
             if abs_path.exists():
                 path = abs_path
             else:
                 logger.warning(f"Config file {path} not found. Using defaults.")
                 return {}
             
        try:
            with open(path, "rb") as f:
                return toml.load(f)
        except Exception as e:
            logger.error(f"Failed to parse config file: {e}")
            return {}

    def _init_llm(self):
        """
        Initialize the LLM client based on configuration.

        Returns:
            LLM: An initialized LLM instance configured for VLM tasks.
        """
        rag_config = self.config.get("raganything", {})
        vision_config = rag_config.get("vision", {})
        
        # Fallback to hardcoded defaults if config is missing or empty
        provider = vision_config.get("provider", "ollama")
        model = vision_config.get("model", "qwen3-vl:2b")
        api_base = vision_config.get("api_base", "http://172.24.160.1:11434")
        timeout = vision_config.get("timeout", 10)
        max_retries = vision_config.get("max_retries", 3)
        
        logger.info(f"Initializing VlmParser with model={model}, provider={provider}, base={api_base}")
        
        # Check network connectivity for Ollama
        import socket
        try:
            # Simple connectivity check
            host = api_base.replace("http://", "").replace("https://", "").split(":")[0]
            port = int(api_base.split(":")[-1])
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            result = s.connect_ex((host, port))
            s.close()
            
            if result != 0:
                logger.warning(f"Ollama might be unreachable at {api_base} (connect_ex returned {result}). Proceeding anyway as requested.")
        except Exception as e:
            logger.warning(f"Network check failed: {e}. Proceeding anyway as requested.")

        cfg = LLMProviderConfig(
            provider=provider,
            model=model,
            api_base=api_base,
            timeout=timeout,
            max_retries=max_retries
        )
        return build_llm(cfg)

    def _encode_image(self, image_path: Path, max_size: Optional[int] = None) -> str:
        """
        Encodes image to base64.
        Includes performance optimization: resizing large images.

        Args:
            image_path (Path): Path to the image file.
            max_size (Optional[int]): Maximum dimension (width or height) for resizing. 
                                      If None, original size is kept.

        Returns:
            str: Base64 encoded image string.

        Raises:
            RuntimeError: If image processing fails.
        """
        try:
            with Image.open(image_path) as img:
                # Performance Optimization: Resize if requested
                if max_size:
                    original_size = img.size
                    if max(original_size) > max_size:
                        img.thumbnail((max_size, max_size))
                        logger.debug(f"Resized image from {original_size} to {img.size}")
                    
                # Convert to RGB to ensure compatibility (e.g. remove alpha channel)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                buffered = io.BytesIO()
                # Use JPEG for efficiency unless it's very small
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Failed to process image {image_path}: {e}")

    async def parse_image_async(self, file_path: Union[str, Path], prompt: str = "Describe this image in detail.", **kwargs) -> Dict[str, Any]:
        """
        Asynchronously parse a single image file and return description.
        
        Args:
            file_path (Union[str, Path]): Path to the image file.
            prompt (str): Prompt to send to the VLM. Defaults to "Describe this image in detail.".
            **kwargs: Additional arguments, e.g., 'max_size' for resizing.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - filename: Name of the file.
                - timestamp: Processing timestamp.
                - description: Generated description.
                - latency_seconds: Processing time in seconds.
                - confidence: Confidence score (mocked as 0.95 for now).
                - error: Error message if any.
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        if not file_path.exists():
             # Error handling
             return {
                "filename": file_path.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "",
                "error": f"File not found: {file_path}",
                "latency_seconds": 0.0,
                "confidence": 0.0
            }

        try:
            # Feature Extraction
            max_size = kwargs.get("max_size", 1024) # Default max dimension 1024px
            image_base64 = self._encode_image(file_path, max_size=max_size)
            
            # Description Generation
            # We pass image_data to the LLM call
            response = await self.llm(
                prompt=prompt,
                image_data=image_base64
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Confidence Score (Mocked as LLM API doesn't standardly return this)
            confidence = 0.95 
            
            return {
                "filename": file_path.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": response,
                "latency_seconds": round(latency, 4),
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error parsing image {file_path}: {e}")
            return {
                "filename": file_path.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "",
                "error": str(e),
                "latency_seconds": round(time.time() - start_time, 4),
                "confidence": 0.0
            }

    def parse(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Synchronous wrapper for parsing.
        
        Args:
            file_path (Union[str, Path]): Path to the image file.
            **kwargs: Additional arguments.

        Returns:
            Dict[str, Any]: Parsing result.
        """
        try:
            # Check for existing event loop to avoid conflicts
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
                
            if loop and loop.is_running():
                # If we are already in an async loop, we can't use run_until_complete easily without nesting
                # But since this is a sync method called potentially from sync code, we assume it's fine.
                # Ideally, users should use parse_image_async if they are in async context.
                # For this task, we assume sync execution via script.
                logger.warning("Calling sync parse() from running event loop is risky. Use parse_image_async() instead.")
                # Use a separate thread or task? For now, just try to run it.
                import nest_asyncio
                nest_asyncio.apply()
                result = loop.run_until_complete(self.parse_image_async(file_path, **kwargs))
            else:
                result = asyncio.run(self.parse_image_async(file_path, **kwargs))
                
            return result
        except Exception as e:
            logger.error(f"Fatal error in parse: {e}")
            return {
                "filename": Path(file_path).name,
                "error": str(e)
            }
