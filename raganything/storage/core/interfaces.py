from abc import ABC, abstractmethod
from typing import Dict, List

class StorageBackend(ABC):
    """
    Abstract base class for storage backends as defined in section 4.1 of the design document.
    """

    @abstractmethod
    async def store_file(self, file_path: str, content: bytes, metadata: Dict) -> str:
        """
        Store a file and return a unique identifier.
        
        Args:
            file_path: The logical path or name for the file.
            content: The binary content of the file.
            metadata: A dictionary containing metadata for the file.
            
        Returns:
            str: The unique identifier (e.g., UUID or object key) for the stored file.
        """
        pass
    
    @abstractmethod
    async def retrieve_file(self, file_id: str) -> bytes:
        """
        Retrieve file content by its identifier.
        
        Args:
            file_id: The unique identifier of the file.
            
        Returns:
            bytes: The content of the file.
        """
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by its identifier.
        
        Args:
            file_id: The unique identifier of the file.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """
        List files matching a prefix.
        
        Args:
            prefix: The prefix to filter files.
            
        Returns:
            List[str]: A list of file identifiers.
        """
        pass
    
    @abstractmethod
    async def get_metadata(self, file_id: str) -> Dict:
        """
        Get metadata for a file.
        
        Args:
            file_id: The unique identifier of the file.
            
        Returns:
            Dict: The metadata dictionary.
        """
        pass
    
    @abstractmethod
    async def update_metadata(self, file_id: str, metadata: Dict) -> bool:
        """
        Update metadata for a file.
        
        Args:
            file_id: The unique identifier of the file.
            metadata: The new metadata dictionary (or updates to apply).
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        pass
