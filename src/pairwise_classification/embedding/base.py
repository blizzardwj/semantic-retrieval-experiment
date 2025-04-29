"""
Abstract base class for embedding models.
"""
from langchain_core.embeddings import Embeddings
from abc import ABC, abstractmethod

class EmbeddingModel(Embeddings, ABC):
    """Abstract base class for embedding models."""
    
    def initialize(self):
        """Initialize and load the model."""
        raise NotImplementedError
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list[float]: The embedding vector
        """
        raise NotImplementedError
    
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            list[list[float]]: Matrix of embedding vectors
        """
        raise NotImplementedError







