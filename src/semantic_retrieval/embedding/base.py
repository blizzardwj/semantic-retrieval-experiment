"""
Abstract base class for embedding models.
"""
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    def initialize(self):
        """Initialize and load the model."""
        raise NotImplementedError
    
    def embed(self, text):
        """Embed a single text.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: The embedding vector
        """
        raise NotImplementedError
    
    def embed_batch(self, texts):
        """Embed a batch of texts.
        
        Args:
            texts (list): List of texts to embed
            
        Returns:
            numpy.ndarray: Matrix of embedding vectors
        """
        raise NotImplementedError







