"""
Implementation for FastText embedding model.
"""
from semantic_retrieval.embedding.base import EmbeddingModel

class FastTextEmbedding(EmbeddingModel):
    """Implementation for FastText embedding model."""
    
    def __init__(self, model_path=None):
        """Initialize with optional model path.
        
        Args:
            model_path (str, optional): Path to pre-trained FastText model
        """
        self.model_path = model_path
        self.model = None
    
    def initialize(self):
        """Initialize and load FastText model."""
        import fasttext
        if self.model_path:
            self.model = fasttext.load_model(self.model_path)
        else:
            # Download or use a default Chinese FastText model
            # This is a placeholder - actual implementation would depend on specific requirements
            # self.model = fasttext.load_model('cc.zh.300.bin')
            raise ValueError("FastText model path is required")
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single text using FastText.
        
        Args:
            text (str): Text to embed
            
        Returns:
            list[float]: The embedding vector
        """
        if self.model is None:
            self.initialize()
        vector = self.model.get_sentence_vector(text)
        return vector.tolist()
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using FastText.
        
        Args:
            texts (list[str]): List of texts to embed
            
        Returns:
            list[list[float]]: Matrix of embedding vectors
        """
        import numpy as np
        if self.model is None:
            self.initialize()
        vectors = [self.model.get_sentence_vector(text) for text in texts]
        return [vector.tolist() for vector in vectors]
