"""
Implementation for BGE embedding models using LangChain and Xinference.
"""
from semantic_retrieval.embedding.base import EmbeddingModel
import numpy as np

class BGEEmbedding(EmbeddingModel):
    """Implementation for BGE embedding models using LangChain and Xinference."""
    
    def __init__(self, model_name):
        """Initialize with model name.
        
        Args:
            model_name (str): Name of the BGE model to use (e.g., "bge_large_zh-v1.5" or "bge-m3")
                              This should match the model name in Xinference
        """
        self.model_name = model_name
        self.embedding_model = None
        self.server_url = None
        self.model_uid = None
    
    def initialize(self, server_url="http://localhost:9997", model_uid=None):
        """Initialize and connect to Xinference embedding model.
        
        Args:
            server_url (str): URL of the Xinference server
            model_uid (str): UID of the launched model in Xinference.
                             If None, you'll need to launch the model manually
                             and provide the UID later.
        """
        from langchain_community.embeddings import XinferenceEmbeddings
        
        self.server_url = server_url
        self.model_uid = model_uid
        
        if model_uid:
            self.embedding_model = XinferenceEmbeddings(
                server_url=server_url,
                model_uid=model_uid
            )
    
    def embed(self, text):
        """Embed a single text using BGE via Xinference.
        
        Args:
            text (str): Text to embed
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        if self.embedding_model is None:
            raise ValueError("Model not initialized. Call initialize() with server_url and model_uid first.")
            
        embedding = self.embedding_model.embed_query(text)
        return np.array(embedding)
    
    def embed_batch(self, texts, batch_size=32):
        """Embed a batch of texts using BGE via Xinference.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Batch size for processing (handled by Xinference internally)
            
        Returns:
            numpy.ndarray: Matrix of embedding vectors
        """
        if self.embedding_model is None:
            raise ValueError("Model not initialized. Call initialize() with server_url and model_uid first.")
        
        # XinferenceEmbeddings handles batching internally
        embeddings = self.embedding_model.embed_documents(texts)
        return np.array(embeddings)

