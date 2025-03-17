"""
Implementation for BGE embedding models using LangChain and Xinference.
"""
from semantic_retrieval.embedding.base import EmbeddingModel
import numpy as np
from typing import Optional, List, Union, Any

class BGEEmbedding(EmbeddingModel):
    """Implementation for BGE embedding models using LangChain and Xinference."""
    
    def __init__(self, model_name: str):
        """Initialize with model name.
        
        Args:
            model_name (str): Name of the BGE model to use (e.g., "bge_large_zh-v1.5" or "bge-m3")
                              This should match the model name in Xinference
        """
        self.model_name = model_name    # general model name
        self.model_uid = None   # specified for xinference client
        self.embedding_model = None
        self.server_url = None  
    
    def initialize(self, server_url: str, model_uid: Optional[str] = None) -> None:
        """Initialize and connect to Xinference embedding model.
        
        Args:
            server_url (str): URL of the Xinference server
            model_uid (str, optional): UID of the launched model in Xinference.
                             If None, model_name will be used as model_uid.
        """
        from langchain_community.embeddings import XinferenceEmbeddings
        
        self.server_url = server_url
        if model_uid is None:
            self.model_uid = self.model_name
        else:
            self.model_uid = model_uid
        
        try:
            self.embedding_model = XinferenceEmbeddings(
                server_url=server_url,
                model_uid=self.model_uid
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize BGE embedding model: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure the model is initialized before use."""
        if self.embedding_model is None:
            raise ValueError("Model not initialized. Call initialize() with server_url and model_uid first.")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using BGE via Xinference.
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            ValueError: If model is not initialized
            Exception: If embedding fails
        """
        self._ensure_initialized()
            
        try:
            embedding = self.embedding_model.embed_query(text)
            return embedding
        except Exception as e:
            raise Exception(f"Error embedding query: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using BGE via Xinference.
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            List[List[float]]: Matrix of embedding vectors
            
        Raises:
            ValueError: If model is not initialized
            Exception: If embedding fails
        """
        self._ensure_initialized()
        
        try:
            # XinferenceEmbeddings handles batching internally
            embeddings = self.embedding_model.embed_documents(texts)
            return embeddings
        except Exception as e:
            raise Exception(f"Error embedding documents: {e}")
