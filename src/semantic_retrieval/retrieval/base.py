"""
Abstract base class for retrieval approaches.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from semantic_retrieval.similarity.calculator import SimilarityCalculator
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
import numpy as np
import os

class RetrievalApproach(ABC):
    """Abstract base class for retrieval approaches."""
    
    def __init__(self, name):
        """Initialize with approach name.
        
        Args:
            name (str): Name of the retrieval approach
        """
        self.name = name
    
    def index_sentences(self, sentence2_list):
        """Index sentence2 list for retrieval.
        
        Args:
            sentence2_list (list): List of sentence2 entries
        """
        raise NotImplementedError

    def retrieve(self, query_sentence, top_k=5):
        """Retrieve top_k similar sentences.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        raise NotImplementedError

    def batch_retrieve(self, query_sentences, top_k=5):
        """Retrieve top_k similar sentences for multiple queries at once.

        Args:
            query_sentences (list): List of query sentences
            top_k (int): Number of top results to return for each query
        """
        pass


class InitialRetrievalStrategy(ABC):
    """Abstract base class for initial retrieval strategies."""
    
    @abstractmethod
    def initialize(self, sentences: List[str], embedding_model: Any) -> None:
        """Initialize the strategy with sentences and embedding model.
        
        Args:
            sentences: List of sentences to index
            embedding_model: Embedding model to use
        """
        pass
    
    @abstractmethod
    def retrieve(self, query: str, embedding_model: Any, top_k: int) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences.
        
        Args:
            query: Query sentence
            embedding_model: Embedding model to use
            top_k: Number of top results to return
            
        Returns:
            Tuple of (top_sentences, top_similarities, top_indices)
        """
        pass


class DirectComparisonStrategy(InitialRetrievalStrategy):
    """Strategy for direct comparison using embeddings."""
    
    def __init__(self):
        self.sentence_list = None
        self.sentence_embeddings = None
        self.similarity_calculator = SimilarityCalculator()
    
    def initialize(self, sentences: List[str], embedding_model: Any) -> None:
        """Initialize by embedding all sentences.
        
        Args:
            sentences: List of sentences to index
            embedding_model: Embedding model to use
        """
        self.sentence_list = sentences
        self.sentence_embeddings = embedding_model.embed_batch(sentences)
    
    def retrieve(self, query: str, embedding_model: Any, top_k: int) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences using direct embedding comparison.
        
        Args:
            query: Query sentence
            embedding_model: Embedding model to use
            top_k: Number of top results to return
            
        Returns:
            Tuple of (top_sentences, top_similarities, top_indices)
        """
        query_embedding = embedding_model.embed(query)
        
        similarities = self.similarity_calculator.calculate_similarity_batch(
            query_embedding, self.sentence_embeddings
        )
        
        # Get top_k indices and corresponding sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        top_sentences = [self.sentence_list[i] for i in top_indices]
        
        return top_sentences, top_similarities.tolist(), top_indices.tolist()


class VectorStoreStrategy(InitialRetrievalStrategy):
    """Strategy for retrieval using vector store."""
    
    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.vector_store = None
        self.sentence_list = None
    
    def initialize(self, sentences: List[str], embedding_model: Any) -> None:
        """Initialize by creating a vector store.
        
        Args:
            sentences: List of sentences to index
            embedding_model: Embedding model to use
        """
        self.sentence_list = sentences
        
        # Convert sentences to Langchain Document objects
        documents = [
            Document(
                page_content=sentence,
                metadata={"id": str(i), "source": f"sentence_{i}"}
            ) for i, sentence in enumerate(sentences)
        ]
        
        # Create the Chroma vector store
        if self.persist_directory:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model
            )
    
    def retrieve(self, query: str, embedding_model: Any, top_k: int) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences using vector store.
        
        Args:
            query: Query sentence
            embedding_model: Embedding model to use
            top_k: Number of top results to return
            
        Returns:
            Tuple of (top_sentences, top_similarities, top_indices)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call initialize() first.")
        
        # Search the vector store to get initial candidates
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )
        
        # Extract results
        top_sentences = []
        top_similarities = []
        top_indices = []
        
        for doc, score in results:
            # Convert score to similarity (Chroma returns distance, not similarity)
            # Assuming cosine distance, similarity = 1 - score
            similarity = 1 - score
            top_sentences.append(doc.page_content)
            top_similarities.append(similarity)
            top_indices.append(int(doc.metadata["id"]))
        
        return top_sentences, top_similarities, top_indices

