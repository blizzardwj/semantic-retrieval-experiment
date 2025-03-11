"""
Abstract base class for retrieval approaches.
"""

from abc import ABC, abstractmethod
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




