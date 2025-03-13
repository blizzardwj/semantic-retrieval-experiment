"""
Implementation of Approach 1: Word2Vec (FastText) embedding.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.fasttext import FastTextEmbedding
from typing import List, Tuple, Optional
from semantic_retrieval.retrieval.base import InitialRetrievalStrategy, DirectComparisonStrategy, VectorStoreStrategy


class Word2VecApproach(RetrievalApproach):
    """Implementation of Approach 1: Word2Vec (FastText) embedding."""
    
    def __init__(self, persist_directory: Optional[str] = None, use_vector_store: bool = False):
        """Initialize Word2Vec approach.
        
        Args:
            persist_directory (str, optional): Directory to persist vector store. If None, 
                                              an in-memory vector store will be used.
            use_vector_store (bool): Whether to use vector store for initial retrieval
        """
        super().__init__("Word2Vec (FastText)")
        self.embedding_model = FastTextEmbedding()
        self.persist_directory = persist_directory
        self.sentence_list = None
        
        # Set the initial retrieval strategy based on the use_vector_store flag
        if use_vector_store:
            self.retrieval_strategy = VectorStoreStrategy(persist_directory)
        else:
            self.retrieval_strategy = DirectComparisonStrategy()
    
    def set_retrieval_strategy(self, strategy: InitialRetrievalStrategy) -> None:
        """Set the initial retrieval strategy.
        
        Args:
            strategy: The retrieval strategy to use
        """
        self.retrieval_strategy = strategy
        # Re-initialize if we already have sentences
        if self.sentence_list:
            self.retrieval_strategy.initialize(self.sentence_list, self.embedding_model)
    
    def index_sentences(self, sentence_list: List[str], use_vector_store: Optional[bool] = None) -> None:
        """Index sentence list using the current retrieval strategy or optionally switch strategy.
        
        Args:
            sentence_list (list): List of sentence entries
            use_vector_store (bool, optional): If provided, switches to vector store strategy (True)
                                              or direct comparison strategy (False)
        """
        self.sentence_list = sentence_list
        self.embedding_model.initialize()
        
        # Switch strategy if requested
        if use_vector_store is not None:
            if use_vector_store:
                self.retrieval_strategy = VectorStoreStrategy(self.persist_directory)
            else:
                self.retrieval_strategy = DirectComparisonStrategy()
        
        # Initialize the strategy with the sentences
        self.retrieval_strategy.initialize(sentence_list, self.embedding_model)
    
    def retrieve(self, query_sentence: str, top_k: int = 5) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences using the current retrieval strategy.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        return self.retrieval_strategy.retrieve(query_sentence, self.embedding_model, top_k)
    
    def batch_retrieve(self, query_sentences: List[str], top_k: int = 5) -> List[Tuple[List[str], List[float], List[int]]]:
        """Retrieve top_k similar sentences for multiple queries.
        
        Args:
            query_sentences (list): List of query sentences
            top_k (int): Number of top results to return for each query
            
        Returns:
            list: List of tuples (top_sentences, top_similarities, top_indices) for each query
        """
        results = []
        
        for query in query_sentences:
            query_result = self.retrieve(query, top_k)
            results.append(query_result)
        
        return results
