"""
Implementation of Approach 3: BGE embedding with LLM reranking.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.llm.model import LLMModel
from typing import List, Tuple, Optional
from semantic_retrieval.retrieval.base import InitialRetrievalStrategy, DirectComparisonStrategy, VectorStoreStrategy


class LLMRerankedBGEApproach(RetrievalApproach):
    """Implementation of Approach 3: BGE embedding with LLM reranking."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", persist_directory: Optional[str] = None, 
                 use_vector_store: bool = False):
        """Initialize LLM-reranked BGE approach.
        
        Args:
            model_name (str): Name of the BGE model to use
            persist_directory (str, optional): Directory to persist vector store. If None, 
                                              an in-memory vector store will be used.
            use_vector_store (bool): Whether to use vector store for initial retrieval
        """
        super().__init__("LLM-Reranked BGE")
        self.embedding_model = BGEEmbedding(model_name)
        self.llm_model = LLMModel("gpt-3.5-turbo")  # This could be changed to a different LLM
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
        self.llm_model.initialize()
        
        # Switch strategy if requested
        if use_vector_store is not None:
            if use_vector_store:
                self.retrieval_strategy = VectorStoreStrategy(self.persist_directory)
            else:
                self.retrieval_strategy = DirectComparisonStrategy()
        
        # Initialize the strategy with the sentences
        self.retrieval_strategy.initialize(sentence_list, self.embedding_model)
    
    def retrieve(self, query_sentence: str, top_k: int = 5, initial_top_k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences using initial retrieval and LLM reranking.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            initial_top_k (int): Number of initial candidates for LLM reranking
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        # First get initial_top_k candidates using the current retrieval strategy
        initial_top_sentences, initial_top_similarities, initial_top_indices = self.retrieval_strategy.retrieve(
            query_sentence, self.embedding_model, initial_top_k
        )
        
        # Use LLM to rerank
        reranked_indices = self.llm_model.rerank(
            query_sentence, initial_top_sentences, initial_top_indices, top_k
        )
        
        # Get final top_k sentences and their similarities
        final_top_indices = [initial_top_indices[i] for i in reranked_indices[:top_k]]
        final_top_similarities = [initial_top_similarities[i] for i in reranked_indices[:top_k]]
        final_top_sentences = [initial_top_sentences[i] for i in reranked_indices[:top_k]]
        
        return final_top_sentences, final_top_similarities, final_top_indices
    
    def batch_retrieve(self, query_sentences: List[str], top_k: int = 5, initial_top_k: int = 10) -> List[Tuple[List[str], List[float], List[int]]]:
        """Retrieve top_k similar sentences for multiple queries.
        
        Args:
            query_sentences (list): List of query sentences
            top_k (int): Number of top results to return for each query
            initial_top_k (int): Number of initial candidates for LLM reranking
            
        Returns:
            list: List of tuples (top_sentences, top_similarities, top_indices) for each query
        """
        results = []
        
        for query in query_sentences:
            query_result = self.retrieve(query, top_k, initial_top_k)
            results.append(query_result)
        
        return results
    