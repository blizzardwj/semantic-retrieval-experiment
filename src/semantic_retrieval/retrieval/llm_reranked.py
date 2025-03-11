"""
Implementation of Approach 3: BGE embedding with LLM reranking.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.similarity.calculator import SimilarityCalculator
from semantic_retrieval.llm.model import LLMModel

class LLMRerankedBGEApproach(RetrievalApproach):
    """Implementation of Approach 3: BGE embedding with LLM reranking."""
    
    def __init__(self):
        """Initialize LLM-reranked BGE approach."""
        super().__init__("LLM-Reranked BGE")
        self.bge_large_model = BGEEmbedding("BAAI/bge-large-zh-v1.5")
        self.bge_m3_model = BGEEmbedding("BAAI/bge-m3")
        self.similarity_calculator = SimilarityCalculator()
        self.llm_model = LLMModel("gpt-3.5-turbo")  # This could be changed to a different LLM
        self.sentence2_large_embeddings = None
        self.sentence2_m3_embeddings = None
        self.sentence2_list = None
    
    def index_sentences(self, sentence2_list):
        """Index sentence2 list by embedding them with both models.
        
        Args:
            sentence2_list (list): List of sentence2 entries
        """
        self.sentence2_list = sentence2_list
        self.bge_large_model.initialize()
        self.bge_m3_model.initialize()
        self.llm_model.initialize()
        self.sentence2_large_embeddings = self.bge_large_model.embed_batch(sentence2_list)
        self.sentence2_m3_embeddings = self.bge_m3_model.embed_batch(sentence2_list)
    
    def retrieve(self, query_sentence, top_k=5, initial_top_k=10):
        """Retrieve top_k similar sentences using LLM reranking.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            initial_top_k (int): Number of initial candidates for LLM reranking
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        import numpy as np
        
        # First get initial_top_k candidates using BGE embeddings
        query_large_embedding = self.bge_large_model.embed(query_sentence)
        query_m3_embedding = self.bge_m3_model.embed(query_sentence)
        
        large_similarities = self.similarity_calculator.calculate_similarity_batch(
            query_large_embedding, self.sentence2_large_embeddings
        )
        m3_similarities = self.similarity_calculator.calculate_similarity_batch(
            query_m3_embedding, self.sentence2_m3_embeddings
        )
        
        # Combine similarities
        combined_similarities = (large_similarities + m3_similarities) / 2
        
        # Get initial_top_k indices and corresponding sentences
        initial_top_indices = np.argsort(combined_similarities)[-initial_top_k:][::-1]
        initial_top_similarities = combined_similarities[initial_top_indices]
        initial_top_sentences = [self.sentence2_list[i] for i in initial_top_indices]
        
        # Use LLM to rerank
        reranked_indices = self.llm_model.rerank(
            query_sentence, initial_top_sentences, initial_top_indices, top_k
        )
        
        # Get final top_k sentences and their similarities
        final_top_indices = [initial_top_indices[i] for i in reranked_indices[:top_k]]
        final_top_similarities = [initial_top_similarities[i] for i in reranked_indices[:top_k]]
        final_top_sentences = [initial_top_sentences[i] for i in reranked_indices[:top_k]]
        
        return final_top_sentences, final_top_similarities, final_top_indices
