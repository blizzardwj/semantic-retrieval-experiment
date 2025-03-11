"""
Implementation of Approach 2: BGE embedding.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.similarity.calculator import SimilarityCalculator

class BGEApproach(RetrievalApproach):
    """Implementation of Approach 2: BGE embedding."""
    
    def __init__(self):
        """Initialize BGE approach."""
        super().__init__("BGE Embedding")
        self.bge_large_model = BGEEmbedding("BAAI/bge-large-zh-v1.5")
        self.bge_m3_model = BGEEmbedding("BAAI/bge-m3")
        self.similarity_calculator = SimilarityCalculator()
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
        self.sentence2_large_embeddings = self.bge_large_model.embed_batch(sentence2_list)
        self.sentence2_m3_embeddings = self.bge_m3_model.embed_batch(sentence2_list)
    
    def retrieve(self, query_sentence, top_k=5):
        """Retrieve top_k similar sentences using BGE embeddings.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        import numpy as np
        
        query_large_embedding = self.bge_large_model.embed(query_sentence)
        query_m3_embedding = self.bge_m3_model.embed(query_sentence)
        
        large_similarities = self.similarity_calculator.calculate_similarity_batch(
            query_large_embedding, self.sentence2_large_embeddings
        )
        m3_similarities = self.similarity_calculator.calculate_similarity_batch(
            query_m3_embedding, self.sentence2_m3_embeddings
        )
        
        # Combine similarities (average of both models)
        combined_similarities = (large_similarities + m3_similarities) / 2
        
        # Get top_k indices and return corresponding sentences
        top_indices = np.argsort(combined_similarities)[-top_k:][::-1]
        top_similarities = combined_similarities[top_indices]
        top_sentences = [self.sentence2_list[i] for i in top_indices]
        
        return top_sentences, top_similarities, top_indices
