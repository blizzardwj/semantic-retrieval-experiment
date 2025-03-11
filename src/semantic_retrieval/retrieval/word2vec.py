"""
Implementation of Approach 1: Word2Vec (FastText) embedding.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.fasttext import FastTextEmbedding
from semantic_retrieval.similarity.calculator import SimilarityCalculator

class Word2VecApproach(RetrievalApproach):
    """Implementation of Approach 1: Word2Vec (FastText) embedding."""
    
    def __init__(self):
        """Initialize Word2Vec approach."""
        super().__init__("Word2Vec (FastText)")
        self.embedding_model = FastTextEmbedding()
        self.similarity_calculator = SimilarityCalculator()
        self.sentence2_embeddings = None
        self.sentence2_list = None
    
    def index_sentences(self, sentence2_list):
        """Index sentence2 list by embedding them.
        
        Args:
            sentence2_list (list): List of sentence2 entries
        """
        import numpy as np
        
        self.sentence2_list = sentence2_list
        self.embedding_model.initialize()
        self.sentence2_embeddings = self.embedding_model.embed_batch(sentence2_list)
    
    def retrieve(self, query_sentence, top_k=5):
        """Retrieve top_k similar sentences using Word2Vec embeddings.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        import numpy as np
        
        query_embedding = self.embedding_model.embed(query_sentence)
        similarities = self.similarity_calculator.calculate_similarity_batch(
            query_embedding, self.sentence2_embeddings
        )
        
        # Get top_k indices and return corresponding sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        top_sentences = [self.sentence2_list[i] for i in top_indices]
        
        return top_sentences, top_similarities, top_indices
