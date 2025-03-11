"""
Calculates similarity between embeddings.
"""

class SimilarityCalculator:
    """Calculates similarity between embeddings."""
    
    def __init__(self, metric='cosine'):
        """Initialize with similarity metric.
        
        Args:
            metric (str): Similarity metric to use (default: 'cosine')
        """
        self.metric = metric
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate similarity between two embeddings.
        
        Args:
            embedding1 (numpy.ndarray): First embedding vector
            embedding2 (numpy.ndarray): Second embedding vector
            
        Returns:
            float: Similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if self.metric == 'cosine':
            return cosine_similarity(
                embedding1.reshape(1, -1), 
                embedding2.reshape(1, -1)
            )[0][0]
        elif self.metric == 'dot':
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")
    
    def calculate_similarity_batch(self, embedding, embedding_list):
        """Calculate similarity between one embedding and a list of embeddings.
        
        Args:
            embedding (numpy.ndarray): Query embedding vector
            embedding_list (numpy.ndarray): Matrix of embedding vectors
            
        Returns:
            numpy.ndarray: Array of similarity scores
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if self.metric == 'cosine':
            return cosine_similarity(
                embedding.reshape(1, -1), 
                embedding_list
            )[0]
        elif self.metric == 'dot':
            return np.dot(embedding_list, embedding)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")

