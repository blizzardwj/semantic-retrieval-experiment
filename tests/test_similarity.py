import pytest
import numpy as np
from semantic_retrieval.similarity.calculator import SimilarityCalculator


class TestSimilarityCalculator:
    def test_calculate_similarity_cosine(self):
        """Test cosine similarity calculation between two vectors"""
        calc = SimilarityCalculator(metric='cosine')
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        
        # Perpendicular vectors should have cosine similarity of 0
        assert calc.calculate_similarity(vec_a, vec_b) == pytest.approx(0.0)
        
        # Same direction vectors should have cosine similarity of 1
        vec_c = np.array([2.0, 0.0])  # same direction as vec_a but different magnitude
        assert calc.calculate_similarity(vec_a, vec_c) == pytest.approx(1.0)
        
        # 45-degree angle should have cosine similarity of 0.707
        vec_d = np.array([1.0, 1.0])
        assert calc.calculate_similarity(vec_a, vec_d) == pytest.approx(0.7071, abs=1e-4)
    
    def test_calculate_similarity_dot(self):
        """Test dot product similarity calculation between two vectors"""
        calc = SimilarityCalculator(metric='dot')
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        
        # Perpendicular vectors should have dot product of 0
        assert calc.calculate_similarity(vec_a, vec_b) == pytest.approx(0.0)
        
        # Dot product should be the sum of element-wise products
        vec_c = np.array([2.0, 3.0])
        vec_d = np.array([4.0, 5.0])
        assert calc.calculate_similarity(vec_c, vec_d) == pytest.approx(23.0)  # 2*4 + 3*5 = 8 + 15 = 23
    
    def test_calculate_similarity_batch_cosine(self):
        """Test batch cosine similarity calculation"""
        calc = SimilarityCalculator(metric='cosine')
        query_vec = np.array([1.0, 0.0])
        
        # Create a batch of vectors for comparison
        batch_vecs = np.array([
            [0.0, 1.0],    # perpendicular to query_vec
            [1.0, 0.0],    # same as query_vec
            [1.0, 1.0],    # 45-degree angle from query_vec
            [-1.0, 0.0],   # opposite to query_vec
        ])
        
        similarities = calc.calculate_similarity_batch(query_vec, batch_vecs)
        
        assert len(similarities) == 4
        assert similarities[0] == pytest.approx(0.0)       # perpendicular
        assert similarities[1] == pytest.approx(1.0)       # same direction
        assert similarities[2] == pytest.approx(0.7071, abs=1e-4)  # 45-degree
        assert similarities[3] == pytest.approx(-1.0)      # opposite
    
    def test_calculate_similarity_batch_dot(self):
        """Test batch dot product similarity calculation"""
        calc = SimilarityCalculator(metric='dot')
        query_vec = np.array([2.0, 3.0])
        
        # Create a batch of vectors for comparison
        batch_vecs = np.array([
            [1.0, 1.0],    # dot product: 2*1 + 3*1 = 5
            [0.0, 0.0],    # dot product: 0
            [4.0, 5.0],    # dot product: 2*4 + 3*5 = 8 + 15 = 23
            [-1.0, -1.0],  # dot product: 2*(-1) + 3*(-1) = -5
        ])
        
        similarities = calc.calculate_similarity_batch(query_vec, batch_vecs)
        
        assert len(similarities) == 4
        assert similarities[0] == pytest.approx(5.0)
        assert similarities[1] == pytest.approx(0.0)
        assert similarities[2] == pytest.approx(23.0)
        assert similarities[3] == pytest.approx(-5.0)
    
    def test_invalid_metric(self):
        """Test that an invalid metric raises a ValueError"""
        calc = SimilarityCalculator(metric='invalid_metric')
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        
        with pytest.raises(ValueError) as excinfo:
            calc.calculate_similarity(vec_a, vec_b)
        
        assert "Unsupported similarity metric" in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            calc.calculate_similarity_batch(vec_a, np.array([vec_b]))
        
        assert "Unsupported similarity metric" in str(excinfo.value)
