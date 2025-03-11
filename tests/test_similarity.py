import pytest
from semantic_retrieval.similarity.calculator import SimilarityCalculator

class TestSimilarity:
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        calc = SimilarityCalculator()
        vec_a = [1.0, 0.0]
        vec_b = [0.0, 1.0]
        assert calc.cosine(vec_a, vec_b) == pytest.approx(0.0)
