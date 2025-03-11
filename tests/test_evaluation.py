import pytest
from semantic_retrieval.evaluation.evaluator import Evaluator

class TestEvaluationMetrics:
    def test_evaluator_init(self):
        """Test evaluator initialization"""
        evaluator = Evaluator()
        assert evaluator is not None
