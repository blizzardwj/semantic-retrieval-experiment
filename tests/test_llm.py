import pytest
from semantic_retrieval.llm.model import LLM

class TestLLMIntegration:
    def test_llm_initialization(self):
        """Test LLM model initialization"""
        model = LLM()
        assert model is not None
