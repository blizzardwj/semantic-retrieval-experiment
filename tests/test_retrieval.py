import pytest
from semantic_retrieval.retrieval.base import BaseRetriever

class TestRetrieval:
    def test_base_retriever(self):
        """Test base retriever functionality"""
        retriever = BaseRetriever()
        assert retriever.retrieve("test query") == []
