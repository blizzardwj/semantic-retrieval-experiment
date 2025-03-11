import pytest
from semantic_retrieval.embedding import BGEEncoder

class TestEmbeddingModels:
    def test_bge_encoder_init(self):
        """Test BGE encoder initialization"""
        encoder = BGEEncoder()
        assert encoder is not None
