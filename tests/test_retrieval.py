import pytest
import os
import shutil
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.retrieval.word2vec import Word2VecApproach
from semantic_retrieval.retrieval.bge import BGEApproach
from semantic_retrieval.retrieval.llm_reranked import LLMRerankedBGEApproach

class TestRetrieval:
    def test_base_retriever(self):
        """Test base retriever functionality"""
        # Create a minimal subclass of RetrievalApproach for testing
        class MinimalRetriever(RetrievalApproach):
            def retrieve(self, query_sentence, top_k=5):
                return []
        
        retriever = MinimalRetriever("Test")
        assert retriever.retrieve("test query") == []
    
    @pytest.fixture
    def sample_sentences(self):
        """Fixture for sample sentences to be indexed"""
        test_dataset_path = Path("tests/test_data/test_dataset.csv")
        test_dataset = pd.read_csv(test_dataset_path)
        return test_dataset["sentence2"].tolist()
    
    @pytest.fixture
    def test_query(self):
        """Fixture for test query"""
        return "人工智能如何应用于自然语言处理"
    
    @pytest.fixture(scope="class")
    def cleanup_vector_stores(self):
        """Fixture to clean up vector store directories before and after tests"""
        # Setup - clean up any existing directories from previous interrupted runs
        test_dirs = [
            "test_persist_word2vec",
            "test_persist_bge_large",
            "test_persist_bge_m3",
            "test_persist_llm_rerank"
        ]
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                
        yield
        # Teardown - clean up directories
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
    
    @pytest.mark.integration
    def test_word2vec_vector_store_retrieval(self, sample_sentences, test_query, cleanup_vector_stores):
        """Integration test for Word2Vec retrieval using vector store"""
        # Create Word2Vec retriever with vector store
        persist_dir = "test_persist_word2vec"
        retriever = Word2VecApproach(persist_directory=persist_dir, use_vector_store=True)
        
        # Index sentences
        retriever.index_sentences(sample_sentences)
        
        # Retrieve top 3 results
        top_k = 3
        results, similarities, indices = retriever.retrieve(test_query, top_k=top_k)
        
        # Verify results
        assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
        assert len(similarities) == len(results), "Results and similarities should have same length"
        assert len(indices) == len(results), "Results and indices should have same length"
        
        # Verify similarities are in descending order
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), "Similarities should be in descending order"
        
        # Verify persistence directory was created
        assert os.path.exists(persist_dir), f"Persistence directory {persist_dir} should exist"
    
    @pytest.mark.integration
    def test_bge_large_vector_store_retrieval(self, sample_sentences, test_query, cleanup_vector_stores):
        """Integration test for BGE-large-zh retrieval using vector store"""
        # Create BGE retriever with vector store
        persist_dir = "test_persist_bge_large"
        model_name = "bge-large-zh-v1.5"
        server_url = "http://20.30.80.200:9997"
        retriever = BGEApproach(
            model_name=model_name,
            server_url=server_url,
            persist_directory=persist_dir,
            use_vector_store=True
        )
        
        # Index sentences
        retriever.index_sentences(sample_sentences)
        
        # Retrieve top 3 results
        top_k = 3
        results, similarities, indices = retriever.retrieve(test_query, top_k=top_k)
        
        # Verify results
        assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
        assert len(similarities) == len(results), "Results and similarities should have same length"
        assert len(indices) == len(results), "Results and indices should have same length"
        
        # Verify similarities are in descending order
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), "Similarities should be in descending order"
        
        # Verify persistence directory was created
        assert os.path.exists(persist_dir), f"Persistence directory {persist_dir} should exist"
    
    @pytest.mark.integration
    def test_bge_m3_vector_store_retrieval(self, sample_sentences, test_query, cleanup_vector_stores):
        """Integration test for BGE-M3 retrieval using vector store"""
        # Create BGE-M3 retriever with vector store
        persist_dir = "test_persist_bge_m3"
        model_name = "bge-m3"
        server_url = "http://20.30.80.200:9997"
        retriever = BGEApproach(
            model_name=model_name,
            server_url=server_url,
            persist_directory=persist_dir,
            use_vector_store=True
        )
        
        # Index sentences
        retriever.index_sentences(sample_sentences)
        
        # Retrieve top 3 results
        top_k = 3
        results, similarities, indices = retriever.retrieve(test_query, top_k=top_k)
        
        # Verify results
        assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
        assert len(similarities) == len(results), "Results and similarities should have same length"
        assert len(indices) == len(results), "Results and indices should have same length"
        
        # Verify similarities are in descending order
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), "Similarities should be in descending order"
        
        # Verify persistence directory was created
        assert os.path.exists(persist_dir), f"Persistence directory {persist_dir} should exist"
    
    @pytest.mark.integration
    def test_llm_reranked_vector_store_retrieval(self, sample_sentences, test_query, cleanup_vector_stores):
        """Integration test for LLM reranked BGE retrieval using vector store"""
        # Create LLM reranked retriever with vector store
        persist_dir = "test_persist_llm_rerank"
        model_name = "bge-large-zh-v1.5"
        server_url = "http://20.30.80.200:9997"
        # LLM Reranked BGE needs more parameters than BGE approach
        retriever = LLMRerankedBGEApproach(
            embedding_model=model_name,
            embedding_server_url=server_url,
            llm_model="deepseek-r1-distill-qwen", 
            llm_server_url=server_url,
            persist_directory=persist_dir,
            use_vector_store=True
        )
        
        # Index sentences
        retriever.index_sentences(sample_sentences)
        
        # Retrieve top 3 results (with initial top 5 for reranking)
        top_k = 3
        initial_top_k = 5
        results, similarities, indices = retriever.retrieve(test_query, top_k=top_k, initial_top_k=initial_top_k)
        
        # Verify results
        assert len(results) <= top_k, f"Expected at most {top_k} results, got {len(results)}"
        assert len(similarities) == len(results), "Results and similarities should have same length"
        assert len(indices) == len(results), "Results and indices should have same length"
        
        # Verify similarities are in descending order (might not be strictly true after LLM reranking)
        # but we keep it as a sanity check
        if len(similarities) > 1:
            assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1)), "Similarities should be in descending order"
        
        # Verify persistence directory was created
        assert os.path.exists(persist_dir), f"Persistence directory {persist_dir} should exist"
