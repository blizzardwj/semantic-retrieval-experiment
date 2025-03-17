import pytest
import os
import numpy as np
import tempfile
import json
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from semantic_retrieval.evaluation.evaluator import Evaluator
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.retrieval.word2vec import Word2VecApproach
from semantic_retrieval.retrieval.bge import BGEApproach
from semantic_retrieval.retrieval.llm_reranked import LLMRerankedBGEApproach


class TestEvaluationMetrics:
    def test_evaluator_init(self):
        """Test evaluator initialization"""
        evaluator = Evaluator()
        assert evaluator is not None
    
    def test_evaluate_perfect_retrieval(self):
        """Test evaluation with perfect retrieval results"""
        evaluator = Evaluator()
        
        # Create test data
        # Assume we have 10 sentences, 5 of which are relevant (label=1)
        ground_truth_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sentence2_indices = list(range(10))
        
        # Perfect retrieval: all relevant items are retrieved first
        retrieved_indices_list = [[0, 2, 4, 6, 8]]  # All relevant indices
        
        # Evaluate
        metrics = evaluator.evaluate(retrieved_indices_list, ground_truth_labels, sentence2_indices)
        
        # Check metrics
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["ndcg"] == 1.0
    
    def test_evaluate_worst_retrieval(self):
        """Test evaluation with worst retrieval results"""
        evaluator = Evaluator()
        
        # Create test data
        # Assume we have 10 sentences, 5 of which are relevant (label=1)
        ground_truth_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sentence2_indices = list(range(10))
        
        # Worst retrieval: only irrelevant items are retrieved
        retrieved_indices_list = [[1, 3, 5, 7, 9]]  # All irrelevant indices
        
        # Evaluate
        metrics = evaluator.evaluate(retrieved_indices_list, ground_truth_labels, sentence2_indices)
        
        # Check metrics
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["ndcg"] == 0.0
    
    def test_evaluate_mixed_retrieval(self):
        """Test evaluation with mixed retrieval results"""
        evaluator = Evaluator()
        
        # Create test data
        # Assume we have 10 sentences, 5 of which are relevant (label=1)
        ground_truth_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sentence2_indices = list(range(10))
        
        # Mixed retrieval: some relevant, some irrelevant
        retrieved_indices_list = [[0, 1, 2, 3, 4]]  # 3 relevant, 2 irrelevant
        
        # Evaluate
        metrics = evaluator.evaluate(retrieved_indices_list, ground_truth_labels, sentence2_indices)
        
        # Check metrics
        assert metrics["precision"] == 0.6
        assert abs(metrics["recall"] - 0.6) < 1e-6  # 3/5 = 0.6
        assert abs(metrics["f1"] - 0.6) < 1e-6  # 2*0.6*0.6/(0.6+0.6) = 0.6
        assert metrics["mrr"] == 1.0  # First item is relevant
        
        # NDCG calculation: DCG = 1/log2(1+1) + 0/log2(2+1) + 1/log2(3+1) + 0/log2(4+1) + 1/log2(5+1)
        # IDCG = 1/log2(1+1) + 1/log2(2+1) + 1/log2(3+1) + 0/log2(4+1) + 0/log2(5+1)
        # Actual calculated NDCG ≈ 0.8855
        assert abs(metrics["ndcg"] - 0.8855) < 0.001
    
    def test_evaluate_multiple_queries(self):
        """Test evaluation with multiple queries"""
        evaluator = Evaluator()
        
        # Create test data
        # Assume we have 10 sentences, 5 of which are relevant (label=1)
        ground_truth_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        sentence2_indices = list(range(10))
        
        # Multiple queries with different retrieval results
        retrieved_indices_list = [
            [0, 2, 4, 6, 8],  # Perfect retrieval
            [1, 3, 5, 7, 9],  # Worst retrieval
            [0, 1, 2, 3, 4]   # Mixed retrieval
        ]
        
        # Evaluate
        metrics = evaluator.evaluate(retrieved_indices_list, ground_truth_labels, sentence2_indices)
        
        # Check metrics (average of perfect, worst, and mixed)
        assert abs(metrics["precision"] - (1.0 + 0.0 + 0.6) / 3) < 1e-6
        assert abs(metrics["recall"] - (1.0 + 0.0 + 0.6) / 3) < 1e-6
        assert abs(metrics["f1"] - (1.0 + 0.0 + 0.6) / 3) < 1e-6
        assert abs(metrics["mrr"] - (1.0 + 0.0 + 1.0) / 3) < 1e-6
        # NDCG is more complex, just check it's between 0 and 1
        assert 0 <= metrics["ndcg"] <= 1
    
    def test_compare_approaches(self):
        """Test comparing different approaches"""
        evaluator = Evaluator()
        
        # Create test results for different approaches
        results = {
            "approach1": {
                "evaluation": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "mrr": 0.9,
                    "ndcg": 0.85
                }
            },
            "approach2": {
                "evaluation": {
                    "precision": 0.6,
                    "recall": 0.8,
                    "f1": 0.69,
                    "mrr": 0.7,
                    "ndcg": 0.75
                }
            }
        }
        
        # Compare approaches
        comparison = evaluator.compare_approaches(results)
        
        # Check comparison
        assert comparison["approach1"]["precision"] == 0.8
        assert comparison["approach1"]["recall"] == 0.7
        assert comparison["approach1"]["f1"] == 0.75
        assert comparison["approach1"]["mrr"] == 0.9
        assert comparison["approach1"]["ndcg"] == 0.85
        
        assert comparison["approach2"]["precision"] == 0.6
        assert comparison["approach2"]["recall"] == 0.8
        assert comparison["approach2"]["f1"] == 0.69
        assert comparison["approach2"]["mrr"] == 0.7
        assert comparison["approach2"]["ndcg"] == 0.75


class TestEvaluatorMethods:
    def test_save_and_load_results(self):
        """Test saving and loading evaluation results"""
        evaluator = Evaluator()
        
        # Create test results
        results = {
            "approach1": {
                "approach_name": "approach1",
                "retrieved_results": [["result1", "result2"]],
                "retrieved_similarities": [[0.9, 0.8]],
                "retrieved_indices": [[0, 1]],
                "evaluation": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1": 0.75,
                    "mrr": 0.9,
                    "ndcg": 0.85
                }
            }
        }
        
        # Use a temporary directory for the test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save results
            filepath = evaluator.save_results(results, output_dir=temp_dir)
            
            # Check file exists
            assert os.path.exists(filepath)
            
            # Load results
            loaded_results = evaluator.load_results(filepath)
            
            # Check loaded results match original
            assert loaded_results["approach1"]["approach_name"] == "approach1"
            assert loaded_results["approach1"]["retrieved_results"] == [["result1", "result2"]]
            assert loaded_results["approach1"]["retrieved_similarities"] == [[0.9, 0.8]]
            assert loaded_results["approach1"]["retrieved_indices"] == [[0, 1]]
            assert loaded_results["approach1"]["evaluation"]["precision"] == 0.8
            assert loaded_results["approach1"]["evaluation"]["recall"] == 0.7
            assert loaded_results["approach1"]["evaluation"]["f1"] == 0.75
            assert loaded_results["approach1"]["evaluation"]["mrr"] == 0.9
            assert loaded_results["approach1"]["evaluation"]["ndcg"] == 0.85
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_visualize_comparison(self, mock_show, mock_savefig):
        """Test visualization of comparison results"""
        evaluator = Evaluator()
        
        # Create test comparison
        comparison = {
            "approach1": {
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "mrr": 0.9,
                "ndcg": 0.85
            },
            "approach2": {
                "precision": 0.6,
                "recall": 0.8,
                "f1": 0.69,
                "mrr": 0.7,
                "ndcg": 0.75
            }
        }
        
        # Test visualization without saving
        evaluator.visualize_comparison(comparison)
        mock_show.assert_called_once()
        
        # Test visualization with saving
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            evaluator.visualize_comparison(comparison, output_path=temp_file.name)
            mock_savefig.assert_called_once_with(temp_file.name)


class MockRetriever(RetrievalApproach):
    """Mock retriever for testing"""
    
    def __init__(self, name, results_to_return):
        super().__init__(name)
        self.results_to_return = results_to_return
        self.indexed = False
    
    def index_sentences(self, sentence2_list):
        self.indexed = True
    
    def retrieve(self, query_sentence, top_k=5, initial_top_k=None):
        results = self.results_to_return["results"][:top_k]
        similarities = self.results_to_return["similarities"][:top_k]
        indices = self.results_to_return["indices"][:top_k]
        return results, similarities, indices


class TestEvaluatorIntegration:
    @pytest.fixture
    def sample_data(self):
        """Fixture for sample data"""
        # Create sample sentences
        sentence2_list = [
            "人工智能正在改变世界",
            "深度学习是人工智能的一个重要分支",
            "大语言模型已经取得了长足的进步",
            "机器学习算法需要大量的数据进行训练",
            "计算机视觉技术可以识别图像中的物体",
            "自然语言处理让机器能够理解人类语言",
            "强化学习是机器学习中的一种方法",
            "知识图谱可以表示实体之间的关系",
            "语义检索可以帮助我们找到相关的信息",
            "人工智能伦理是一个重要的研究领域"
        ]
        
        # Create sample queries
        query_sentences = [
            "人工智能如何应用于自然语言处理",
            "机器学习需要什么样的数据"
        ]
        
        # Create ground truth labels (1 for relevant, 0 for irrelevant)
        # For the first query, sentences 0, 1, 2, 5, 8 are relevant
        # For the second query, sentences 3, 4, 6 are relevant
        ground_truth_labels = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        
        return {
            "sentence2_list": sentence2_list,
            "query_sentences": query_sentences,
            "ground_truth_labels": ground_truth_labels
        }
    
    def test_evaluate_approach(self, sample_data):
        """Test evaluating a single approach"""
        evaluator = Evaluator()
        
        # Create mock retriever with predetermined results
        mock_results = {
            "results": sample_data["sentence2_list"][:5],
            "similarities": [0.9, 0.8, 0.7, 0.6, 0.5],
            "indices": [0, 1, 2, 5, 8]  # All relevant for the first query
        }
        
        mock_retriever = MockRetriever("mock_approach", mock_results)
        
        # Evaluate approach
        results = evaluator.evaluate_approach(
            approach_name="mock_approach",
            retriever=mock_retriever,
            query_sentences=sample_data["query_sentences"][:1],  # Just use the first query
            sentence2_list=sample_data["sentence2_list"],
            ground_truth_labels=sample_data["ground_truth_labels"],
            top_k=5
        )
        
        # Check results
        assert results["approach_name"] == "mock_approach"
        assert len(results["retrieved_results"]) == 1  # One query
        assert len(results["retrieved_similarities"]) == 1
        assert len(results["retrieved_indices"]) == 1
        
        # Check evaluation metrics
        assert results["evaluation"]["precision"] == 1.0  # All retrieved items are relevant
        assert abs(results["evaluation"]["recall"] - 1.0) < 1e-6  # All relevant items are retrieved
        assert abs(results["evaluation"]["f1"] - 1.0) < 1e-6
        assert results["evaluation"]["mrr"] == 1.0  # First item is relevant
        assert abs(results["evaluation"]["ndcg"] - 1.0) < 1e-6
    
    def test_evaluate_multiple_approaches(self, sample_data):
        """Test evaluating multiple approaches"""
        evaluator = Evaluator()
        
        # Create mock retrievers with predetermined results
        mock_results1 = {
            "results": sample_data["sentence2_list"][:5],
            "similarities": [0.9, 0.8, 0.7, 0.6, 0.5],
            "indices": [0, 1, 2, 5, 8]  # All relevant for the first query
        }
        
        mock_results2 = {
            "results": sample_data["sentence2_list"][5:10],
            "similarities": [0.85, 0.75, 0.65, 0.55, 0.45],
            "indices": [5, 8, 3, 4, 7]  # Mixed relevant and irrelevant
        }
        
        mock_retriever1 = MockRetriever("approach1", mock_results1)
        mock_retriever2 = MockRetriever("approach2", mock_results2)
        
        retrievers = {
            "approach1": mock_retriever1,
            "approach2": mock_retriever2
        }
        
        # Evaluate approaches
        results = evaluator.evaluate_multiple_approaches(
            retrievers=retrievers,
            query_sentences=sample_data["query_sentences"][:1],  # Just use the first query
            sentence2_list=sample_data["sentence2_list"],
            ground_truth_labels=sample_data["ground_truth_labels"],
            top_k=5
        )
        
        # Check results
        assert "approach1" in results
        assert "approach2" in results
        
        # Check approach1 results
        assert results["approach1"]["approach_name"] == "approach1"
        assert results["approach1"]["evaluation"]["precision"] == 1.0
        
        # Check approach2 results
        assert results["approach2"]["approach_name"] == "approach2"
        # For approach2, only 2 out of 5 retrieved items are relevant
        assert abs(results["approach2"]["evaluation"]["precision"] - 0.4) < 1e-6
    
    @pytest.mark.integration
    def test_integration_with_real_retrievers(self, sample_data):
        """Integration test with real retrievers"""
        # Skip this test if running in CI environment
        if os.environ.get('CI') == 'true':
            pytest.skip("Skipping integration test in CI environment")
        
        evaluator = Evaluator()
        
        # Create real retrievers
        word2vec_retriever = Word2VecApproach(persist_directory="test_eval_word2vec", use_vector_store=True)
        
        # Initialize retrievers
        retrievers = {
            "Word2Vec": word2vec_retriever
        }
        
        try:
            # Evaluate approaches
            results = evaluator.evaluate_multiple_approaches(
                retrievers=retrievers,
                query_sentences=sample_data["query_sentences"],
                sentence2_list=sample_data["sentence2_list"],
                ground_truth_labels=sample_data["ground_truth_labels"],
                top_k=3
            )
            
            # Check results
            assert "Word2Vec" in results
            assert "evaluation" in results["Word2Vec"]
            
            # Get comparison
            comparison = evaluator.compare_approaches(results)
            assert "Word2Vec" in comparison
            
            # Check metrics are between 0 and 1
            for metric in ["precision", "recall", "f1", "mrr", "ndcg"]:
                assert 0 <= comparison["Word2Vec"][metric] <= 1
        
        finally:
            # Clean up
            if os.path.exists("test_eval_word2vec"):
                import shutil
                shutil.rmtree("test_eval_word2vec")
    
    @pytest.mark.integration
    def test_integration_with_bge_and_llm(self, sample_data):
        """Integration test with BGE and LLM reranked retrievers"""
        # Skip this test by default as it requires external servers
        # pytest.skip("Skipping test that requires external servers")
        
        evaluator = Evaluator()
        
        # Create real retrievers with server URLs
        server_url = "http://20.30.80.200:9997"
        
        bge_retriever = BGEApproach(
            model_name="bge-large-zh-v1.5",
            server_url=server_url,
            persist_directory="test_eval_bge",
            use_vector_store=True
        )
        
        llm_reranked_retriever = LLMRerankedBGEApproach(
            embedding_model="bge-large-zh-v1.5",
            embedding_server_url=server_url,
            llm_model="deepseek-r1-distill-qwen",
            llm_server_url=server_url,
            persist_directory="test_eval_llm_rerank",
            use_vector_store=True
        )
        
        # Initialize retrievers
        retrievers = {
            "BGE": bge_retriever,
            "LLM-Reranked": llm_reranked_retriever
        }
        
        # Initial top_k for LLM reranking
        initial_top_k = {
            "LLM-Reranked": 5
        }
        
        try:
            # Evaluate approaches
            results = evaluator.evaluate_multiple_approaches(
                retrievers=retrievers,
                query_sentences=sample_data["query_sentences"],
                sentence2_list=sample_data["sentence2_list"],
                ground_truth_labels=sample_data["ground_truth_labels"],
                top_k=3,
                initial_top_k=initial_top_k
            )
            
            # Check results
            assert "BGE" in results
            assert "LLM-Reranked" in results
            
            # Get comparison
            comparison = evaluator.compare_approaches(results)
            
            # Visualize comparison
            evaluator.visualize_comparison(comparison, output_path="evaluation_comparison.png")
            
            # Check file was created
            assert os.path.exists("evaluation_comparison.png")
            
            # Save results
            filepath = evaluator.save_results(results)
            assert os.path.exists(filepath)
            
        finally:
            # Clean up
            for dir_name in ["test_eval_bge", "test_eval_llm_rerank"]:
                if os.path.exists(dir_name):
                    import shutil
                    shutil.rmtree(dir_name)
            
            if os.path.exists("evaluation_comparison.png"):
                os.remove("evaluation_comparison.png")
            
            if os.path.exists("evaluation_results"):
                import shutil
                shutil.rmtree("evaluation_results")
