import pytest
from semantic_retrieval.experiment.runner import ExperimentRunner

class TestExperimentRunner:
    def test_experiment_initialization(self):
        """Test experiment runner initialization"""
        runner = ExperimentRunner()
        assert runner is not None

    def test_run_experiment(self, mock_data_loader):
        """Test full experiment execution flow"""
        runner = ExperimentRunner()
        results = runner.run(mock_data_loader)
        assert results is not None
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
