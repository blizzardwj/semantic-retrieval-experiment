"""
语义匹配实验运行器

该脚本使用数据源和结果累加器处理大型数据集，执行不同模型的语义匹配实验，
并将结果保存到新的CSV文件中。
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

from src.runner_worker.data_source import DataSource, CSVFileDataSource
from src.runner_worker.checkpoint_manager import ResultAccumulator, DataFrameAccumulator
from src.runner_worker.experiment_runner import ExperimentRunner, ModelInitError
# Assuming the new runner is in the same directory or appropriately pathed
from src.pairwise_classification.similarity_model_runner import SimilarityModelRunner
# Assuming embedding models are importable
from src.pairwise_classification.embedding.fasttext import FastTextEmbedding
from src.pairwise_classification.embedding.bge import BGEEmbedding

class SemanticMatchingRunner(ExperimentRunner):
    """
    语义匹配实验运行器，使用ExperimentRunner基类协调数据管理和模型运行。
    
    通过实现 _init_data_source, _init_result_accumulator 和 _init_model_runners 
    来配置具体的数据源和模型。
    """

    def __init__(self, 
        data_path: str, 
        chunksize: int = 10000, 
        experiment_id: Optional[str] = None,
        checkpoint_interval: int = 1000,
        batch_size: int = 32,
        n_jobs: int = 4,
        fasttext_model_path: str = "",
        xinference_server_url: str = "",
        log_level: int = None
    ):
        """
        Initialize the semantic matching runner.
        
        Args:
            data_path: Path to the data file.
            chunksize: Size of data chunks to process.
            experiment_id: ID for the experiment (used for output and checkpoints).
            checkpoint_interval: How often to save checkpoints (in number of rows).
            batch_size: Size of batches for parallel processing.
            n_jobs: Number of parallel jobs (-1 uses all available cores).
            fasttext_model_path: Path to the FastText model binary.
            xinference_server_url: URL of the XInference server for BGE models.
            log_level: Optional logging level (uses default from parent if None).
        """
        # Store model configurations
        if not (fasttext_model_path and xinference_server_url):
            raise ValueError("fasttext_model_path or xinference_server_url not provided")
            
        self._fasttext_model_path = fasttext_model_path
        self._xinference_server_url = xinference_server_url

        # Define the expected result columns (used by both data source and result accumulator)
        self._result_columns = [
            'word2vec_match_res',
            'bge_large_match_res',
            'bge_m3_match_res'
        ]

        # Initialize the base ExperimentRunner
        super().__init__(
            data_path=data_path,
            chunksize=chunksize,
            experiment_id=experiment_id,
            checkpoint_interval=checkpoint_interval,
            batch_size=batch_size,
            n_jobs=n_jobs,
            log_level=log_level,
            logger_name="SemanticMatchingRunner"
        )
        
    def _init_data_source(
        self, data_path: str, chunksize: int, experiment_id: Optional[str]
    ) -> DataSource:
        """
        Create and return a DataSource instance for CSV files.
        """
        self.logger.debug(f"Creating CSVFileDataSource for {data_path}")
        
        # Custom transformation function to add result columns
        def transform_func(df):
            # Add result columns if they don't exist
            for col in self._result_columns:
                if col not in df.columns:
                    df[col] = None
            return df
            
        return CSVFileDataSource(file_path=data_path, transform_func=transform_func)
    
    def _init_result_accumulator(
        self, experiment_id: Optional[str], data_path: str
    ) -> ResultAccumulator:
        """
        Create and return a ResultAccumulator instance for storing results.
        """
        self.logger.debug(f"Creating DataFrameAccumulator for experiment {experiment_id}")
        
        # Use the DataFrameAccumulator implementation
        return DataFrameAccumulator(chunk_size=self.chunksize)

    def _init_model_runners(self) -> Dict[str, SimilarityModelRunner]:
        """
        Initialize embedding models and wrap them in SimilarityModelRunners.
        Returns a dictionary mapping result column names to their runners.
        """
        model_runners = {}
        input_cols = ['sentence1', 'sentence2']
        
        # Define the models we'll try to initialize
        models_to_init = [
            {
                'name': 'word2vec_match_res',
                'init_func': self._init_fasttext_model,
                'required': True
            },
            {
                'name': 'bge_large_match_res',
                'init_func': lambda: self._init_bge_model("bge-large-zh-v1.5"),
                'required': False
            },
            {
                'name': 'bge_m3_match_res',
                'init_func': lambda: self._init_bge_model("bge-m3"),
                'required': False
            }
        ]
        
        # Initialize each model
        for model_info in models_to_init:
            try:
                self.logger.info(f"Initializing {model_info['name']} model...")
                embedding_model = model_info['init_func']()
                model_runners[model_info['name']] = SimilarityModelRunner(
                    embedding_model=embedding_model,
                    input_columns=input_cols
                )
                self.logger.info(f"{model_info['name']} model initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize {model_info['name']} model: {e}")
                if model_info['required']:
                    raise ModelInitError(f"Required model {model_info['name']} failed to initialize: {e}") from e

        if not model_runners:
            raise ModelInitError("No models could be initialized. Cannot run experiment.")
            
        self.logger.info(f"Successfully initialized {len(model_runners)} model runners")
        return model_runners
        
    def _init_fasttext_model(self):
        """Helper method to initialize FastText model"""
        self.logger.debug(f"Initializing FastText with model path: {self._fasttext_model_path}")
        fasttext_embedding = FastTextEmbedding(model_path=self._fasttext_model_path)
        fasttext_embedding.initialize()
        return fasttext_embedding
        
    def _init_bge_model(self, model_name):
        """Helper method to initialize BGE models"""
        self.logger.debug(f"Initializing {model_name} with server URL: {self._xinference_server_url}")
        bge_embedding = BGEEmbedding(model_name=model_name)
        bge_embedding.initialize(
            server_url=self._xinference_server_url, 
            model_uid=model_name
        )
        return bge_embedding


# Main execution block (optional, can be moved to a separate script)
# -----------------------------------------------------------------
if __name__ == "__main__":
    import logging
    import argparse
    from pathlib import Path
    from src.utils.utils import load_config
    
    # Load configuration from config.yml
    config_path = Path(__file__).parent / "config.yml"
    config = load_config(config_path)
    experiment_config = config.get('experiment_config', {})
    
    # Define arguments with defaults from config
    parser = argparse.ArgumentParser(description="Run Semantic Matching Experiment")
    parser.add_argument("--data_path", type=str, 
        default=experiment_config.get('data_path'), 
        help="Path to the input CSV data file."
    )
    parser.add_argument("--chunksize", type=int, 
        default=experiment_config.get('chunksize', 10000), 
        help="Chunk size for reading data."
    )
    parser.add_argument("--experiment_id", type=str, 
        default=experiment_config.get('experiment_id'), 
        help="Optional experiment ID."
    )
    parser.add_argument("--checkpoint_interval", type=int, 
        default=experiment_config.get('checkpoint_interval', 1000), 
        help="Checkpoint save interval (rows)."
    )
    parser.add_argument("--batch_size", type=int, 
        default=experiment_config.get('batch_size', 32), 
        help="Batch size for parallel processing."
    )
    parser.add_argument("--n_jobs", type=int, 
        default=experiment_config.get('n_jobs', 4), 
        help="Number of parallel jobs."
    )
    parser.add_argument("--fasttext_model_path", type=str, 
        default=experiment_config.get('fasttext_model_path'), 
        help="Path to FastText model binary."
    )
    parser.add_argument("--xinference_server_url", type=str, 
        default=experiment_config.get('xinference_server_url'), 
        help="URL for XInference server."
    )
    parser.add_argument("--verbose", action="store_true", 
        help="Enable verbose logging."
    )
    parser.add_argument("--max_rows", type=int, 
        default=None, 
        help="Maximum rows to process (for testing)."
    )
    
    args = parser.parse_args()
    
    # Set log level based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Initialize and run the experiment
    runner = SemanticMatchingRunner(
        data_path=args.data_path,
        chunksize=args.chunksize,
        experiment_id=args.experiment_id,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        fasttext_model_path=args.fasttext_model_path,
        xinference_server_url=args.xinference_server_url,
        log_level=log_level
    )
    
    # Run the experiment
    runner.run(max_rows=args.max_rows)
