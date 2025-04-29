"""
语义匹配实验运行器

该脚本使用ChunkedDataManager处理大型数据集，执行不同模型的语义匹配实验，
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

# 添加src目录到Python路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.runner_worker.data_manager import ChunkedDataManager
from src.runner_worker.experiment_runner import ExperimentRunner
# Assuming the new runner is in the same directory or appropriately pathed
from src.pairwise_classification.similarity_model_runner import SimilarityModelRunner
# Assuming embedding models are importable
from src.pairwise_classification.embedding.fasttext import FastTextEmbedding
from src.pairwise_classification.embedding.bge import BGEEmbedding
from src.utils import build_logger
from src.utils.utils import load_config

logger = build_logger(__name__)

class SemanticMatchingRunner(ExperimentRunner):
    """
    语义匹配实验运行器，使用ExperimentRunner基类协调数据管理和模型运行。
    
    通过实现 _init_data_manager 和 _init_model_runners 来配置具体的数据源和模型。
    """

    def __init__(self, 
                 data_path: str, 
                 chunksize: int = 10000, 
                 experiment_id: Optional[str] = None,
                 checkpoint_interval: int = 1000,
                 batch_size: int = 32,
                 n_jobs: int = 4,
                 fasttext_model_path: str = "",
                 xinference_server_url: str = ""):
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
        """
        # Store model configurations
        if fasttext_model_path and xinference_server_url:
            self._fasttext_model_path = fasttext_model_path
            self._xinference_server_url = xinference_server_url
        else:
            raise ValueError("fasttext_model_path or xinference_server_url not provided")

        # Initialize the base ExperimentRunner
        super().__init__(
            data_path=data_path,
            chunksize=chunksize,
            experiment_id=experiment_id,
            checkpoint_interval=checkpoint_interval,
            batch_size=batch_size,
            n_jobs=n_jobs
        )

        logger.info(f"Initialized SemanticMatchingRunner (inheriting from ExperimentRunner)")
        logger.info(f"Data Path: {self.data_manager.csv_path}")
        logger.info(f"Total Rows: {self.data_manager.total_rows}")
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Chunk Size: {self.chunksize}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Checkpoint Interval: {self.checkpoint_interval}")
        logger.info(f"Number of Jobs: {self.n_jobs}")
        logger.info(f"Result Columns: {list(self.model_runners.keys())}")
        logger.info(f"Output Path: {self.data_manager.output_path}")
        
    def _init_data_manager(
        self, data_path: str, chunksize: int, experiment_id: Optional[str]
    ) -> ChunkedDataManager:
        """
        Create and return the ChunkedDataManager instance.
        Result columns are determined by the keys of the dictionary returned by _init_model_runners.
        """
        # Define the expected result columns based on model runner keys
        # This ensures the data manager is aware of the columns the runners will produce.
        # We can get these keys after _init_model_runners is called, but DataManager needs them at init.
        # Let's hardcode them here based on what _init_model_runners *will* return.
        result_columns = [
            'word2vec_match_res',
            'bge_large_match_res',
            'bge_m3_match_res'
        ]
        
        return ChunkedDataManager(
            csv_path=data_path,
            chunksize=chunksize,
            experiment_id=experiment_id,
            result_columns=result_columns # Pass the expected columns
        )

    def _init_model_runners(self) -> Dict[str, SimilarityModelRunner]:
        """
        Initialize embedding models and wrap them in SimilarityModelRunners.
        Returns a dictionary mapping result column names to their runners.
        """
        model_runners = {}
        input_cols = ['sentence1', 'sentence2']

        try:
            # Initialize FastText
            logger.info("Initializing FastText model...")
            fasttext_embedding = FastTextEmbedding(model_path=self._fasttext_model_path)
            fasttext_embedding.initialize()
            model_runners['word2vec_match_res'] = SimilarityModelRunner(
                embedding_model=fasttext_embedding,
                input_columns=input_cols
            )
            logger.info("FastText model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize FastText model: {e}. Skipping Word2Vec matching.")

        try:
            # Initialize BGE-Large
            logger.info("Initializing BGE-Large model...")
            bge_large_embedding = BGEEmbedding(model_name="bge-large-zh-v1.5")
            bge_large_embedding.initialize(
                server_url=self._xinference_server_url, 
                model_uid="bge-large-zh-v1.5"
            )
            model_runners['bge_large_match_res'] = SimilarityModelRunner(
                embedding_model=bge_large_embedding,
                input_columns=input_cols
            )
            logger.info("BGE-Large model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize BGE-Large model: {e}. Skipping BGE-Large matching.")
        
        try:
            # Initialize BGE-M3
            logger.info("Initializing BGE-M3 model...")
            bge_m3_embedding = BGEEmbedding(model_name="bge-m3")
            bge_m3_embedding.initialize(
                server_url=self._xinference_server_url, 
                model_uid="bge-m3"
            )
            model_runners['bge_m3_match_res'] = SimilarityModelRunner(
                embedding_model=bge_m3_embedding,
                input_columns=input_cols
            )
            logger.info("BGE-M3 model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize BGE-M3 model: {e}. Skipping BGE-M3 matching.")

        if not model_runners:
            raise RuntimeError("No models could be initialized. Cannot run experiment.")
            
        return model_runners

    # --------------------------------------------------------------------------
    # Methods below are now inherited from ExperimentRunner or are no longer needed
    # --------------------------------------------------------------------------
    # - __init__ is modified to call super()
    # - _initialize_fasttext_model, _initialize_bge_large_model, _initialize_bge_m3_model 
    #   logic moved into _init_model_runners
    # - run_word2vec_matching, run_bge_large_matching, run_bge_m3_matching
    #   functionality handled by SimilarityModelRunner.predict
    # - process_row, process_batch, process_chunk, save_checkpoint, run, cleanup
    #   are provided by the ExperimentRunner base class.
    # - Model attributes (_fasttext_model, etc.) are now encapsulated within the runners.
    # - data_manager, processed_indices, results_cache are handled by the base class.


# Main execution block (optional, can be moved to a separate script)
# -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semantic Matching Experiment")
    
    # Load configuration from config.yml
    config_path = Path(__file__).parent / "config.yml"
    config = load_config(config_path)
    experiment_config = config.get('experiment_config', {})
    
    # Define arguments with defaults from config
    parser.add_argument("--data_path", type=str, 
                      default=experiment_config.get('data_path'), 
                      help="Path to the input CSV data file.")
    parser.add_argument("--chunksize", type=int, 
                      default=experiment_config.get('chunksize', 10000), 
                      help="Chunk size for reading data.")
    parser.add_argument("--experiment_id", type=str, 
                      default=experiment_config.get('experiment_id'), 
                      help="Optional experiment ID.")
    parser.add_argument("--checkpoint_interval", type=int, 
                      default=experiment_config.get('checkpoint_interval', 1000), 
                      help="Checkpoint save interval (rows).")
    parser.add_argument("--batch_size", type=int, 
                      default=experiment_config.get('batch_size', 32), 
                      help="Batch size for parallel processing.")
    parser.add_argument("--n_jobs", type=int, 
                      default=experiment_config.get('n_jobs', 4), 
                      help="Number of parallel jobs.")
    parser.add_argument("--max_rows", type=int, 
                      default=experiment_config.get('max_rows'), 
                      help="Maximum number of rows to process (for testing).")
    parser.add_argument("--fasttext_path", type=str, 
                      default=experiment_config.get('fasttext_path', "/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin"), 
                      help="Path to FastText model.")
    parser.add_argument("--xinference_url", type=str, 
                      default=experiment_config.get('xinference_url', "http://20.30.80.200:9997"), 
                      help="XInference server URL.")
    
    args = parser.parse_args()
    
    # Create and run the experiment
    runner = SemanticMatchingRunner(
        data_path=args.data_path,
        chunksize=args.chunksize,
        experiment_id=args.experiment_id,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs,
        fasttext_model_path=args.fasttext_path,
        xinference_server_url=args.xinference_url
    )
    
    runner.run(max_rows=args.max_rows)
    runner.cleanup()
