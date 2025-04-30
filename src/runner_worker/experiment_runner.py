# -*- coding: utf-8 -*-
"""
Abstract base class for running experiments over datasets.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, List, Tuple, Union, Callable

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.runner_worker.data_source import DataSource
from src.runner_worker.checkpoint_manager import ResultAccumulator
from src.runner_worker.model_runner import ModelRunner
from src.utils import build_logger


class ExperimentError(Exception):
    """Base exception class for experiment-related errors."""
    pass


class DataLoadError(ExperimentError):
    """Exception raised for errors in data loading."""
    pass


class ModelInitError(ExperimentError):
    """Exception raised for errors in model initialization."""
    pass


class ProcessingError(ExperimentError):
    """Exception raised for errors during data processing."""
    pass


class CheckpointError(ExperimentError):
    """Exception raised for errors during checkpoint saving/loading."""
    pass


class ExperimentRunner(ABC):
    """
    Abstract base class for coordinating data management and model execution in experiments.

    Subclasses must implement data source, result accumulator and model runner initialization.
    Provides generic implementations for batch processing, chunk processing,
    checkpointing, and experiment execution flow.
    
    Cross-cutting concerns like logging, error handling, and progress tracking
    are centralized in this base class.
    """

    def __init__(
        self,
        data_path: str,
        chunksize: int = 10000,
        experiment_id: Optional[str] = None,
        checkpoint_interval: int = 1000,
        batch_size: int = 32,
        n_jobs: int = 4,
        log_level: int = logging.INFO,
        logger_name: Optional[str] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            data_path: Path to the input CSV file.
            chunksize: Number of rows per data chunk.
            experiment_id: Optional experiment ID for checkpointing.
            checkpoint_interval: Number of results to cache before saving a checkpoint.
            batch_size: Number of rows per processing batch.
            n_jobs: Number of threads for parallel processing.
            log_level: Logging level (default: logging.INFO).
            logger_name: Custom logger name (default: class name).
        """
        # Initialize logger
        self.logger = build_logger(logger_name or self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        try:
            # Store basic parameters
            self.data_path = data_path
            self.chunksize = chunksize
            self.experiment_id = experiment_id or f"exp_{int(time.time())}"
            self.checkpoint_interval = checkpoint_interval
            self.batch_size = batch_size
            self.n_jobs = n_jobs
            
            # Initialize data source
            self.logger.info(f"Initializing data source for {data_path}")
            self.data_source: DataSource = self._init_data_source(
                data_path, chunksize, experiment_id
            )
            
            # Initialize result accumulator
            self.logger.info(f"Initializing result accumulator for experiment {self.experiment_id}")
            self.result_accumulator: ResultAccumulator = self._init_result_accumulator(
                experiment_id, data_path
            )
            
            # Create temp directory for the experiment
            self.temp_dir = self.result_accumulator.create_experiment_tempdir(
                self.experiment_id, data_path
            )

            # Load processed indices from checkpoint if present
            self.processed_indices: Set[int] = self._load_processed_indices()
            self.results_cache: Dict[int, Dict[str, Any]] = {}
            
            # Total rows for progress calculation
            self.total_rows = self.data_source.total_rows
            
            # Initialize model runners
            self.logger.info("Initializing model runners")
            self.model_runners: Dict[str, ModelRunner] = self._init_model_runners()
            
            # Log successful initialization
            self.logger.info(f"Experiment runner initialized: id={self.experiment_id}, "
                           f"total_rows={self.total_rows}, "
                           f"processed_rows={len(self.processed_indices)}")
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment runner: {e}")
            raise ExperimentError(f"Initialization failed: {e}") from e

    @abstractmethod
    def _init_data_source(
        self, data_path: str, chunksize: int, experiment_id: Optional[str]
    ) -> DataSource:
        """
        Create and return a DataSource instance.
        """
        pass
    
    @abstractmethod
    def _init_result_accumulator(
        self, experiment_id: Optional[str], data_path: str
    ) -> ResultAccumulator:
        """
        Create and return a ResultAccumulator instance.
        """
        pass

    @abstractmethod
    def _init_model_runners(self) -> Dict[str, ModelRunner]:
        """
        Create and return a dict mapping model names to ModelRunner instances.
        Keys will be used as result column names.
        """
        pass
    
    def _load_processed_indices(self) -> Set[int]:
        """
        Load already processed row indices from checkpoint.
        
        Returns:
            Set of processed row indices
        """
        # Implement logic to load processed indices
        # This would typically be handled by your ResultAccumulator implementation
        return set()

    def _handle_row_error(self, idx: int, error: Exception) -> None:
        """
        Handle errors during row processing.
        
        Subclasses can override this to customize error handling.
        
        Args:
            idx: The index of the row that caused the error.
            error: The exception that was raised.
        """
        self.logger.error(f"Error processing row {idx}: {error}")

    def process_batch(
        self, rows: List[Tuple[int, pd.Series]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process a list of rows and return mapping from global index to model results.
        
        This is a template method that handles thread management and error handling.
        """
        # Ensure all models are initialized
        for name, runner in self.model_runners.items():
            try:
                runner.initialize()
                self.logger.debug(f"Ensured initialization of model runner: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize model runner {name}: {e}")
                raise ModelInitError(f"Failed to initialize {name}: {e}") from e

        results: Dict[int, Dict[str, Any]] = {}

        def _run(idx: int, row: pd.Series):
            try:
                data = row.to_dict()
                model_results = {}
                
                for name, runner in self.model_runners.items():
                    try:
                        model_results[name] = runner.predict(data)
                    except Exception as e:
                        self.logger.warning(f"Model {name} failed for row {idx}: {e}")
                        model_results[name] = None
                
                return idx, model_results
            except Exception as e:
                self.logger.error(f"Failed to process row {idx}: {e}")
                raise ProcessingError(f"Row processing failed: {e}") from e

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_run, idx, row): idx for idx, row in rows}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    key, res = future.result()
                    results[key] = res
                    self.logger.debug(f"Processed row {idx} successfully")
                except Exception as e:
                    self._handle_row_error(idx, e)
        
        self.logger.debug(f"Batch processing complete, {len(results)}/{len(rows)} rows successful")
        return results

    def process_chunk(
        self, chunk_idx: int, chunk: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Process a data chunk, handling batching, checkpointing, and progress.

        Returns a dict containing:
            - results: mapping of index to model results
            - processed_in_chunk: number of newly processed rows
            - chunk_completion: percent completion of this chunk
            - dataset_completion: percent completion of entire dataset
        """
        self.logger.info(f"Starting processing of chunk {chunk_idx} with {len(chunk)} rows")
        
        chunk_results: Dict[int, Dict[str, Any]] = {}
        processed_in_chunk = 0

        global_col = self.data_source.global_index_column
        indices = [int(r[global_col]) for _, r in chunk.iterrows()]
        total = len(indices)
        done = len(set(indices) & self.processed_indices)
        
        self.logger.info(f"Chunk {chunk_idx}: {done}/{total} rows already processed")

        queue: List[Tuple[int, pd.Series]] = []
        try:
            for _, row in chunk.iterrows():
                idx = int(row[global_col])
                if idx in self.processed_indices:
                    continue
                queue.append((idx, row))
                if len(queue) >= self.batch_size:
                    self.logger.debug(f"Processing batch of {len(queue)} rows")
                    batch_res = self.process_batch(queue)
                    chunk_results.update(batch_res)
                    processed_in_chunk += len(batch_res)
                    for i, r in batch_res.items():
                        self.processed_indices.add(i)
                        self.results_cache[i] = r
                    queue.clear()
                    if len(self.results_cache) >= self.checkpoint_interval:
                        self.save_checkpoint(chunk_idx)
                        
            # flush remaining
            if queue:
                self.logger.debug(f"Processing final batch of {len(queue)} rows")
                batch_res = self.process_batch(queue)
                chunk_results.update(batch_res)
                processed_in_chunk += len(batch_res)
                for i, r in batch_res.items():
                    self.processed_indices.add(i)
                    self.results_cache[i] = r
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_idx}: {e}")
            # Still try to save what we have
            if self.results_cache:
                self.logger.info("Attempting to save partial results after error")
                self.save_checkpoint(chunk_idx)
            raise ProcessingError(f"Chunk {chunk_idx} processing failed: {e}") from e

        processed_total = processed_in_chunk + done
        chunk_completion = (processed_total / total * 100) if total > 0 else 100.0
        dataset_completion = (len(self.processed_indices) / self.total_rows * 100)

        self.logger.info(f"Completed chunk {chunk_idx}: processed {processed_in_chunk} new rows, "
                       f"{chunk_completion:.2f}% of chunk, {dataset_completion:.2f}% of total dataset")

        return {
            "results": chunk_results,
            "processed_in_chunk": processed_in_chunk,
            "chunk_completion": chunk_completion,
            "dataset_completion": dataset_completion,
        }

    def save_checkpoint(self, chunk_idx: int) -> None:
        """
        Save the current results cache to disk and clear it.
        """
        if not self.results_cache:
            self.logger.debug("No results to checkpoint, skipping save")
            return
            
        start = chunk_idx * self.chunksize
        end = min(start + self.chunksize, self.total_rows)
        processed = sum(
            1 for i in self.processed_indices if start <= i < end
        )
        size = end - start
        pct = (processed / size * 100) if size > 0 else 100.0

        self.logger.info(
            f"Saving checkpoint: chunk={chunk_idx}, "
            f"cache_size={len(self.results_cache)}, "
            f"chunk_progress={pct:.2f}%"
        )
        
        try:
            # Save results to accumulator
            for idx, result in self.results_cache.items():
                self.result_accumulator.add_result(chunk_idx, result)
                
            # Force save the chunk
            self.result_accumulator.save_chunk(force=True)
            
            # Clear cache after successful save
            self.results_cache.clear()
            
            overall = len(self.processed_indices) / self.total_rows * 100
            self.logger.info(f"Overall progress: {overall:.2f}%")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise CheckpointError(f"Failed to save checkpoint: {e}") from e

    def run(self, max_rows: Optional[int] = None) -> None:
        """
        Run the experiment, processing all chunks and handling checkpoints.
        
        This template method coordinates the overall experiment flow,
        including error handling and resource cleanup.
        """
        self.logger.info(f"Starting experiment {self.experiment_id}...")
        start_time = time.time()
        processed_count = 0
        chunk_idx = 0

        try:
            if not self.processed_indices:
                self.logger.info("Starting new experiment from beginning")
                chunks = self._get_chunks()
            else:
                self.logger.info(f"Resuming experiment, already {len(self.processed_indices)} rows done")
                chunks = self._get_unprocessed_chunks()

            for chunk_idx, (start, chunk) in enumerate(chunks):
                self.logger.info(f"Processing chunk {chunk_idx}, rows={len(chunk)}...")
                try:
                    info = self.process_chunk(chunk_idx, chunk)
                    processed_count += info["processed_in_chunk"]
                    self.logger.info(
                        f"Chunk {chunk_idx} done: {info['chunk_completion']:.2f}% chunk, "
                        f"{info['dataset_completion']:.2f}% total"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                    # Continue with next chunk instead of breaking
                    continue
                    
                if max_rows and processed_count >= max_rows:
                    self.logger.info(f"Reached max_rows={max_rows}, stopping")
                    break

            if self.results_cache:
                self.logger.info("Saving final checkpoint...")
                self.save_checkpoint(chunk_idx)
                
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            if self.results_cache:
                self.logger.info("Saving emergency checkpoint due to failure...")
                try:
                    self.save_checkpoint(chunk_idx)
                except Exception as checkpoint_e:
                    self.logger.error(f"Failed to save emergency checkpoint: {checkpoint_e}")
            raise
        finally:
            self.cleanup()
            duration = time.time() - start_time
            self.logger.info(
                f"Experiment {self.experiment_id} finished, "
                f"processed {len(self.processed_indices)} rows in {duration:.2f}s"
            )
    
    def _get_chunks(self) -> List[Tuple[int, pd.DataFrame]]:
        """
        Get all chunks from the data source.
        
        Returns:
            List of (start_index, DataFrame) tuples
        """
        chunks = []
        total_chunks = self.data_source.get_total_chunks(self.chunksize)
        
        for chunk_idx in range(total_chunks):
            start_idx, chunk_df, is_last = self.data_source.get_chunk(chunk_idx, self.chunksize)
            if chunk_df is not None:
                chunks.append((start_idx, chunk_df))
            
            if is_last:
                break
                
        return chunks
    
    def _get_unprocessed_chunks(self) -> List[Tuple[int, pd.DataFrame]]:
        """
        Get chunks that have unprocessed rows.
        
        Returns:
            List of (start_index, DataFrame) tuples containing unprocessed rows
        """
        # This is a simplified implementation
        # Full implementation would filter chunks based on processed_indices
        return self._get_chunks()

    def cleanup(self) -> None:
        """
        Cleanup resources after experiment completion.
        
        This template method ensures resources are released properly,
        handling any errors that might occur during cleanup.
        """
        self.logger.info("Cleaning up model resources...")
        for name, runner in self.model_runners.items():
            try:
                runner.cleanup()
                self.logger.debug(f"Cleaned up model runner: {name}")
            except Exception as e:
                self.logger.error(f"Error cleaning runner {name}: {e}")
                # Continue cleanup for other runners
