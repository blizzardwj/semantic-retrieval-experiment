# -*- coding: utf-8 -*-
"""
Abstract base class for running experiments over datasets.
"""
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, List, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.runner_worker.data_manager import ChunkedDataManager
from src.runner_worker.model_runner import ModelRunner


class ExperimentRunner(ABC):
    """
    Abstract base class for coordinating data management and model execution in experiments.

    Subclasses must implement data manager and model runner initialization.
    Provides generic implementations for batch processing, chunk processing,
    checkpointing, and experiment execution flow.
    """

    def __init__(
        self,
        data_path: str,
        chunksize: int = 10000,
        experiment_id: Optional[str] = None,
        checkpoint_interval: int = 1000,
        batch_size: int = 32,
        n_jobs: int = 4,
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
        """
        # Initialize data manager
        self.data_manager: ChunkedDataManager = self._init_data_manager(
            data_path, chunksize, experiment_id
        )
        self.chunksize = chunksize
        self.experiment_id = self.data_manager.experiment_id
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.n_jobs = n_jobs

        # Load processed indices from checkpoint if present
        self.processed_indices: Set[int] = set(self.data_manager.processed_indices)
        self.results_cache: Dict[int, Dict[str, Any]] = {}

        # Initialize model runners
        self.model_runners: Dict[str, ModelRunner] = self._init_model_runners()

    @abstractmethod
    def _init_data_manager(
        self, data_path: str, chunksize: int, experiment_id: Optional[str]
    ) -> ChunkedDataManager:
        """
        Create and return a ChunkedDataManager instance.
        """
        pass

    @abstractmethod
    def _init_model_runners(self) -> Dict[str, ModelRunner]:
        """
        Create and return a dict mapping model names to ModelRunner instances.
        Keys will be used as result column names.
        """
        pass

    def process_batch(
        self, rows: List[Tuple[int, pd.Series]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Process a list of rows and return mapping from global index to model results.
        """
        # Ensure all models are initialized
        for runner in self.model_runners.values():
            runner.initialize()

        results: Dict[int, Dict[str, Any]] = {}

        def _run(idx: int, row: pd.Series):
            data = row.to_dict()
            return idx, {
                name: runner.predict(data)
                for name, runner in self.model_runners.items()
            }

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(_run, idx, row): idx for idx, row in rows}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    key, res = future.result()
                    results[key] = res
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
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
        chunk_results: Dict[int, Dict[str, Any]] = {}
        processed_in_chunk = 0

        global_col = self.data_manager.global_index_column
        indices = [int(r[global_col]) for _, r in chunk.iterrows()]
        total = len(indices)
        done = len(set(indices) & self.processed_indices)

        queue: List[Tuple[int, pd.Series]] = []
        for _, row in chunk.iterrows():
            idx = int(row[global_col])
            if idx in self.processed_indices:
                continue
            queue.append((idx, row))
            if len(queue) >= self.batch_size:
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
            batch_res = self.process_batch(queue)
            chunk_results.update(batch_res)
            processed_in_chunk += len(batch_res)
            for i, r in batch_res.items():
                self.processed_indices.add(i)
                self.results_cache[i] = r

        processed_total = processed_in_chunk + done
        chunk_completion = (processed_total / total * 100) if total > 0 else 100.0
        dataset_completion = (len(self.processed_indices) / self.data_manager.total_rows * 100)

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
            return
        start = chunk_idx * self.chunksize
        end = min(start + self.chunksize, self.data_manager.total_rows)
        processed = sum(
            1 for i in self.processed_indices if start <= i < end
        )
        size = end - start
        pct = (processed / size * 100) if size > 0 else 100.0

        print(
            f"Saving checkpoint chunk={chunk_idx}, "
            f"cache_size={len(self.results_cache)}, "
            f"chunk_progress={pct:.2f}%"
        )
        ok = self.data_manager.save_results(chunk_idx, self.results_cache)
        if not ok:
            print("Warning: checkpoint save failed")
        self.results_cache.clear()
        overall = len(self.processed_indices) / self.data_manager.total_rows * 100
        print(f"Overall progress: {overall:.2f}%")

    def run(self, max_rows: Optional[int] = None) -> None:
        """
        Execute the full experiment, reading chunks and processing them.
        """
        print(f"Starting experiment {self.experiment_id}...")
        start_time = time.time()
        processed_count = 0

        if not self.processed_indices:
            chunks = self.data_manager.get_chunks()
        else:
            print(f"Resuming experiment, already {len(self.processed_indices)} rows done")
            chunks = self.data_manager.get_unprocessed_chunks(self.processed_indices)

        for chunk_idx, (start, chunk) in enumerate(chunks):
            print(f"\nProcessing chunk {chunk_idx}, rows={len(chunk)}...")
            info = self.process_chunk(chunk_idx, chunk)
            processed_count += info["processed_in_chunk"]
            print(
                f"Chunk {chunk_idx} done: {info['chunk_completion']:.2f}% chunk, "
                f"{info['dataset_completion']:.2f}% total"
            )
            if max_rows and processed_count >= max_rows:
                print(f"Reached max_rows={max_rows}, stopping")
                break

        if self.results_cache:
            print("Saving final checkpoint...")
            self.save_checkpoint(chunk_idx)

        self.cleanup()
        duration = time.time() - start_time
        print(
            f"Experiment {self.experiment_id} finished, "
            f"processed {len(self.processed_indices)} rows in {duration:.2f}s"
        )

    def cleanup(self) -> None:
        """
        Cleanup resources after experiment.
        """
        print("Cleaning up model resources...")
        for runner in self.model_runners.values():
            try:
                runner.cleanup()
            except Exception as e:
                print(f"Error cleaning runner {runner}: {e}")
