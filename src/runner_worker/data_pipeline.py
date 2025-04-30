import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Iterator, Tuple
from src.runner_worker.model_runner import ModelRunner
from src.runner_worker.data_source import DataSource, InMemoryDataSource, CSVFileDataSource
from src.runner_worker.checkpoint_manager import ResultAccumulator, DataFrameAccumulator


class DataChunkProcessor:
    """
    A class that processes data chunks and delegates result storage to an accumulator.
    """
    
    def __init__(self, 
        data_source: DataSource,
        model_runner: Optional[ModelRunner] = None,
        accumulator: Optional[ResultAccumulator] = None, 
        chunk_size: int = 1024
    ):
        """
        Initialize the DataChunkProcessor with a result accumulator and optional data source.
        
        Args:
            accumulator: An implementation of ResultAccumulator to store results
            data_source: Optional data source to read data from (if None, synthetic data will be generated)
            model_runner: Optional model runner to process data chunks
            chunk_size: Size of each chunk to process
        """
        self.data_source = data_source
        self.model_runner = model_runner
        self.accumulator = accumulator
        self.chunk_size = chunk_size

    # Set key columns of data source
    def set_key_columns(self, key_columns: List[str]) -> None:
        # Key columns in the dataset
        self.key_columns = key_columns

    # Process a single data chunk and generate results
    def process_chunk(self, chunk_idx: int, chunk_size: int) -> bool:
        """
        Process a single data chunk and generate results.
        
        Args:
            chunk_idx: Index of the current chunk
            chunk_size: Size of the chunk to process
            
        Returns:
            bool: True if data was processed, False if no more data (end of chunks)
        """
        
        if self.data_source:
            # Get data from the data source
            _, chunk_df = self.data_source.get_chunk(chunk_idx, chunk_size)
            if chunk_df is None:
                return False  # Signal that there's no more data to process
            elif ((chunk_idx + 1) * chunk_size > self.data_source.total_rows):
                return True     # todo: Signal that it is the last chunk
            # Process the item by ModelRunner class
            result_dicts = self.model_runner.batch_predict(chunk_df)
            self.accumulator.add_result(result_dicts)
            return True  # Signal that data was processed successfully
        else:
            # Use the original synthetic data generation
            for i in range(chunk_size):
                data_item = {
                    'query_id': f"q{i % 100}",
                    'chunk_id': chunk_idx,
                    'label': np.random.randint(0, 2),
                }
                # Simulate the model response
                result = {
                    'score': np.random.random(),
                    'rank': i % 10,
                    'result': f"result_{chunk_idx}_{i}"
                }
                # Combine two dicts
                result.update(data_item)
                self.accumulator.add_result(result)
            return True
    
    def process_all_chunks(self, num_chunks: Optional[int] = None, 
                          save_interval: Optional[int] = None) -> pd.DataFrame:
        """
        Process all chunks and return the final dataframe.
        
        Args:
            num_chunks: Number of data chunks to process. If None and data_source is provided, 
                        will use data_source.get_total_chunks()
            save_interval: Optional override for when to save results
            
        Returns:
            Final dataframe with all results
        """
        # Determine the number of chunks to process
        if num_chunks is None and self.data_source:
            num_chunks = self.data_source.get_total_chunks(self.chunk_size)
        elif num_chunks is None:
            raise ValueError("num_chunks must be provided if no data_source is available")
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            if not self.process_chunk(chunk_idx, self.chunk_size):
                break  # Stop processing if there's no more data
            
            # Check if we've reached the save interval (if specified)
            if save_interval and (chunk_idx + 1) % save_interval == 0:
                # todo: how to use save_interval
                self.accumulator.save_chunk(force=True)
        
        # Return the final results
        return self.accumulator.get_final_results()


def process_csv_file(file_path: str, chunk_size: int = 1000, save_interval: int = 10,
                    transform_func=None) -> Tuple[pd.DataFrame, float]:
    """
    Process a CSV file in chunks and return the results.
    
    Args:
        file_path: Path to the CSV file
        chunk_size: Size of each chunk to process
        save_interval: Number of chunks to process before saving to a dataframe
        transform_func: Optional function to transform raw CSV rows
        
    Returns:
        Tuple of (resulting dataframe, execution time in seconds)
    """
    start_time = time.time()
    
    # Create data source for the CSV file
    data_source = CSVFileDataSource(file_path, transform_func)
    
    # Create the accumulator and processor with dependency injection
    accumulator = DataFrameAccumulator(chunk_size=chunk_size)
    processor = DataChunkProcessor(accumulator, data_source, chunk_size=chunk_size)
    
    # Process all chunks
    result_df = processor.process_all_chunks(save_interval=save_interval)
    
    end_time = time.time()
    return result_df, end_time - start_time


def method2_accumulate_and_concat(num_chunks: int, chunk_size: int, save_interval: int) -> float:
    """
    Method 2: Store results in a simple structure, periodically save to dataframe, and concat at the end.
    
    Args:
        num_chunks: Number of data chunks to process
        chunk_size: Size of each chunk
        save_interval: Number of chunks to process before saving to a dataframe
    
    Returns:
        Execution time in seconds
    """
    start_time = time.time()
    
    # Create the data source for synthetic data
    data_source = InMemoryDataSource(total_items=num_chunks * chunk_size)
    
    # Create the accumulator and processor with dependency injection
    accumulator = DataFrameAccumulator(chunk_size=chunk_size)
    processor = DataChunkProcessor(accumulator, data_source)
    
    # Process all chunks
    processor.process_all_chunks(num_chunks, chunk_size, save_interval)
    
    end_time = time.time()
    return end_time - start_time


def run_performance_test(
    num_chunks_list: List[int], 
    chunk_size: int, 
    save_intervals: List[int],
    num_trials: int = 3
) -> Dict[str, Dict[str, List[float]]]:
    """
    Run performance tests for both methods with different parameters.
    
    Args:
        num_chunks_list: List of different numbers of chunks to test
        chunk_size: Size of each chunk
        save_intervals: List of save intervals to test for method 2
        num_trials: Number of trials to run for each configuration
    
    Returns:
        Dictionary containing execution times for each method
    """
    results = {
        'method1': {chunks: [] for chunks in num_chunks_list},
        'method2': {f"{chunks}_interval_{interval}": [] 
                   for chunks in num_chunks_list 
                   for interval in save_intervals}
    }
    
    # Run trials for method 2 with different save intervals
    for chunks in num_chunks_list:
        for interval in save_intervals:
            for _ in range(num_trials):
                time_method2 = method2_accumulate_and_concat(chunks, chunk_size, interval)
                results['method2'][f"{chunks}_interval_{interval}"].append(time_method2)
    
    return results

def main():
    # Test parameters
    num_chunks_list = [10, 50, 100]
    chunk_size = 1000  # 1000 rows per chunk
    save_intervals = [1, 10, 50, 100]  # Different save intervals for method 2
    
    # Run performance tests
    print("Running performance tests...")
    results = run_performance_test(num_chunks_list, chunk_size, save_intervals)
    
    # Calculate and print average execution times
    print("\nAverage Execution Times:")
    for method, method_results in results.items():
        for key, times in method_results.items():
            avg_time = np.mean(times)
            print(f"{method} - {key}: {avg_time:.6f} seconds")

if __name__ == "__main__":
    main()