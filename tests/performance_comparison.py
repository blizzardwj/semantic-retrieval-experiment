"""
性能比较测试

该脚本用于比较两种数据处理方法的性能差异。
Method 1: 预分配数据框并填充结果
Method 2: 存储结果在简单结构中，定期保存到数据框并连接

"""

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# 首先定义ResultAccumulator的抽象接口
class ResultAccumulator(ABC):
    """
    Abstract base class for result accumulation strategies.
    Defines the interface for storing, accumulating and retrieving results.
    """
    
    @abstractmethod
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single result to the accumulator.
        
        Args:
            result: A dictionary containing a single result
        """
        pass
    
    @abstractmethod
    def save_batch(self, force: bool = False, is_final: bool = False) -> None:
        """
        Save the current batch of results according to the accumulation strategy.
        
        Args:
            force: Whether to force saving regardless of internal conditions
            is_final: Whether this is the final save operation
        """
        pass
    
    @abstractmethod
    def get_final_results(self) -> pd.DataFrame:
        """
        Get the final accumulated results as a DataFrame.
        
        Returns:
            A DataFrame containing all accumulated results
        """
        pass


class DataFrameAccumulator(ResultAccumulator):
    """
    A specific implementation of ResultAccumulator that periodically saves results 
    to DataFrames and concatenates them at the end.
    """
    
    def __init__(self, batch_size: int):
        """
        Initialize the DataFrameAccumulator.
        
        Args:
            batch_size: Number of results to accumulate before saving to a DataFrame
        """
        self.batch_size = batch_size
        self.all_dataframes: List[pd.DataFrame] = []
        self.current_results: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single result to the accumulator.
        
        Args:
            result: A dictionary containing a single result
        """
        self.current_results.append(result)
        
        # Automatically save batch if we've reached the batch size
        if len(self.current_results) >= self.batch_size:
            self.save_batch()
    
    def save_batch(self, force: bool = False, is_final: bool = False) -> None:
        """
        Save the current batch of results to a DataFrame.
        
        Args:
            force: Whether to force saving regardless of batch size
            is_final: Whether this is the final save operation
        """
        if (force or is_final or len(self.current_results) >= self.batch_size) and self.current_results:
            # Convert accumulated results to dataframe and append to list
            chunk_df = pd.DataFrame(self.current_results)
            self.all_dataframes.append(chunk_df)
            self.current_results = []  # Reset the accumulator
    
    def get_final_results(self) -> pd.DataFrame:
        """
        Get the final accumulated results as a DataFrame.
        
        Returns:
            A DataFrame containing all accumulated results
        """
        # Save any remaining results
        self.save_batch(is_final=True)
        
        # Concatenate all dataframes
        if self.all_dataframes:
            return pd.concat(self.all_dataframes, ignore_index=True)
        return pd.DataFrame()


class DataChunkProcessor:
    """
    A class that processes data chunks and delegates result storage to an accumulator.
    """
    
    def __init__(self, accumulator: ResultAccumulator):
        """
        Initialize the DataChunkProcessor with a result accumulator.
        
        Args:
            accumulator: An implementation of ResultAccumulator to store results
        """
        self.accumulator = accumulator
    
    def process_chunk(self, chunk_idx: int, chunk_size: int) -> None:
        """
        Process a single data chunk and generate results.
        
        Args:
            chunk_idx: Index of the current chunk
            chunk_size: Size of the chunk to process
        """
        # Simulate processing time for each chunk
        time.sleep(0.001)  # Small delay to simulate processing
        
        # Generate results for this chunk and store using the accumulator
        for i in range(chunk_size):
            result = {
                'query_id': f"q{i % 100}",
                'chunk_id': chunk_idx,
                'score': np.random.random(),
                'rank': i % 10,
                'result': f"result_{chunk_idx}_{i}"
            }
            self.accumulator.add_result(result)
    
    def process_all_chunks(self, num_chunks: int, chunk_size: int, save_interval: Optional[int] = None) -> pd.DataFrame:
        """
        Process all chunks and return the final dataframe.
        
        Args:
            num_chunks: Number of data chunks to process
            chunk_size: Size of each chunk
            save_interval: Optional override for when to save results
            
        Returns:
            Final dataframe with all results
        """
        # Simulate processing chunks
        for chunk_idx in range(num_chunks):
            self.process_chunk(chunk_idx, chunk_size)
            
            # Check if we've reached the save interval (if specified)
            if save_interval and (chunk_idx + 1) % save_interval == 0:
                self.accumulator.save_batch(force=True)
        
        # Return the final results
        return self.accumulator.get_final_results()


def method1_preallocated_dataframe(num_chunks: int, chunk_size: int) -> float:
    """
    Method 1: Pre-allocate a dataframe and fill it with results.
    
    Args:
        num_chunks: Number of data chunks to process
        chunk_size: Size of each chunk
    
    Returns:
        Execution time in seconds
    """
    start_time = time.time()
    
    # Initialize a dataframe with empty values
    total_rows = num_chunks * chunk_size
    columns = ['query_id', 'chunk_id', 'score', 'rank', 'result']
    df = pd.DataFrame(index=range(total_rows), columns=columns)
    
    # Simulate processing chunks and filling the dataframe
    for chunk_idx in range(num_chunks):
        # Simulate processing time for each chunk
        time.sleep(0.001)  # Small delay to simulate processing
        
        # Generate results for this chunk
        for i in range(chunk_size):
            row_idx = chunk_idx * chunk_size + i
            df.loc[row_idx, 'query_id'] = f"q{i % 100}"
            df.loc[row_idx, 'chunk_id'] = chunk_idx
            df.loc[row_idx, 'score'] = np.random.random()
            df.loc[row_idx, 'rank'] = i % 10
            df.loc[row_idx, 'result'] = f"result_{chunk_idx}_{i}"
    
    end_time = time.time()
    return end_time - start_time


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
    
    # Create the accumulator and processor with dependency injection
    accumulator = DataFrameAccumulator(batch_size=chunk_size * save_interval)
    processor = DataChunkProcessor(accumulator)
    
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
    
    # Run trials for method 1
    for chunks in num_chunks_list:
        for _ in range(num_trials):
            time_method1 = method1_preallocated_dataframe(chunks, chunk_size)
            results['method1'][chunks].append(time_method1)
    
    # Run trials for method 2 with different save intervals
    for chunks in num_chunks_list:
        for interval in save_intervals:
            for _ in range(num_trials):
                time_method2 = method2_accumulate_and_concat(chunks, chunk_size, interval)
                results['method2'][f"{chunks}_interval_{interval}"].append(time_method2)
    
    return results


def plot_results(results: Dict[str, Dict[str, List[float]]], chunk_size: int) -> None:
    """
    Plot the performance comparison results.
    
    Args:
        results: Dictionary containing execution times for each method
        chunk_size: Size of each chunk used in the test
    """
    plt.figure(figsize=(12, 8))
    
    # Extract and plot method 1 results
    method1_chunks = sorted(list(results['method1'].keys()))
    method1_times = [np.mean(results['method1'][chunks]) for chunks in method1_chunks]
    plt.plot(method1_chunks, method1_times, marker='o', label='Method 1: Pre-allocated DataFrame')
    
    # Extract and group method 2 results by save interval
    method2_keys = sorted(list(results['method2'].keys()))
    intervals = set()
    for key in method2_keys:
        chunks_str, interval_str = key.split('_interval_')
        intervals.add(int(interval_str))
    
    # Plot method 2 results for each interval
    for interval in sorted(intervals):
        chunks_list = []
        times_list = []
        for chunks in method1_chunks:  # Use same chunks as method 1
            key = f"{chunks}_interval_{interval}"
            if key in results['method2']:
                chunks_list.append(chunks)
                times_list.append(np.mean(results['method2'][key]))
        
        plt.plot(chunks_list, times_list, marker='s', 
                label=f'Method 2: Accumulate & Concat (interval={interval})')
    
    plt.xlabel('Number of Chunks')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Comparison (Chunk Size: {chunk_size})')
    plt.legend()
    plt.grid(True)
    plt.savefig('method_comparison.png')
    plt.close()


def run_memory_usage_test(
    num_chunks: int, 
    chunk_size: int, 
    save_interval: int
) -> Dict[str, float]:
    """
    Measure memory usage for both methods.
    
    Args:
        num_chunks: Number of data chunks to process
        chunk_size: Size of each chunk
        save_interval: Save interval for method 2
    
    Returns:
        Dictionary containing peak memory usage for each method
    """
    import tracemalloc
    
    memory_usage = {}
    
    # Measure memory for method 1
    tracemalloc.start()
    method1_preallocated_dataframe(num_chunks, chunk_size)
    current, peak = tracemalloc.get_traced_memory()
    memory_usage['method1'] = peak / 1024 / 1024  # Convert to MB
    tracemalloc.stop()
    
    # Measure memory for method 2
    tracemalloc.start()
    method2_accumulate_and_concat(num_chunks, chunk_size, save_interval)
    current, peak = tracemalloc.get_traced_memory()
    memory_usage['method2'] = peak / 1024 / 1024  # Convert to MB
    tracemalloc.stop()
    
    return memory_usage


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
    
    # Plot results
    plot_results(results, chunk_size)
    print(f"Plot saved as 'method_comparison.png'")
    
    # Run memory usage test for the largest configuration
    print("\nRunning memory usage test...")
    memory_usage = run_memory_usage_test(1000, chunk_size, 50)
    print("\nPeak Memory Usage:")
    for method, usage in memory_usage.items():
        print(f"{method}: {usage:.2f} MB")

if __name__ == "__main__":
    main()
