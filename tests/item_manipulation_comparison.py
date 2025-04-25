"""
Compare the performances of different item manipulations for data processing.

Methods Compared:
1. Method 1 (Dict conversion): 
    从csv file中读取一个chunk 到dataframe中，然后将dataframe items转换为dict，
    然后与模型分析这些items，分析结果dict合并，保存到一个 dict list中，最后将dict list转成为dataframe

2. Method 2 (DataFrame direct): 
    从csv file中读取一个chunk 到dataframe中，然后模型分析dataframe items，
    然后将模型分析的结果写回到dataframe中
    
3. Method 2-Opt (iloc+loc): DataFrame direct method with iloc/loc instead of iterrows/at
    
4. Method 3 (Pandas apply): Using pandas apply function for vectorized operations
    
5. Method 3-Opt (direct array): Optimized pandas method using pre-allocated arrays
    
6. Method 4 (Parallel processing): Using multiprocessing to process items in parallel
    
7. Method 4-Opt (chunked MP): Optimized parallel processing with better chunk handling
    
8. Method 5 (Hybrid approach): Combining dict efficiency with parallel processing

实验结果:
Method 1 (Dict conversion): 1.1760 seconds (avg)
Method 2 (DataFrame direct): 1.6746 seconds (avg)
Method 2-Opt (iloc+loc): 2.5153 seconds (avg)
Method 3 (Pandas apply): 1.2721 seconds (avg)
Method 3-Opt (direct array): 1.5358 seconds (avg)
Method 4 (Parallel processing): 1.3566 seconds (avg)
Method 4-Opt (chunked MP): 1.3730 seconds (avg)
Method 5 (Hybrid approach): 1.4815 seconds (avg)

结论:
1. 最快的方法是 Method 1 (Dict conversion) - 简单的字典转换方法
2. 所有优化后的方法实际上比原始方法更慢（负优化）
3. 尝试使用 iloc/loc 替代 iterrows/at 导致了最严重的性能下降
4. 并行处理的开销超过了其潜在的性能收益
5. 对于此类数据处理任务，最简单的方法往往是最高效的

要求：采用高性能的方式，但不能过于复杂 - 实验证明简单的方法（Method 1）性能最佳

"""

import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from functools import partial
import os
import matplotlib.pyplot as plt


# Mock function to simulate model analysis
def mock_model_analysis(item):
    """Simulate model analysis on an item"""
    # Simulate some computational work
    time.sleep(0.001)  # Simulate model latency
    # Return a mock analysis result
    return {"sentiment": np.random.choice(["positive", "negative", "neutral"]),
            "confidence": np.random.random()}


# Generate sample data for testing
def generate_sample_data(rows=1000, output_file="sample_data.csv"):
    """Generate a sample CSV dataset for testing"""
    df = pd.DataFrame({
        "id": range(rows),
        "text": [f"Sample text {i}" for i in range(rows)],
        "category": np.random.choice(["A", "B", "C"], size=rows)
    })
    df.to_csv(output_file, index=False)
    return output_file


# Method 1: DataFrame → dict → process → dict list → DataFrame
def method1(chunk_df):
    """Original Method 1 approach"""
    # Convert DataFrame to list of dicts
    items = chunk_df.to_dict('records')
    
    # Process each item with model
    results = []
    for item in items:
        # Model analysis
        analysis_result = mock_model_analysis(item)
        
        # Merge original item with analysis results
        item.update(analysis_result)
        results.append(item)
    
    # Convert results back to DataFrame
    return pd.DataFrame(results)


# Method 2: DataFrame → process → update DataFrame
def method2(chunk_df):
    """Original Method 2 approach"""
    # Create a copy to avoid modifying the original
    result_df = chunk_df.copy()
    
    # Process each row and update DataFrame
    for idx, row in result_df.iterrows():
        analysis_result = mock_model_analysis(row)
        for key, value in analysis_result.items():
            result_df.at[idx, key] = value
    
    return result_df

# Optimized Method 2: Using loc for better performance
def method2_optimized(chunk_df):
    """Optimized Method 2 approach using loc instead of iterrows"""
    # Create a copy to avoid modifying the original
    result_df = chunk_df.copy()
    
    # Pre-allocate columns for results to avoid dynamic column creation
    result_df['sentiment'] = None
    result_df['confidence'] = None
    
    # Process each row by index (faster than iterrows)
    for idx in range(len(result_df)):
        # Get row as Series
        row = result_df.iloc[idx]
        
        # Model analysis
        analysis_result = mock_model_analysis(row)
        
        # Update values using loc (faster than at for batch updates)
        result_df.loc[idx, 'sentiment'] = analysis_result['sentiment']
        result_df.loc[idx, 'confidence'] = analysis_result['confidence']
    
    return result_df


# Method 3: DataFrame with apply/vectorization
def method3(chunk_df):
    """Method using pandas apply function"""
    # Create a copy to avoid modifying the original
    result_df = chunk_df.copy()
    
    # Apply model analysis to each row and create new columns
    analysis_results = result_df.apply(mock_model_analysis, axis=1)
    
    # Extract results and add to DataFrame
    result_df['sentiment'] = analysis_results.apply(lambda x: x['sentiment'])
    result_df['confidence'] = analysis_results.apply(lambda x: x['confidence'])
    
    return result_df

# Optimized Method 3: Using apply with more efficient extraction
def method3_optimized(chunk_df):
    """Optimized method using pandas apply with direct column creation"""
    # Create a copy to avoid modifying the original
    result_df = chunk_df.copy()
    
    # Pre-allocate result arrays for better performance
    sentiments = []
    confidences = []
    
    # Apply model analysis and collect results (avoids second apply operation)
    for _, row in chunk_df.iterrows():
        result = mock_model_analysis(row)
        sentiments.append(result['sentiment'])
        confidences.append(result['confidence'])
    
    # Assign in bulk (faster than individual assignments)
    result_df['sentiment'] = sentiments
    result_df['confidence'] = confidences
    
    return result_df


# Method 4: Parallel processing with multiprocessing
def process_item(item):
    """Worker function for parallel processing"""
    analysis_result = mock_model_analysis(item)
    item.update(analysis_result)
    return item

def method4(chunk_df, n_processes=None):
    """Method using parallel processing"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1 or 1
    
    # Convert DataFrame to records
    items = chunk_df.to_dict('records')
    
    # Process items in parallel
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_item, items)
    
    # Convert results back to DataFrame
    return pd.DataFrame(results)

# Optimized Method 4: Using more efficient parallelization
def method4_optimized(chunk_df, n_processes=None, chunksize=None):
    """Optimized parallel processing with better chunk handling"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1 or 1
    
    # Convert DataFrame to records (this is still needed for multiprocessing)
    items = chunk_df.to_dict('records')
    
    # Calculate optimal chunksize if not specified
    if chunksize is None:
        chunksize = max(1, len(items) // (n_processes * 4))
    
    # Process items in parallel with better chunk distribution
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_item, items, chunksize=chunksize)
    
    # Convert results back to DataFrame more efficiently
    return pd.DataFrame(results)

# Method 5: New approach - Hybrid method combining best of 1 and 4
def method5(chunk_df, n_processes=None, chunksize=None):
    """Hybrid method combining dict efficiency with parallel processing"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1 or 1
    
    if n_processes <= 1:
        # If only one process, use the efficient method1 approach
        return method1(chunk_df)
    
    # Convert DataFrame to records
    items = chunk_df.to_dict('records')
    
    # Calculate optimal chunksize if not specified
    if chunksize is None:
        chunksize = max(1, len(items) // (n_processes * 4))
    
    # Process items in parallel with optimized chunksize
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(process_item, items, chunksize=chunksize)
    
    # Convert results directly to DataFrame
    return pd.DataFrame(results)


# Run comparison experiment
def compare_methods(csv_file, chunk_size=1000, n_chunks=5):
    """Compare the performance of different methods"""
    results = {
        "Method 1 (Dict conversion)": [],
        "Method 2 (DataFrame direct)": [],
        "Method 2-Opt (iloc+loc)": [],
        "Method 3 (Pandas apply)": [],
        "Method 3-Opt (direct array)": [],
        "Method 4 (Parallel processing)": [],
        "Method 4-Opt (chunked MP)": [],
        "Method 5 (Hybrid approach)": []
    }
    
    # Read the CSV in chunks and process each chunk
    for i, chunk_df in enumerate(pd.read_csv(csv_file, chunksize=chunk_size)):
        if i >= n_chunks:
            break
        
        print(f"Processing chunk {i+1}/{n_chunks}")
        
        # Method 1
        start_time = time.time()
        result1 = method1(chunk_df)
        results["Method 1 (Dict conversion)"].append(time.time() - start_time)
        
        # Method 2
        start_time = time.time()
        result2 = method2(chunk_df)
        results["Method 2 (DataFrame direct)"].append(time.time() - start_time)
        
        # Method 2 Optimized
        start_time = time.time()
        result2_opt = method2_optimized(chunk_df)
        results["Method 2-Opt (iloc+loc)"].append(time.time() - start_time)
        
        # Method 3
        start_time = time.time()
        result3 = method3(chunk_df)
        results["Method 3 (Pandas apply)"].append(time.time() - start_time)
        
        # Method 3 Optimized
        start_time = time.time()
        result3_opt = method3_optimized(chunk_df)
        results["Method 3-Opt (direct array)"].append(time.time() - start_time)
        
        # Method 4
        start_time = time.time()
        result4 = method4(chunk_df)
        results["Method 4 (Parallel processing)"].append(time.time() - start_time)
        
        # Method 4 Optimized
        start_time = time.time()
        result4_opt = method4_optimized(chunk_df)
        results["Method 4-Opt (chunked MP)"].append(time.time() - start_time)
        
        # Method 5
        start_time = time.time()
        result5 = method5(chunk_df)
        results["Method 5 (Hybrid approach)"].append(time.time() - start_time)
        
        # Verify all methods produce equivalent results
        assert set(result1.columns) == set(result2.columns) == set(result2_opt.columns) == set(result3.columns) == \
               set(result3_opt.columns) == set(result4.columns) == set(result4_opt.columns) == set(result5.columns), \
               "Methods produced different columns"
    
    return results


def plot_results(results):
    """Plot the performance comparison results"""
    # Calculate average time for each method
    avg_times = {method: np.mean(times) for method, times in results.items()}
    
    methods = list(avg_times.keys())
    times = list(avg_times.values())
    
    plt.figure(figsize=(10, 6))
    plt.barh(methods, times, color='skyblue')
    plt.xlabel('Average Processing Time (seconds)')
    plt.title('Performance Comparison of Data Processing Methods')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(times):
        plt.text(v + 0.01, i, f"{v:.4f}s", va='center')
    
    plt.tight_layout()
    plt.savefig('method_comparison_results.png')
    plt.close()


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample data if needed
    csv_file = "sample_data.csv"
    if not os.path.exists(csv_file):
        csv_file = generate_sample_data(rows=10000, output_file=csv_file)
    
    # Run the comparison
    results = compare_methods(csv_file, chunk_size=1000, n_chunks=10)
    
    # Print results
    print("\nPerformance Results:")
    for method, times in results.items():
        avg_time = np.mean(times)
        print(f"{method}: {avg_time:.4f} seconds (avg)")
    
    # Determine the fastest method
    fastest_method = min(results.items(), key=lambda x: np.mean(x[1]))
    print(f"\nFastest method: {fastest_method[0]} with average time {np.mean(fastest_method[1]):.4f} seconds")
    
    # Plot the results
    plot_results(results)
    print("\nResults visualization saved to 'method_comparison_results.png'")
