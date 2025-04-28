from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple
import os
import csv

# 首先定义 DataSource 抽象接口
class DataSource(ABC):
    """
    Abstract base class for data sources.
    Defines the interface for retrieving data chunks for processing.
    """
    
    @abstractmethod
    def get_chunk(self, chunk_idx: int, chunk_size: int) -> Tuple[int, pd.DataFrame]:
        """
        Get a chunk of data from the source.
        
        Args:
            chunk_idx: Index of the chunk to retrieve
            chunk_size: Size of the chunk to retrieve
            
        Returns:
            Iterator of dictionaries representing the data items
        """
        pass
    
    @abstractmethod
    def get_total_chunks(self, chunk_size: int) -> int:
        """
        Get the total number of chunks in this data source.
        
        Args:
            chunk_size: Size of each chunk
            
        Returns:
            Total number of chunks
        """
        pass


class InMemoryDataSource(DataSource):
    """
    A data source that generates synthetic data in memory.
    Used for testing and benchmarking.
    """
    
    def __init__(self, total_items: int):
        """
        Initialize the in-memory data source.
        
        Args:
            total_items: Total number of items to generate
        """
        self.total_items = total_items
    
    def get_chunk(self, chunk_idx: int, chunk_size: int) -> Tuple[int, pd.DataFrame]:
        """
        Generate a chunk of synthetic data.
        
        Args:
            chunk_idx: Index of the chunk to generate
            chunk_size: Size of the chunk to generate
            
        Returns:
            Iterator of dictionaries representing synthetic data items
        """
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, self.total_items)
        
        for i in range(start_idx, end_idx):
            yield {
                'query_id': f"q{i % 100}",
                'chunk_id': chunk_idx,
                'score': np.random.random(),
                'rank': i % 10,
                'result': f"result_{chunk_idx}_{i - start_idx}"
            }
    
    def get_total_chunks(self, chunk_size: int) -> int:
        """
        Get the total number of chunks based on total items.
        
        Args:
            chunk_size: Size of each chunk
            
        Returns:
            Total number of chunks
        """
        return (self.total_items + chunk_size - 1) // chunk_size  # Ceiling division


class CSVFileDataSource(DataSource):
    """
    A data source that reads data from a CSV file in chunks.
    """
    
    def __init__(self, file_path: str, transform_func=None):
        """
        Initialize the CSV file data source.
        
        Args:
            file_path: Path to the CSV file
            transform_func: Optional function to transform raw CSV rows into the desired format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
            
        self.file_path = file_path
        self.transform_func = transform_func
        
        # Count the number of lines in the file
        with open(file_path, 'r') as f:
            self.total_rows = sum(1 for _ in f)
        
        # Subtract 1 if there's a header row
        with open(file_path, 'r') as f:
            sample = f.readline()
            if csv.Sniffer().has_header(sample):
                self.total_rows -= 1
                self.has_header = True
            else:
                self.has_header = False
        
        # Global index column name
        self.global_index_column = 'global_index'   
    
    def get_chunk(self, chunk_idx: int, chunk_size: int) -> Tuple[int, pd.DataFrame, bool]:
        """
        Iteratively read a chunk of data from the CSV file.
        
        Args:
            chunk_idx: Start index of the chunk to read
            chunk_size: Size of the chunk to read
            
        Returns:
            start index of line and chunk data
        """
        # Calculate the starting line and how many lines to skip
        start_line = chunk_idx * chunk_size
        
        # If we have a header and this is not the first chunk, add 1 to account for header
        skip_rows = start_line
        if self.has_header and chunk_idx > 0:
            skip_rows += 1
        
        # Calculate how many rows to read
        rows_to_read = min(chunk_size, self.total_rows - start_line)
        if rows_to_read <= 0:
            return None, None, None  # No more data to read
        
        is_last_chunk = (start_line + rows_to_read >= self.total_rows)
        # Read the chunk from the CSV file
        chunk_df = pd.read_csv(
            self.file_path,
            skiprows=skip_rows if chunk_idx > 0 else None,  # Skip appropriate rows based on chunk
            nrows=rows_to_read,
            header=0 if (chunk_idx == 0 and self.has_header) else None
        )
        
        # If there's no header, assign column names
        if not self.has_header:
            chunk_df.columns = [f'col_{i}' for i in range(len(chunk_df.columns))]

        # Add global index column
        chunk_df[self.global_index_column] = range(start_line, start_line + len(chunk_df))
        
        return start_line, chunk_df, is_last_chunk
    
    def get_total_chunks(self, chunk_size: int) -> int:
        """
        Get the total number of chunks based on file size.
        
        Args:
            chunk_size: Size of each chunk
            
        Returns:
            Total number of chunks
        """
        return (self.total_rows + chunk_size - 1) // chunk_size  # Ceiling division