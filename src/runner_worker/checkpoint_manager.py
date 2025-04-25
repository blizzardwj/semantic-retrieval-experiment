from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import pandas as pd
import time
from pathlib import Path

# 首先定义ResultAccumulator的抽象接口
class ResultAccumulator(ABC):
    """
    Abstract base class for result accumulation strategies.
    Defines the interface for storing, accumulating and retrieving results.
    """
    
    @abstractmethod
    def create_experiment_tempdir(self, experiment_id: str, csv_path: str) -> str:
        """
        Create a temporary directory for the experiment.
        
        Args:
            experiment_id: ID of the experiment
            csv_path: Path to the original CSV file
        
        Returns:
            str: Path to the temporary directory
        """
        pass

    @abstractmethod
    def add_result(self, chunk_idx: int, result: Dict[str, Any]) -> None:
        """
        Add a single result to the accumulator.
        
        Args:
            chunk_idx: Index of the chunk
            result: A dictionary containing a single result
        """
        pass
    
    @abstractmethod
    def save_chunk(self, force: bool = False, is_final: bool = False) -> None:
        """
        Save the current chunk of results according to the accumulation strategy.
        
        Args:
            force: Whether to force saving regardless of internal conditions
            is_final: Whether this is the final save operation
        """
        pass

    @abstractmethod
    def save_chunk_to_file(self, chunk_index: int) -> str:
        """
        Save the current chunk of results to a file.
        
        Args:
            chunk_index: Index of the chunk to save
        
        Returns:
            str: Path to the saved file
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
    
    def __init__(self, chunk_size: int, saver: Callable[[pd.DataFrame], str] = None):
        """
        Initialize the DataFrameAccumulator.
        
        Args:
            chunk_size: Number of results to accumulate before saving to a DataFrame
        """
        self.chunk_size = chunk_size
        self.completed_chunk_df: pd.DataFrame = pd.DataFrame()
        self.current_results: List[Dict[str, Any]] = []
        self.saver = saver or self.default_saver
    
    def default_saver(self, chunk_index: int, completed_chunk_df: pd.DataFrame) -> str:
        """
        将处理完成的数据块保存到临时文件。
        
        Args:
            chunk_index (int): 数据块索引
            
        Returns:
            str: 临时文件路径
        """
        
        # 创建临时文件路径
        temp_file = Path(self.temp_dir) / f"chunk_{chunk_index}.csv"
        
        try:
            # 保存到临时文件
            completed_chunk_df.to_csv(temp_file, index=False)
            
            return temp_file
        except Exception as e:
            print(f"保存数据块结果时出错: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def create_experiment_tempdir(self, experiment_id: str, csv_path: str) -> str:
        """
        Create a temporary directory for the experiment.
        
        Args:
            experiment_id: ID of the experiment
            csv_path: Path to the CSV file
            
        Returns:
            str: Path to the temporary directory
        """
        # set experiment ID and temporary file directory
        if experiment_id:
            self.experiment_id = experiment_id
            # check if temp dir exists
            temp_dir = Path(csv_path).parent / f"temp_{experiment_id}"
            if temp_dir.exists():
                # initialize from checkpoint
                self.temp_dir = temp_dir
            else:
                # start a new experiment
                self.temp_dir = temp_dir
        else:
            # start a new experiment
            self.experiment_id = f"exp_{int(time.time())}"
            self.temp_dir = Path(csv_path).parent / f"temp_{self.experiment_id}"
        
        return self.temp_dir

    def add_result(self, chunk_idx: int, is_last_chunk: bool, result: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
        """
        Add a single result to the accumulator dict list.
        
        Args:
            chunk_idx: Index of the chunk
            result: A dictionary containing a single result
        """
        if isinstance(result, list):
            self.current_results.extend(result)
        else:
            self.current_results.append(result)
        
        # Automatically save chunk if we've reached the chunk size
        if len(self.current_results) >= self.chunk_size or is_last_chunk:
            self.save_chunk(is_final=is_last_chunk)
            self.saver(chunk_idx, self.completed_chunk_df)
    
    def save_chunk(self, force: bool = False, is_final: bool = False) -> None:
        """
        Save the current chunk of results to a DataFrame.
        
        Args:
            force: Whether to force saving regardless of chunk size
            is_final: Whether this is the final save operation
        """
        if (force or is_final or len(self.current_results) >= self.chunk_size) and self.current_results:
            # Convert accumulated results to dataframe and append to list
            self.completed_chunk_df = pd.DataFrame(self.current_results)
            self.current_results = []  # Reset the accumulator

    def get_final_results(self) -> pd.DataFrame:
        """
        Get the final accumulated results as a DataFrame.
        
        Returns:
            A DataFrame containing all accumulated results
        """
        # Save any remaining results
        self.save_chunk(is_final=True)
        
        # Concatenate all dataframes
        if self.all_dataframes:
            return pd.concat(self.all_dataframes, ignore_index=True)
        return pd.DataFrame()