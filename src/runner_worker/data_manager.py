"""
Main data flow: 
- update_result2df: update result columns in chunk df (in-memory)
- save_chunk_results: save chunk df to temp file (disk)
- save_results: interface for saving results in other classes
- merge_temp_files: merge temp files into final file
"""

import os
from typing import Dict, Iterator, Tuple
import pandas as pd
from src.runner_worker.checkpoint_manager import RobustCheckpointManager as CheckpointManager

class ChunkedDataManager:
    """
    负责分块读取和处理大型CSV文件，优化内存使用。
    
    主要职责:
    - 高效分块读取大型CSV文件，避免将全部数据加载到内存
    - 提供行级访问接口，支持随机访问特定行
    - 管理数据更新和保存，先将处理后的chunk保存到临时文件，在文件名中添加编号
    - 最后将临时文件合并成完整的新的CSV文件，使用原始文件名+实验编号
    - 若提供了experiment_id，则使用断点续处理方法检测任务进度

    数据流向:
    - 输入: 从CSV文件读取原始数据，请勿修改原始数据
    - 输出: 向ParallelProcessor提供数据块，接收并保存处理结果
    """
    
    def __init__(self, csv_path, chunksize=10000, result_columns=None, experiment_id=None):
        """
        初始化数据管理器。
        
        Args:
            csv_path (str): CSV文件路径
            chunksize (int): 每次读取的数据块大小，影响内存使用量
            result_columns (list, optional): 存储分析结果的列名列表，默认为语义匹配模型结果列
            experiment_id (str, optional): 实验ID，用于生成输出文件名，默认使用时间戳
        """
        import time
        
        self.csv_path = csv_path
        self.chunksize = chunksize
        
        # default result columns, containing matching results from different models
        self.result_columns = result_columns or [
            'word2vec_match_res',  # Word2Vec model matching results
            'bge_large_match_res', # BGE Large model matching results
            'bge_m3_match_res'     # BGE M3 model matching results
        ]
        
        # check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        # calculate total rows
        self.total_rows = self._count_rows()
        
        # set experiment ID and temporary file directory
        if experiment_id:
            self.experiment_id = experiment_id
            # check if temp dir exists
            temp_dir = os.path.join(os.path.dirname(csv_path), f"temp_{experiment_id}")
            if os.path.exists(temp_dir):
                # initialize from checkpoint
                self.temp_dir = temp_dir
                self.checkpoint_manager = CheckpointManager(temp_dir)
                self.processed_indices = self.checkpoint_manager.get_processed_indices()
            else:
                # start a new experiment
                self.experiment_id = experiment_id
                self.temp_dir = temp_dir
                self.checkpoint_manager = CheckpointManager(temp_dir)
                self.processed_indices = set()
        else:
            # start a new experiment
            self.experiment_id = f"exp_{int(time.time())}"
            self.temp_dir = os.path.join(os.path.dirname(csv_path), f"temp_{self.experiment_id}")
            self.checkpoint_manager = CheckpointManager(self.temp_dir)
            self.processed_indices = set()
        
        # create temp directory
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # output file path
        filename = os.path.basename(csv_path)
        base_name, ext = os.path.splitext(filename)
        self.output_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_{self.experiment_id}{ext}")
        
        # global index column name
        self.global_index_column = 'global_index'

        # Only save results for the current chunk
        self.chunk_results_cache = {}
        
    
    def _count_rows(self):
        """Calculate the total number of rows in the CSV file for progress tracking."""
        
        try:
            # Use pandas to quickly calculate the number of rows
            with pd.read_csv(self.csv_path, chunksize=self.chunksize) as reader:
                return sum(len(chunk) for chunk in reader)
        except Exception as e:
            print(f"Error calculating rows: {e}")
            # Use backup method to calculate rows
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1  # Subtract the header row
        
    def get_chunks(self) -> Iterator[Tuple[int, pd.DataFrame]]:
        """
        Returns an iterator of CSV chunks with a global index column.
        
        Returns:
            iterator: Iterator of (start_index, DataFrame) tuples
        """
        
        try:
            # read CSV file in chunks
            start_index = 0
            with pd.read_csv(self.csv_path, chunksize=self.chunksize) as reader:
                for chunk in reader:
                    # add global index column
                    chunk[self.global_index_column] = range(start_index, start_index + len(chunk))
                    
                    # add result columns
                    for col in self.result_columns:
                        if col not in chunk.columns:
                            chunk[col] = None
                    
                    yield start_index, chunk
                    start_index += len(chunk)
        except Exception as e:
            print(f"Error getting data chunk: {e}")
            raise
    
    def get_row(self, global_index: int) -> pd.Series:
        """
        Returns a row from the CSV file based on the global index.
        
        Args:
            global_index (int): Global index of the row
            
        Returns:
            pandas.Series: Row data
        """
        
        try:
            # calculate row position
            chunk_index = global_index // self.chunksize
            local_index = global_index % self.chunksize
            
            # read header row to get correct column names
            headers = pd.read_csv(self.csv_path, nrows=1).columns.tolist()
            
            # calculate rows to skip (skip header and previous chunks)
            skiprows = 1 + chunk_index * self.chunksize
            
            # read specific chunk, without header
            df = pd.read_csv(self.csv_path, skiprows=skiprows, nrows=self.chunksize, header=None)
            
            # set correct column names
            df.columns = headers
            
            # return specific row
            if local_index < len(df):
                row = df.iloc[local_index].copy()
                # add global index
                row[self.global_index_column] = global_index
                return row
            else:
                raise IndexError(f"Row index {global_index} out of range")
        except Exception as e:
            print(f"Error getting row {global_index}: {e}")
            raise
    
    def get_unprocessed_chunks(self, processed_indices):
        """
        Returns unprocessed data chunks, skipping processed rows.
        
        Works with CheckpointManager to ensure only new data is processed.
        
        Args:
            processed_indices (set): Set of processed row indices
            
        Returns:
            list: List of (start_index, DataFrame) tuples
        """
        
        unprocessed_chunks = []
        
        try:
            # 遍历所有数据块
            for start_index, chunk in self.get_chunks():
                # 创建全局索引范围
                global_indices = range(start_index, start_index + len(chunk))
                
                # 找出未处理的行索引
                unprocessed_mask = [i not in processed_indices for i in global_indices]
                
                # 如果块中有未处理的行
                if any(unprocessed_mask):
                    # 提取未处理的行
                    unprocessed_rows = chunk.iloc[[i for i, mask in enumerate(unprocessed_mask) if mask]].copy()
                    
                    # 如果有未处理的行，添加到结果列表
                    if len(unprocessed_rows) > 0:
                        # 计算未处理行的全局起始索引
                        first_unprocessed_idx = next(i for i, mask in enumerate(unprocessed_mask) if mask)
                        unprocessed_start = start_index + first_unprocessed_idx
                        
                        unprocessed_chunks.append((unprocessed_start, unprocessed_rows))
            
            return unprocessed_chunks
        except Exception as e:
            print(f"获取未处理数据块时出错: {e}")
            raise
    
    def _update_result2df(self, chunk_df: pd.DataFrame, results_dict: Dict[int, Dict[str, str]]):
        """
        更新DataFrame中的结果列
        
        Args:
            chunk_df (pandas.DataFrame): 数据块DataFrame
            results_dict (dict): 索引到结果的映射字典，格式为 {行索引: {结果列名: 结果值}}
        
        Returns:
            pandas.DataFrame: 更新后的DataFrame
        """
        for global_idx, result in results_dict.items():
            # 先找到对应的行
            local_idx = chunk_df[chunk_df[self.global_index_column] == global_idx].index
            if len(local_idx) > 0:
                # 再更新结果列
                for col_name, value in result.items():
                    if col_name in self.result_columns:
                        chunk_df.loc[local_idx[0], col_name] = value
        return chunk_df
    
    def save_chunk_results(self, 
        chunk_index: int, 
        chunk_df: pd.DataFrame, 
        results_dict: Dict[int, Dict[str, str]] = {}
    ):
        """
        将处理完成的数据块保存到临时文件。
        
        Args:
            chunk_index (int): 数据块索引
            chunk_df (pandas.DataFrame): 已包含处理结果的数据块DataFrame
            results_dict (dict): 额外的结果映射字典，如果非空，会在保存前更新到DataFrame
                                格式为 {行索引: {结果列名: 结果值}}
            
        Returns:
            tuple: (临时文件路径, 保存的DataFrame)
        """
        
        # 创建临时文件路径
        temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_index}.csv")
        
        try:
            # 如果有额外的结果需要更新，则更新DataFrame
            if results_dict:
                chunk_df = self._update_result2df(chunk_df, results_dict)
            
            # 保存到临时文件
            chunk_df.to_csv(temp_file, index=False)
            
            return temp_file, chunk_df
        except Exception as e:
            print(f"保存数据块结果时出错: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise
    
    def save_results(self, 
        chunk_idx: int,
        results_dict: Dict[int, Dict], 
        output_path: str = None
    ):
        """
        将处理结果保存到CSV文件，不修改原始数据文件，而是创建新文件。
        判断chunks 是否处理完毕，如果处理完毕，将临时文件合并到输出文件。
        内部调用save_chunk_results and merge_temp_files
        
        Args:
            chunk_idx (int): 数据块索引
            results_dict (dict): 索引到结果的映射字典，格式为 {global index: {结果列名: 结果值}}，其长度可以整除chunksize
                注意： 默认global indices 在dict中是连续的，之前的indices已处理，所以其长度只是chunksize的一部分
            output_path (str, optional): 输出文件路径，默认使用构造函数中设置的输出路径
            
        Returns:
            bool: 保存是否成功
        """

        try:
            # 计算当前chunk的起始和结束索引
            chunk_start_idx = chunk_idx * self.chunksize
            chunk_end_idx = min((chunk_idx + 1) * self.chunksize, self.total_rows)
            
            # 如果这是当前chunk的第一批结果，初始化缓存
            if chunk_idx not in self.chunk_results_cache:
                # 直接读取对应的chunk，更高效
                for start_idx, chunk_df in self.get_chunks():
                    if start_idx == chunk_start_idx:
                        self.chunk_results_cache[chunk_idx] = chunk_df
                        break
                
                # 如果没有找到对应的chunk，返回失败
                if chunk_idx not in self.chunk_results_cache:
                    print(f"错误：找不到索引为 {chunk_idx} 的数据块")
                    return False
            
            # 更新当前chunk的结果到缓存
            if results_dict:
                self.chunk_results_cache[chunk_idx] = self._update_result2df(
                    self.chunk_results_cache[chunk_idx], 
                    results_dict
                )
                
                # 先更新内存中的集合    
                self.processed_indices.update(results_dict.keys())

                # # 每处理 N 个索引就批量保存一次检查点
                # if len(self.processed_indices) % 1000 == 0:  # 可以根据实际情况调整批量大小
                #     self.checkpoint_manager.save_checkpoint(self.processed_indices)
                # self.checkpoint_manager.add_processed_index(global_idx)
            
            # 判断当前chunk是否处理完毕
            max_result_index = max(results_dict.keys()) if results_dict else -1
            chunk_complete = max_result_index >= chunk_end_idx - 1
            
            # 判断整个数据集是否处理完毕
            last_chunk_idx = (self.total_rows - 1) // self.chunksize
            dataset_complete = chunk_complete and chunk_idx == last_chunk_idx
            
            # 保存当前chunk的结果到临时文件
            temp_file = None
            if chunk_complete and self.chunk_results_cache.get(chunk_idx) is not None:
                # 使用current chunk结果字典保存完整的chunk
                temp_file, _ = self.save_chunk_results(
                    chunk_idx, 
                    self.chunk_results_cache[chunk_idx], 
                    # {}  # 空字典，因为结果已经在缓存中更新
                )
                
                # 清除缓存
                del self.chunk_results_cache[chunk_idx]
            
            # 根据处理状态决定是否合并文件
            if temp_file:
                if dataset_complete:
                    output_path = output_path or self.output_path
                    # 如果整个数据集处理完毕，合并所有临时文件到输出文件
                    self.merge_temp_files(output_path)
                    print(f"整个数据集处理完毕，已合并结果到文件: {output_path}")
                    return True
                else:
                    # 如果当前chunk处理完毕但整个数据集未完成，保留临时文件等待后续合并
                    print(f"当前数据块 {chunk_idx} 处理完毕，临时文件已保存：{temp_file}")
                    return True
            else:
                # 如果没有保存文件，说明结果已缓存但尚未达到保存条件
                if chunk_idx in self.chunk_results_cache:
                    processed_count = len([i for i in range(chunk_start_idx, chunk_end_idx) 
                                          if i in self.processed_indices])
                    total_count = chunk_end_idx - chunk_start_idx
                    print(f"当前数据块 {chunk_idx} 部分结果已缓存，已处理 {processed_count}/{total_count} 条结果")
                return True
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
            return False
    
    def merge_temp_files(self, output_path):
        """
        合并所有临时文件到一个输出文件。
        
        Args:
            output_path (str): 输出文件路径
        """
        import glob
        
        # 获取所有临时文件，按块索引排序
        temp_files = sorted(glob.glob(os.path.join(self.temp_dir, "chunk_*.csv")), 
                           key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        
        if not temp_files:
            print("没有找到临时文件，无法合并")
            return
        
        try:
            # 读取第一个文件以获取列名
            first_df = pd.read_csv(temp_files[0])
            headers = first_df.columns.tolist()
            
            # 打开输出文件
            with open(output_path, 'w', encoding='utf-8') as outfile:
                # 写入标题行
                outfile.write(','.join(headers) + '\n')
                
                # 逐个处理临时文件
                for i, temp_file in enumerate(temp_files):
                    # 跳过第一个文件的标题行
                    with open(temp_file, 'r', encoding='utf-8') as infile:
                        next(infile)  # 跳过标题行
                        # 复制内容
                        for line in infile:
                            outfile.write(line)
            
            print(f"已合并 {len(temp_files)} 个临时文件到 {output_path}")
            
        except Exception as e:
            print(f"合并临时文件时出错: {e}")
            raise
    
    def cleanup_temp_files(self):
        """清理临时文件和目录"""
        import shutil
        
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"已清理临时目录: {self.temp_dir}")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
