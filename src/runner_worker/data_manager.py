import os
from typing import Dict
import pandas as pd

class ChunkedDataManager:
    """
    负责分块读取和处理大型CSV文件，优化内存使用。
    
    主要职责:
    - 高效分块读取大型CSV文件，避免将全部数据加载到内存
    - 提供行级访问接口，支持随机访问特定行
    - 管理数据更新和保存，先将处理后的chunk保存到临时文件，在文件名中添加编号
    - 最后将临时文件合并成完整的新的CSV文件，使用原始文件名+实验编号
    
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
        
        # 默认结果列名，包含不同模型的匹配结果
        self.result_columns = result_columns or [
            'word2vec_match_res',  # Word2Vec模型匹配结果
            'bge_large_match_res', # BGE Large模型匹配结果
            'bge_m3_match_res'     # BGE M3模型匹配结果
        ]
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
            
        # 计算总行数
        self.total_rows = self._count_rows()
        
        # 设置实验ID和临时文件目录
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.temp_dir = os.path.join(os.path.dirname(csv_path), f"temp_{self.experiment_id}")
        
        # 创建临时目录
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        # 输出文件路径
        filename = os.path.basename(csv_path)
        base_name, ext = os.path.splitext(filename)
        self.output_path = os.path.join(os.path.dirname(csv_path), f"{base_name}_{self.experiment_id}{ext}")
        
        # 全局索引列名
        self.global_index_column = 'global_index'
        
    def _count_rows(self):
        """计算CSV文件的总行数，用于进度跟踪"""
        
        try:
            # 使用pandas快速计算行数
            with pd.read_csv(self.csv_path, chunksize=self.chunksize) as reader:
                return sum(len(chunk) for chunk in reader)
        except Exception as e:
            print(f"计算行数时出错: {e}")
            # 使用备用方法计算行数
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1  # 减去标题行
        
    def get_chunks(self):
        """
        返回CSV数据块的迭代器，每个块添加全局索引列。
        
        Returns:
            iterator: 数据块迭代器，每个元素是(起始索引, DataFrame块)元组
        """
        
        try:
            # 读取CSV文件，分块返回
            start_index = 0
            with pd.read_csv(self.csv_path, chunksize=self.chunksize) as reader:
                for chunk in reader:
                    # 添加全局索引列
                    chunk[self.global_index_column] = range(start_index, start_index + len(chunk))
                    
                    # 添加结果列
                    for col in self.result_columns:
                        if col not in chunk.columns:
                            chunk[col] = None
                    
                    yield start_index, chunk
                    start_index += len(chunk)
        except Exception as e:
            print(f"获取数据块时出错: {e}")
            raise
    
    def get_row(self, global_index: int) -> pd.Series:
        """
        获取指定全局索引的行数据。
        
        Args:
            global_index (int): 行的全局索引
            
        Returns:
            pandas.Series: 指定行的数据
        """
        
        try:
            # 计算行所在的块和块内索引
            chunk_index = global_index // self.chunksize
            local_index = global_index % self.chunksize
            
            # 读取CSV文件的标题行，以获取正确的列名
            headers = pd.read_csv(self.csv_path, nrows=1).columns.tolist()
            
            # 计算要跳过的行数（跳过标题行和之前的块）
            skiprows = 1 + chunk_index * self.chunksize
            
            # 读取特定块，不使用标题行
            df = pd.read_csv(self.csv_path, skiprows=skiprows, nrows=self.chunksize, header=None)
            
            # 设置正确的列名
            df.columns = headers
            
            # 返回特定行
            if local_index < len(df):
                row = df.iloc[local_index].copy()
                # 添加全局索引
                row[self.global_index_column] = global_index
                return row
            else:
                raise IndexError(f"行索引 {global_index} 超出范围")
        except Exception as e:
            print(f"获取行 {global_index} 时出错: {e}")
            raise
    
    def get_unprocessed_chunks(self, processed_indices):
        """
        获取未处理的数据块，跳过已处理的行。
        
        与CheckpointManager协作，确保只处理新数据。
        
        Args:
            processed_indices (set): 已处理行的索引集合
            
        Returns:
            list: 包含(全局起始索引, DataFrame块)的元组列表
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
    
    def save_chunk_results(self, 
        chunk_index: int, 
        chunk_df: pd.DataFrame, 
        results_dict: Dict[int, Dict[str, str]]
    ):
        """
        将处理结果保存到临时文件。
        
        Args:
            chunk_index (int): 数据块索引
            chunk_df (pandas.DataFrame): 数据块DataFrame
            results_dict (dict): 索引到结果的映射字典，格式为 {行索引: {结果列名: 结果值}}
            
        Returns:
            str: 临时文件路径
        """
        
        # 创建临时文件路径
        temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_index}.csv")
        
        try:
            # 更新结果列
            for global_idx, result in results_dict.items():
                # 先找到对应的行
                local_idx = chunk_df[chunk_df[self.global_index_column] == global_idx].index
                if len(local_idx) > 0:
                    # 再更新结果列
                    for col_name, value in result.items():
                        if col_name in self.result_columns:
                            chunk_df.loc[local_idx[0], col_name] = value
            
            # 保存到临时文件
            chunk_df.to_csv(temp_file, index=False)
            
            return temp_file
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
            
            # 获取结果字典中的最大全局索引
            max_result_index = max(results_dict.keys()) if results_dict else -1
            
            # 判断当前chunk是否处理完毕 - 简化判断逻辑
            # 如果最大结果索引已经达到或超过了当前chunk的结束索引，则认为当前chunk处理完毕
            chunk_complete = max_result_index >= chunk_end_idx - 1
            
            # 判断整个数据集是否处理完毕
            # 如果当前chunk是最后一个chunk，且该chunk处理完毕，则认为整个数据集处理完毕
            last_chunk_idx = (self.total_rows - 1) // self.chunksize
            dataset_complete = chunk_complete and chunk_idx == last_chunk_idx
            
            # 保存当前chunk的结果到临时文件
            temp_file = None
            if results_dict:
                # 获取当前chunk的数据
                for _, (start_index, chunk_df) in enumerate(self.get_chunks()):
                    if start_index == chunk_start_idx:
                        # 找出属于当前数据块的结果
                        chunk_results = {}
                        for global_idx, result in results_dict.items():
                            # 检查索引是否在当前块范围内
                            if start_index <= global_idx < start_index + len(chunk_df):
                                chunk_results[global_idx] = result
                        
                        # 如果当前块有结果需要保存
                        if chunk_results:
                            temp_file = self.save_chunk_results(chunk_idx, chunk_df, chunk_results)
                        break
            
            # 根据处理状态决定是否合并文件
            if temp_file:
                if dataset_complete:
                    output_path = output_path or self.output_path
                    # 如果整个数据集处理完毕，合并所有临时文件到输出文件
                    self.merge_temp_files(output_path)
                    print(f"整个数据集处理完毕，已合并结果到文件: {output_path}")
                    return True
                elif chunk_complete:
                    # 如果当前chunk处理完毕但整个数据集未完成，保留临时文件等待后续合并
                    print(f"当前数据块 {chunk_idx} 处理完毕，临时文件已保存：{temp_file}")
                    return True
                else:
                    # 如果当前chunk未处理完毕，只保存临时文件
                    print(f"当前数据块 {chunk_idx} 部分处理，临时文件已保存：{temp_file}")
                    return True
            else:
                print("没有找到需要保存的结果")
                return False
            
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
