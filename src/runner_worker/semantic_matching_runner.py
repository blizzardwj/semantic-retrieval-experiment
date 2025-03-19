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

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.runner_worker.data_manager import ChunkedDataManager

class SemanticMatchingRunner:
    """
    语义匹配实验运行器，负责协调数据管理和模型运行。
    
    主要功能:
    - 使用ChunkedDataManager分块读取大型数据集
    - 对每个数据块执行不同模型的语义匹配
    - 将结果保存到新的CSV文件中
    """
    
    def __init__(self, 
                 data_path: str, 
                 chunksize: int = 10000, 
                 experiment_id: Optional[str] = None,
                 checkpoint_interval: int = 1000,
                 batch_size: int = 32,
                 n_jobs: int = 4):
        """
        初始化语义匹配运行器。
        
        Args:
            data_path: 数据文件路径
            chunksize: 每次处理的数据块大小
            experiment_id: 实验ID，用于标识输出文件
            checkpoint_interval: 保存检查点的间隔行数
            batch_size: 并行处理的批量大小
            n_jobs: 并行处理的作业数，-1表示使用所有可用CPU
        """
        self.data_path = data_path
        self.chunksize = chunksize
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.checkpoint_interval = checkpoint_interval
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        
        # 初始化数据管理器
        self.data_manager = ChunkedDataManager(
            csv_path=data_path,
            chunksize=chunksize,
            experiment_id=self.experiment_id
        )
        
        # 已处理的行索引集合，用于断点续传
        self.processed_indices: Set[int] = set()
        
        # 结果缓存，定期保存到文件
        self.results_cache: Dict[int, Dict[str, float]] = {}
        
        # 模型缓存，避免重复初始化
        self._fasttext_model = None
        self._bge_large_model = None
        self._bge_m3_model = None
        
        # 模型配置
        self._fasttext_model_path = "/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin"
        self._xinference_server_url = "http://20.30.80.200:9997"
        
        print(f"初始化语义匹配运行器，数据文件: {data_path}")
        print(f"数据集总行数: {self.data_manager.total_rows}")
        print(f"实验ID: {self.experiment_id}")
        print(f"并行配置: 批量大小={batch_size}, 作业数={n_jobs}")
    
    def _initialize_fasttext_model(self):
        """初始化FastText模型（如果尚未初始化）"""
        if self._fasttext_model is None:
            from semantic_retrieval.embedding.fasttext import FastTextEmbedding
            print("初始化FastText模型...")
            self._fasttext_model = FastTextEmbedding(model_path=self._fasttext_model_path)
            self._fasttext_model.initialize()
        return self._fasttext_model
    
    def _initialize_bge_large_model(self):
        """初始化BGE-Large模型（如果尚未初始化）"""
        if self._bge_large_model is None:
            from semantic_retrieval.embedding.bge import BGEEmbedding
            print("初始化BGE-Large模型...")
            self._bge_large_model = BGEEmbedding(model_name="bge-large-zh-v1.5")
            self._bge_large_model.initialize(
                server_url=self._xinference_server_url, 
                model_uid="bge-large-zh-v1.5"
            )
        return self._bge_large_model
    
    def _initialize_bge_m3_model(self):
        """初始化BGE-M3模型（如果尚未初始化）"""
        if self._bge_m3_model is None:
            from semantic_retrieval.embedding.bge import BGEEmbedding
            print("初始化BGE-M3模型...")
            self._bge_m3_model = BGEEmbedding(model_name="bge-m3")
            self._bge_m3_model.initialize(
                server_url=self._xinference_server_url, 
                model_uid="bge-m3"
            )
        return self._bge_m3_model
    
    def run_word2vec_matching(self, sentence1: str, sentence2: str) -> float:
        """
        使用Word2Vec模型执行语义匹配。
        
        使用FastText模型计算两个句子的嵌入向量，然后计算余弦相似度。
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            
        Returns:
            相似度分数 (0-1)
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 获取或初始化FastText模型
        model = self._initialize_fasttext_model()
        
        # 获取句子嵌入向量
        embedding1 = np.array(model.embed_query(sentence1)).reshape(1, -1)
        embedding2 = np.array(model.embed_query(sentence2)).reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # 确保相似度在0-1范围内
        return float(max(0, min(1, similarity)))
    
    def run_bge_large_matching(self, sentence1: str, sentence2: str) -> float:
        """
        使用BGE-Large模型执行语义匹配。
        
        使用BGE-Large模型计算两个句子的嵌入向量，然后计算余弦相似度。
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            
        Returns:
            相似度分数 (0-1)
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 获取或初始化BGE-Large模型
        model = self._initialize_bge_large_model()
        
        # 获取句子嵌入向量
        embedding1 = np.array(model.embed_query(sentence1)).reshape(1, -1)
        embedding2 = np.array(model.embed_query(sentence2)).reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # 确保相似度在0-1范围内
        return float(max(0, min(1, similarity)))
    
    def run_bge_m3_matching(self, sentence1: str, sentence2: str) -> float:
        """
        使用BGE-M3模型执行语义匹配。
        
        使用BGE-M3模型计算两个句子的嵌入向量，然后计算余弦相似度。
        
        Args:
            sentence1: 第一个句子
            sentence2: 第二个句子
            
        Returns:
            相似度分数 (0-1)
        """
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 获取或初始化BGE-M3模型
        model = self._initialize_bge_m3_model()
        
        # 获取句子嵌入向量
        embedding1 = np.array(model.embed_query(sentence1)).reshape(1, -1)
        embedding2 = np.array(model.embed_query(sentence2)).reshape(1, -1)
        
        # 计算余弦相似度
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # 确保相似度在0-1范围内
        return float(max(0, min(1, similarity)))
    
    def process_row(self, row: pd.Series) -> Dict[str, float]:
        """
        处理单行数据，执行所有模型的语义匹配。
        
        Args:
            row: 数据行，包含sentence1和sentence2
            
        Returns:
            包含各模型结果的字典
        """
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        
        # 执行各模型的语义匹配
        word2vec_score = self.run_word2vec_matching(sentence1, sentence2)
        bge_large_score = self.run_bge_large_matching(sentence1, sentence2)
        bge_m3_score = self.run_bge_m3_matching(sentence1, sentence2)
        
        # 返回结果字典
        return {
            'word2vec_match_res': word2vec_score,
            'bge_large_match_res': bge_large_score,
            'bge_m3_match_res': bge_m3_score
        }
    
    def process_batch(self, rows: List[Tuple[int, pd.Series]]) -> Dict[int, Dict[str, float]]:
        """
        并行处理一批数据行。
        
        Args:
            rows: 包含(全局索引, 数据行)的元组列表
            
        Returns:
            全局索引到结果字典的映射
        """
        # 确保所有模型都已初始化，避免在并行任务中重复初始化
        self._initialize_fasttext_model()
        self._initialize_bge_large_model()
        self._initialize_bge_m3_model()
        
        # 使用线程池而不是进程池，避免pickling问题
        results = {}
        
        # 定义单行处理函数
        def process_single_row(global_idx, row):
            result = self.process_row(row)
            return global_idx, result
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_single_row, global_idx, row): global_idx 
                for global_idx, row in rows
            }
            
            # 收集结果
            for future in future_to_idx:
                try:
                    global_idx, result = future.result()
                    results[global_idx] = result
                except Exception as e:
                    print(f"处理行时出错: {e}")
        
        return results
    
    def process_chunk(self, chunk_idx: int, chunk: pd.DataFrame) -> Dict[str, Any]:
        """
        处理数据块，对每行执行语义匹配。
        
        Args:
            chunk_idx: 数据块索引
            chunk: 数据块DataFrame
            
        Returns:
            包含处理进度信息的字典，包括：
            - results: 索引到结果的映射字典
            - processed_in_chunk: 本次处理的行数
            - chunk_completion: 当前数据块的完成百分比
            - dataset_completion: 整个数据集的完成百分比
        """
        chunk_results = {}
        processed_in_chunk = 0
        
        # 获取数据块的全局索引范围
        global_indices = set(int(row[self.data_manager.global_index_column]) for _, row in chunk.iterrows())
        total_in_chunk = len(global_indices)
        already_processed_in_chunk = len(global_indices.intersection(self.processed_indices))
        
        # 收集未处理的行
        unprocessed_rows = []
        for _, row in chunk.iterrows():
            # 获取全局索引
            global_index = int(row[self.data_manager.global_index_column])
            
            # 如果已经处理过，跳过
            if global_index in self.processed_indices:
                continue
                
            # 添加到未处理行列表
            unprocessed_rows.append((global_index, row))
            
            # 当积累了足够的未处理行，进行批量处理
            if len(unprocessed_rows) >= self.batch_size:
                # 批量处理行
                batch_results = self.process_batch(unprocessed_rows)
                
                # 更新结果和计数
                chunk_results.update(batch_results)
                processed_in_chunk += len(batch_results)
                
                # 更新已处理索引集合和结果缓存
                for idx, result in batch_results.items():
                    self.processed_indices.add(idx)
                    self.results_cache[idx] = result
                
                # 清空未处理行列表
                unprocessed_rows = []
                
                # 定期保存结果
                if len(self.results_cache) >= self.checkpoint_interval:
                    self.save_checkpoint(chunk_idx)
                    self.results_cache.clear()
        
        # 处理剩余的未处理行
        if unprocessed_rows:
            batch_results = self.process_batch(unprocessed_rows)
            
            # 更新结果和计数
            chunk_results.update(batch_results)
            processed_in_chunk += len(batch_results)
            
            # 更新已处理索引集合和结果缓存
            for idx, result in batch_results.items():
                self.processed_indices.add(idx)
                self.results_cache[idx] = result
        
        # 计算完成百分比
        processed_in_chunk_total = processed_in_chunk + already_processed_in_chunk
        chunk_completion = (processed_in_chunk_total / total_in_chunk * 100) if total_in_chunk > 0 else 100.0
        dataset_completion = (len(self.processed_indices) / self.data_manager.total_rows * 100)
        
        return {
            "results": chunk_results,
            "processed_in_chunk": processed_in_chunk,
            "chunk_completion": chunk_completion,
            "dataset_completion": dataset_completion,
        }
    
    def save_checkpoint(self, chunk_idx: int) -> None:
        """
        保存当前结果缓存到文件，并清空缓存
        
        Args:
            chunk_idx: 当前处理的数据块索引
        """
        if not self.results_cache:
            return
        
        # 计算当前chunk的起始和结束索引
        chunk_start_idx = chunk_idx * self.data_manager.chunksize
        chunk_end_idx = min((chunk_idx + 1) * self.data_manager.chunksize, self.data_manager.total_rows)
        
        # 获取结果缓存中的最大和最小索引
        min_result_index = min(self.results_cache.keys()) if self.results_cache else -1
        max_result_index = max(self.results_cache.keys()) if self.results_cache else -1
        
        # 计算当前chunk的完成百分比
        chunk_size = chunk_end_idx - chunk_start_idx
        processed_in_chunk = sum(1 for idx in self.processed_indices if chunk_start_idx <= idx < chunk_end_idx)
        chunk_completion = (processed_in_chunk / chunk_size * 100) if chunk_size > 0 else 100.0
        
        print(f"保存检查点，chunk_idx={chunk_idx}，包含 {len(self.results_cache)} 条结果...")
        print(f"当前chunk范围: {chunk_start_idx}-{chunk_end_idx-1}，处理进度: {chunk_completion:.2f}%")
        print(f"结果索引范围: {min_result_index}-{max_result_index}")
        
        # 保存结果
        success = self.data_manager.save_results(chunk_idx, self.results_cache)
        if not success:
            print("警告: 保存结果失败")
        else:
            print(f"检查点保存成功")
        
        # 清空缓存
        self.results_cache.clear()
        
        # 显示整体进度
        dataset_completion = len(self.processed_indices) / self.data_manager.total_rows * 100
        print(f"已处理 {len(self.processed_indices)} 行，占总数的 {dataset_completion:.2f}%")
    
    def run(self, max_rows: Optional[int] = None) -> None:
        """
        运行语义匹配实验。
        
        Args:
            max_rows: 最大处理行数，用于测试，None表示处理所有行
        """
        print(f"开始运行语义匹配实验...")
        start_time = time.time()
        
        # 获取未处理的数据块
        processed_count = 0
        chunk_count = 0
        
        # 如果是第一次运行，获取所有数据块
        if not self.processed_indices:
            for chunk_idx, (_, chunk) in enumerate(self.data_manager.get_chunks()):
                chunk_count += 1
                print(f"\n处理数据块 {chunk_idx}，包含 {len(chunk)} 行数据...")
                
                # 处理数据块
                chunk_results = self.process_chunk(chunk_idx, chunk)
                
                # 更新处理计数
                processed_count += chunk_results["processed_in_chunk"]
                
                # 显示处理进度
                print(f"数据块 {chunk_idx} 处理完成: {chunk_results['chunk_completion']:.2f}% 完成")
                print(f"整体进度: {chunk_results['dataset_completion']:.2f}% ({len(self.processed_indices)}/{self.data_manager.total_rows})")
                
                # 如果达到最大行数，停止处理
                if max_rows is not None and processed_count >= max_rows:
                    print(f"已达到指定的最大处理行数 {max_rows}，停止处理")
                    break
        else:
            # 如果是断点续传，只获取未处理的数据块
            unprocessed_chunks = self.data_manager.get_unprocessed_chunks(self.processed_indices)
            print(f"断点续传: 已处理 {len(self.processed_indices)} 行，还有 {len(unprocessed_chunks)} 个数据块未处理完全")
            
            for chunk_idx, (_, chunk) in enumerate(unprocessed_chunks):
                chunk_count += 1
                print(f"\n处理未完成的数据块 {chunk_idx}/{len(unprocessed_chunks)}，包含 {len(chunk)} 行数据...")
                
                # 处理数据块
                chunk_results = self.process_chunk(chunk_idx, chunk)
                
                # 更新处理计数
                processed_count += chunk_results["processed_in_chunk"]
                
                # 显示处理进度
                print(f"数据块 {chunk_idx} 处理完成: {chunk_results['chunk_completion']:.2f}% 完成")
                print(f"整体进度: {chunk_results['dataset_completion']:.2f}% ({len(self.processed_indices)}/{self.data_manager.total_rows})")
                
                if chunk_results["processed_in_chunk"] > 0:
                    print(f"本次处理了 {chunk_results['processed_in_chunk']} 行数据")
                
                # 如果达到最大行数，停止处理
                if max_rows is not None and processed_count >= max_rows:
                    print(f"已达到指定的最大处理行数 {max_rows}，停止处理")
                    break
        
        # 保存最后的结果缓存
        if self.results_cache:
            print("\n保存最后的结果缓存...")
            self.save_checkpoint(chunk_idx)
        
        # 清理资源
        self.cleanup()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n语义匹配实验完成!")
        print(f"总处理行数: {len(self.processed_indices)}")
        print(f"总耗时: {duration:.2f} 秒")
        if len(self.processed_indices) > 0:
            print(f"平均每行处理时间: {duration / len(self.processed_indices):.4f} 秒")
        print(f"结果保存在: {self.data_manager.get_output_path()}")
    
    def cleanup(self):
        """清理资源，释放模型内存"""
        print("清理资源...")
        # 释放模型资源
        self._fasttext_model = None
        self._bge_large_model = None
        self._bge_m3_model = None

def main():
    """主函数，解析命令行参数并运行实验"""
    parser = argparse.ArgumentParser(description="语义匹配实验运行器")
    parser.add_argument("--data-path", type=str, default="data/dataset1.csv", help="数据文件路径")
    parser.add_argument("--chunksize", type=int, default=8192, help="数据块大小")
    parser.add_argument("--experiment-id", type=str, help="实验ID")
    parser.add_argument("--checkpoint-interval", type=int, default=1024, help="检查点间隔行数")
    parser.add_argument("--max-rows", type=int, default=None, help="最大处理行数，用于测试")
    parser.add_argument("--batch-size", type=int, default=64, help="并行处理的批量大小")
    parser.add_argument("--n-jobs", type=int, default=8, help="并行处理的作业数，-1表示使用所有可用CPU")
    
    args = parser.parse_args()
    
    # 初始化运行器
    runner = SemanticMatchingRunner(
        data_path=args.data_path,
        chunksize=args.chunksize,
        experiment_id=args.experiment_id,
        checkpoint_interval=args.checkpoint_interval,
        batch_size=args.batch_size,
        n_jobs=args.n_jobs
    )
    
    # 运行实验
    runner.run(max_rows=args.max_rows)

if __name__ == "__main__":
    main()
