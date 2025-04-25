"""
语义匹配模型实现

该模块包含了基于ModelRunner抽象类的具体模型实现，用于执行语义匹配实验。
包括：
- FastTextModelRunner: 基于FastText的语义匹配模型
- BGEModelRunner: 基于BGE的语义匹配模型
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity

from src.runner_worker.model_runner import ModelRunner

class FastTextModelRunner(ModelRunner):
    """基于FastText的语义匹配模型实现"""
    
    def __init__(self, model_path: str, input_columns: Optional[List[str]] = None, device: str = 'cpu'):
        """
        初始化FastText模型运行器
        
        Args:
            model_path: FastText模型文件路径
            input_columns: 输入列名，默认为['sentence1', 'sentence2']
            device: 计算设备，默认为'cpu'
        """
        super().__init__(model_path, input_columns or ['sentence1', 'sentence2'], device)
    
    def _load_model(self):
        """
        加载FastText语义匹配模型
        
        Returns:
            加载的FastText模型对象
        """
        from semantic_retrieval.embedding.fasttext import FastTextEmbedding
        model = FastTextEmbedding(model_path=self.model_path)
        model.initialize()
        return model
    
    def preprocess(self, row_data: Dict[str, Any]):
        """
        对输入行数据进行预处理
        
        Args:
            row_data: 包含句子对的数据行
            
        Returns:
            预处理后的句子对
        """
        if isinstance(row_data, pd.Series):
            sentence1 = str(row_data[self.input_columns[0]])
            sentence2 = str(row_data[self.input_columns[1]])
        elif isinstance(row_data, dict):
            sentence1 = str(row_data[self.input_columns[0]])
            sentence2 = str(row_data[self.input_columns[1]])
        elif isinstance(row_data, (list, tuple)) and len(row_data) >= 2:
            sentence1 = str(row_data[0])
            sentence2 = str(row_data[1])
        else:
            raise ValueError(f"不支持的数据类型: {type(row_data)}")
        
        return sentence1, sentence2
    
    def predict(self, row_data: Dict[str, Any]):
        """
        使用FastText模型执行语义匹配预测
        
        Args:
            row_data: 包含句子对的数据行
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            if not self.is_initialized:
                self.initialize()
            
            sentence1, sentence2 = self.preprocess(row_data)
            
            # 获取句子嵌入向量
            embedding1 = np.array(self.model.embed_query(sentence1)).reshape(1, -1)
            embedding2 = np.array(self.model.embed_query(sentence2)).reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # 确保相似度在0-1范围内
            return float(max(0, min(1, similarity)))
        except Exception as e:
            print(f"FastText语义匹配预测错误: {e}")
            return 0.0
    
    def batch_predict(self, rows_data:Union[pd.DataFrame, List[Dict[str, Any]]]):
        """
        对多行数据批量运行预测
        
        Args:
            rows_data: 包含多行句子对的数据
            
        Returns:
            行索引到相似度分数的映射
        """
        results = {}
        
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 如果是DataFrame
            if isinstance(rows_data, pd.DataFrame):
                for idx, row in rows_data.iterrows():
                    results[idx] = self.predict(row)
            # 如果是字典列表
            elif isinstance(rows_data, list) and all(isinstance(item, dict) for item in rows_data):
                for i, row in enumerate(rows_data):
                    results[i] = self.predict(row)
            # 如果是(索引,行)的元组列表
            elif isinstance(rows_data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in rows_data):
                for idx, row in rows_data:
                    results[idx] = self.predict(row)
            else:
                raise ValueError(f"不支持的批量数据类型: {type(rows_data)}")
        except Exception as e:
            print(f"FastText批量预测错误: {e}")
        
        return results


class BGEModelRunner(ModelRunner):
    """基于BGE的语义匹配模型实现"""
    
    def __init__(self, model_name: str, server_url: str, model_uid: Optional[str] = None, 
                 input_columns: Optional[List[str]] = None, device: str = 'cpu'):
        """
        初始化BGE模型运行器
        
        Args:
            model_name: BGE模型名称，如'bge-large-zh-v1.5'或'bge-m3'
            server_url: Xinference服务器URL
            model_uid: 模型UID，如果不提供则使用model_name
            input_columns: 输入列名，默认为['sentence1', 'sentence2']
            device: 计算设备，默认为'cpu'
        """
        self.model_name = model_name
        self.server_url = server_url
        self.model_uid = model_uid or model_name
        
        # 使用server_url作为model_path
        super().__init__(model_path=server_url, input_columns=input_columns or ['sentence1', 'sentence2'], device=device)
    
    def _load_model(self):
        """
        加载BGE语义匹配模型
        
        Returns:
            加载的BGE模型对象
        """
        from semantic_retrieval.embedding.bge import BGEEmbedding
        model = BGEEmbedding(model_name=self.model_name)
        model.initialize(server_url=self.server_url, model_uid=self.model_uid)
        return model
    
    def preprocess(self, row_data: Dict[str, Any]):
        """
        对输入行数据进行预处理
        
        Args:
            row_data: 包含句子对的数据行
            
        Returns:
            预处理后的句子对
        """
        if isinstance(row_data, pd.Series):
            sentence1 = str(row_data[self.input_columns[0]])
            sentence2 = str(row_data[self.input_columns[1]])
        elif isinstance(row_data, dict):
            sentence1 = str(row_data[self.input_columns[0]])
            sentence2 = str(row_data[self.input_columns[1]])
        elif isinstance(row_data, (list, tuple)) and len(row_data) >= 2:
            sentence1 = str(row_data[0])
            sentence2 = str(row_data[1])
        else:
            raise ValueError(f"不支持的数据类型: {type(row_data)}")
        
        return sentence1, sentence2
    
    def predict(self, row_data: Dict[str, Any]):
        """
        使用BGE模型执行语义匹配预测
        
        Args:
            row_data: 包含句子对的数据行
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            if not self.is_initialized:
                self.initialize()
            
            sentence1, sentence2 = self.preprocess(row_data)
            
            # 获取句子嵌入向量
            embedding1 = np.array(self.model.embed_query(sentence1)).reshape(1, -1)
            embedding2 = np.array(self.model.embed_query(sentence2)).reshape(1, -1)
            
            # 计算余弦相似度
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            
            # 确保相似度在0-1范围内
            return float(max(0, min(1, similarity)))
        except Exception as e:
            print(f"BGE语义匹配预测错误: {e}")
            return 0.0
    
    def batch_predict(self, rows_data: Union[pd.DataFrame, List[Dict[str, Any]]]):
        """
        对多行数据批量运行预测
        
        Args:
            rows_data: 包含多行句子对的数据
            
        Returns:
            行索引到相似度分数的映射
        """
        results = {}
        
        if not self.is_initialized:
            self.initialize()
        
        try:
            # 如果是DataFrame
            if isinstance(rows_data, pd.DataFrame):
                for idx, row in rows_data.iterrows():
                    results[idx] = self.predict(row)
            # 如果是字典列表
            elif isinstance(rows_data, list) and all(isinstance(item, dict) for item in rows_data):
                for i, row in enumerate(rows_data):
                    results[i] = self.predict(row)
            # 如果是(索引,行)的元组列表
            elif isinstance(rows_data, list) and all(isinstance(item, tuple) and len(item) == 2 for item in rows_data):
                for idx, row in rows_data:
                    results[idx] = self.predict(row)
            else:
                raise ValueError(f"不支持的批量数据类型: {type(rows_data)}")
        except Exception as e:
            print(f"BGE批量预测错误: {e}")
        
        return results


# 可以使用BGEModelRunner创建具体的BGE变体
class BGELargeModelRunner(BGEModelRunner):
    """BGE-Large模型运行器"""
    
    def __init__(self, server_url: str, input_columns: Optional[List[str]] = None, device: str = 'cpu'):
        """
        初始化BGE-Large模型运行器
        
        Args:
            server_url: Xinference服务器URL
            input_columns: 输入列名，默认为['sentence1', 'sentence2']
            device: 计算设备，默认为'cpu'
        """
        super().__init__(
            model_name="bge-large-zh-v1.5",
            server_url=server_url,
            model_uid="bge-large-zh-v1.5",
            input_columns=input_columns,
            device=device
        )


class BGEM3ModelRunner(BGEModelRunner):
    """BGE-M3模型运行器"""
    
    def __init__(self, server_url: str, input_columns: Optional[List[str]] = None, device: str = 'cpu'):
        """
        初始化BGE-M3模型运行器
        
        Args:
            server_url: XInference服务器URL
            input_columns: 输入列名，默认为['sentence1', 'sentence2']
            device: 计算设备，默认为'cpu'
        """
        super().__init__(
            model_name="bge-m3",
            server_url=server_url,
            model_uid="bge-m3",
            input_columns=input_columns,
            device=device
        )
