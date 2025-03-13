import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock

# 设置模拟模块，但在每个测试中重新创建模拟对象
sys.modules['fasttext'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_community.embeddings'] = MagicMock()
sys.modules['langchain_community.embeddings.XinferenceEmbeddings'] = MagicMock()

# 导入嵌入模型
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.embedding.fasttext import FastTextEmbedding

"""
embedding model config:
    server_url="http://20.30.80.200:9997",
    model_name="bge-large-zh-v1.5"
    model_uid="your-model-uid-here"

fasttext model path: /home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin
"""

class TestBGEEmbedding:
    def setup_method(self):
        """在每个测试前重置模拟对象"""
        self.mock_xinference = MagicMock()
        sys.modules['langchain_community.embeddings.XinferenceEmbeddings'] = self.mock_xinference
    
    def test_bge_init(self):
        """Test BGE embedding initialization"""
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        assert model is not None
        assert model.model_name == "bge-large-zh-v1.5"
        assert model.embedding_model is None
    
    def test_bge_initialize(self):
        """Test BGE embedding initialization with server connection"""
        # 创建模拟对象
        mock_embedding = MagicMock()
        self.mock_xinference.XinferenceEmbeddings.return_value = mock_embedding
        
        # 测试初始化
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        model.initialize(server_url="http://20.30.80.200:9997", model_uid="test-model-uid")
        
        # 验证
        assert model.server_url == "http://20.30.80.200:9997"
        assert model.model_uid == "test-model-uid"
        assert model.embedding_model is not None
    
    def test_bge_embed(self):
        """Test BGE embedding of a single text"""
        # 创建模拟对象
        mock_embedding = MagicMock()
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # 初始化模型
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        model.embedding_model = mock_embedding
        
        # 测试嵌入
        embedding = model.embed("测试文本")
        
        # 验证
        mock_embedding.embed_query.assert_called_once_with("测试文本")
        assert isinstance(embedding, np.ndarray)
        assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    def test_bge_embed_batch(self):
        """Test BGE embedding of a batch of texts"""
        # 创建模拟对象
        mock_embedding = MagicMock()
        mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # 初始化模型
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        model.embedding_model = mock_embedding
        
        # 测试批量嵌入
        texts = ["测试文本1", "测试文本2"]
        embeddings = model.embed_batch(texts)
        
        # 验证
        mock_embedding.embed_documents.assert_called_once_with(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)
        assert embeddings.tolist() == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    def test_bge_not_initialized(self):
        """Test error when BGE model is not initialized"""
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        
        with pytest.raises(ValueError, match="Model not initialized"):
            model.embed("测试文本")
        
        with pytest.raises(ValueError, match="Model not initialized"):
            model.embed_batch(["测试文本1", "测试文本2"])


class TestFastTextEmbedding:
    def setup_method(self):
        """在每个测试前重置模拟对象"""
        self.mock_fasttext = MagicMock()
        sys.modules['fasttext'] = self.mock_fasttext
    
    def test_fasttext_init(self):
        """Test FastText embedding initialization"""
        model = FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        assert model is not None
        assert model.model_path == "/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin"
        assert model.model is None
    
    def test_fasttext_initialize(self):
        """Test FastText model initialization"""
        # 创建模拟对象
        mock_model = MagicMock()
        self.mock_fasttext.load_model.return_value = mock_model
        
        # 测试初始化
        model = FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        model.initialize()
        
        # 验证
        self.mock_fasttext.load_model.assert_called_once_with("/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        assert model.model == mock_model
    
    def test_fasttext_initialize_default(self):
        """Test FastText model initialization with default path"""
        # 创建模拟对象
        mock_model = MagicMock()
        self.mock_fasttext.load_model.return_value = mock_model
        
        # 测试默认路径初始化
        model = FastTextEmbedding()
        model.initialize()
        
        # 验证
        self.mock_fasttext.load_model.assert_called_once_with('cc.zh.300.bin')
        assert model.model == mock_model
    
    def test_fasttext_embed(self):
        """Test FastText embedding of a single text"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_model.get_sentence_vector.return_value = np.array([0.1, 0.2, 0.3])
        
        # 初始化和测试
        model = FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        model.model = mock_model
        embedding = model.embed("测试文本")
        
        # 验证
        mock_model.get_sentence_vector.assert_called_once_with("测试文本")
        assert isinstance(embedding, np.ndarray)
        assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    def test_fasttext_embed_auto_init(self):
        """Test FastText auto-initialization when embedding without explicit init"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_model.get_sentence_vector.return_value = np.array([0.1, 0.2, 0.3])
        self.mock_fasttext.load_model.return_value = mock_model
        
        # 测试自动初始化
        model = FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        embedding = model.embed("测试文本")
        
        # 验证模型自动加载
        self.mock_fasttext.load_model.assert_called_once_with("/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        mock_model.get_sentence_vector.assert_called_once_with("测试文本")
        assert isinstance(embedding, np.ndarray)
    
    def test_fasttext_embed_batch(self):
        """Test FastText embedding of a batch of texts"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_model.get_sentence_vector.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # 初始化和测试
        model = FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        model.model = mock_model
        texts = ["测试文本1", "测试文本2"]
        embeddings = model.embed_batch(texts)
        
        # 验证
        assert mock_model.get_sentence_vector.call_count == 2
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)
        assert embeddings.tolist() == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
