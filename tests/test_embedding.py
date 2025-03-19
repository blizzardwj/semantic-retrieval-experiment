import pytest
import numpy as np
import sys
from unittest.mock import patch, MagicMock
import os

# 导入嵌入模型
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.embedding.fasttext import FastTextEmbedding

"""
embedding model config:
    server_url="http://20.30.80.200:9997",
    model_name="bge-large-zh-v1.5"  # the same as model_uid

fasttext model path: /home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin
"""

# 全局模拟设置
@pytest.fixture(autouse=True)
def mock_dependencies():
    """全局模拟所有依赖模块"""
    # 保存原始模块
    original_modules = {}
    mocks = {
        'fasttext': MagicMock(),
        'langchain_community.embeddings': MagicMock(),
    }
    
    # 保存并应用模拟
    for module_name, mock in mocks.items():
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = mock
    
    yield
    
    # 测试后恢复原始模块
    for module_name in mocks:
        if module_name in original_modules:
            sys.modules[module_name] = original_modules[module_name]
        else:
            # 如果之前不存在，则从 sys.modules 中移除
            if module_name in sys.modules:
                del sys.modules[module_name]

@pytest.fixture
def mock_xinference():
    """创建并配置xinference模拟对象"""
    mock = MagicMock()
    sys.modules['langchain_community.embeddings.XinferenceEmbeddings'] = mock
    return mock

@pytest.fixture
def mock_fasttext():
    """创建并配置fasttext模拟对象"""
    mock = MagicMock()
    sys.modules['fasttext'] = mock
    return mock

class TestBGEEmbedding:
    @pytest.fixture
    def bge_model(self):
        """返回未初始化的BGE模型实例"""
        return BGEEmbedding(model_name="bge-large-zh-v1.5")
    
    @pytest.fixture
    def initialized_bge_model(self, bge_model, mock_xinference):
        """返回已初始化的BGE模型实例"""
        mock_embedding = MagicMock()
        mock_xinference.XinferenceEmbeddings.return_value = mock_embedding
        
        bge_model.initialize(server_url="http://20.30.80.200:9997", model_uid="bge-large-zh-v1.5")
        # 确保模型的embedding_model属性是我们的模拟对象
        bge_model.embedding_model = mock_embedding
        return bge_model
    
    def test_bge_init(self, bge_model):
        """Test BGE embedding initialization"""
        assert bge_model is not None
        assert bge_model.model_name == "bge-large-zh-v1.5"
        assert bge_model.embedding_model is None
    
    def test_bge_initialize(self, bge_model, mock_xinference):
        """Test BGE embedding initialization with server connection"""
        # 创建模拟对象
        mock_embedding = MagicMock()
        mock_xinference.XinferenceEmbeddings.return_value = mock_embedding
        
        # 测试初始化
        bge_model.initialize(server_url="http://20.30.80.200:9997", model_uid="test-model-uid")
        
        # 验证
        assert bge_model.server_url == "http://20.30.80.200:9997"
        assert bge_model.model_uid == "test-model-uid"
        assert bge_model.embedding_model is not None
    
    def test_bge_embed(self, initialized_bge_model):
        """Test BGE embedding of a single text"""
        # 配置模拟返回值
        mock_embedding = initialized_bge_model.embedding_model
        mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        
        # 测试嵌入
        embedding = initialized_bge_model.embed_query("测试文本")
        
        # 验证
        mock_embedding.embed_query.assert_called_once_with("测试文本")
        assert isinstance(embedding, list)
        assert embedding == [0.1, 0.2, 0.3]
    
    def test_bge_embed_batch(self, initialized_bge_model):
        """Test BGE embedding of a batch of texts"""
        # 配置模拟返回值
        mock_embedding = initialized_bge_model.embedding_model
        mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # 测试批量嵌入
        texts = ["测试文本1", "测试文本2"]
        embeddings = initialized_bge_model.embed_documents(texts)
        
        # 验证
        mock_embedding.embed_documents.assert_called_once_with(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]
    
    def test_bge_not_initialized(self, bge_model):
        """Test error when BGE model is not initialized"""
        with pytest.raises(ValueError, match="Model not initialized"):
            bge_model.embed_query("测试文本")
        
        with pytest.raises(ValueError, match="Model not initialized"):
            bge_model.embed_documents(["测试文本1", "测试文本2"])


class TestFastTextEmbedding:
    @pytest.fixture
    def fasttext_model(self):
        """返回未初始化的FastText模型实例"""
        return FastTextEmbedding(model_path="/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
    
    @pytest.fixture
    def initialized_fasttext_model(self, fasttext_model, mock_fasttext):
        """返回已初始化的FastText模型实例"""
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model
        
        fasttext_model.initialize()
        return fasttext_model
    
    def test_fasttext_init(self, fasttext_model):
        """Test FastText embedding initialization"""
        assert fasttext_model is not None
        assert fasttext_model.model_path == "/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin"
        assert fasttext_model.model is None
    
    def test_fasttext_initialize(self, fasttext_model, mock_fasttext):
        """Test FastText model initialization"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model
        
        # 测试初始化
        fasttext_model.initialize()
        
        # 验证
        mock_fasttext.load_model.assert_called_once_with("/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        assert fasttext_model.model == mock_model
    
    def test_fasttext_initialize_default(self, mock_fasttext):
        """Test FastText model initialization with default path"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_fasttext.load_model.return_value = mock_model
        
        # 测试默认路径初始化
        model = FastTextEmbedding()
        model.initialize()
        
        # 验证
        mock_fasttext.load_model.assert_called_once_with('cc.zh.300.bin')
        assert model.model == mock_model
    
    def test_fasttext_embed(self, initialized_fasttext_model, mock_fasttext):
        """Test FastText embedding of a single text"""
        # 配置模拟返回值
        mock_model = initialized_fasttext_model.model
        mock_model.get_sentence_vector.return_value = np.array([0.1, 0.2, 0.3])
        
        # 测试嵌入
        embedding = initialized_fasttext_model.embed("测试文本")
        
        # 验证
        mock_model.get_sentence_vector.assert_called_once_with("测试文本")
        assert isinstance(embedding, np.ndarray)
        assert embedding.tolist() == [0.1, 0.2, 0.3]
    
    def test_fasttext_embed_auto_init(self, fasttext_model, mock_fasttext):
        """Test FastText auto-initialization when embedding without explicit init"""
        # 创建模拟对象
        mock_model = MagicMock()
        mock_model.get_sentence_vector.return_value = np.array([0.1, 0.2, 0.3])
        mock_fasttext.load_model.return_value = mock_model
        
        # 测试自动初始化
        embedding = fasttext_model.embed_query("测试文本")
        
        # 验证模型自动加载
        mock_fasttext.load_model.assert_called_once_with("/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin")
        mock_model.get_sentence_vector.assert_called_once_with("测试文本")
        assert isinstance(embedding, list)
    
    def test_fasttext_embed_batch(self, initialized_fasttext_model):
        """Test FastText embedding of a batch of texts"""
        # 配置模拟返回值
        mock_model = initialized_fasttext_model.model
        mock_model.get_sentence_vector.side_effect = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6])
        ]
        
        # 测试批量嵌入
        texts = ["测试文本1", "测试文本2"]
        embeddings = initialized_fasttext_model.embed_batch(texts)
        
        # 验证
        assert mock_model.get_sentence_vector.call_count == 2
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 3)
        assert embeddings.tolist() == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

# 集成测试，可选择性注释掉如果没有实际环境
@pytest.mark.integration
@pytest.mark.skipif(False, reason="仅在有实际环境时运行")
class TestIntegration:
    @pytest.fixture(autouse=True)
    def disable_mocks(self):
        """临时禁用模拟对象，使集成测试使用实际模块而非模拟"""
        # 移除当前的模拟模块
        mock_modules = [
            'fasttext',
            'langchain_community.embeddings',
        ]
        for module_name in mock_modules:
            if module_name in sys.modules:
                sys.modules.pop(module_name)
    
        # 导入实际模块
        import fasttext
        from langchain_community.embeddings import XinferenceEmbeddings
        
        yield
        
        # 测试结束后，不需要手动恢复
        # mock_dependencies fixture 会在下一个测试前重新应用模拟
    
    def test_bge_actual_embedding(self):
        """测试实际BGE模型嵌入功能"""
        model = BGEEmbedding(model_name="bge-large-zh-v1.5")
        model.initialize(
            server_url="http://20.30.80.200:9997",
            model_uid="bge-large-zh-v1.5"
        )
        
        embedding = model.embed_query("测试文本")
        assert isinstance(embedding, list)
        assert len(embedding) > 0  # 确保嵌入维度正确
        
        # 批量测试
        embeddings = model.embed_documents(["测试文本1", "测试文本2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        
    def test_fasttext_actual_embedding(self):
        """测试实际FastText模型嵌入功能"""
        # 检查模型文件是否存在
        model_path = "/home/zfwj/workspace/new_code/fasttext_model/cc.zh.300.bin"
        if not os.path.exists(model_path):
            pytest.skip(f"FastText模型文件不存在: {model_path}")
        
        model = FastTextEmbedding(model_path=model_path)
        
        embedding = model.embed_query("测试文本")
        assert isinstance(embedding, list)
        assert len(embedding) == 300  # FastText通常是300维
        
        # 批量测试
        embeddings = model.embed_documents(["测试文本1", "测试文本2"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
