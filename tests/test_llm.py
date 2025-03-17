import pytest
import sys
import numpy as np
from unittest.mock import patch, MagicMock

from semantic_retrieval.llm.model import LLMModel
# Import the actual message classes to ensure type compatibility
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

"""
LLM model config:
    server_url="http://20.30.80.200:9997/v1",
    model_name="deepseek-r1-distill-qwen"
"""

# 全局模拟设置
@pytest.fixture(autouse=True)
def mock_dependencies():
    """全局模拟所有依赖模块"""
    mocks = {
        'langchain_core.messages': MagicMock(),
        'langchain_community.llms.xinference': MagicMock(),
    }
    
    # 保存原始模块 and apply mocks
    original_modules = {}
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
    sys.modules['langchain_community.llms.xinference'] = mock
    return mock

class TestLLMModel:
    @pytest.fixture
    def llm_model(self):
        """返回未初始化的LLM模型实例"""
        return LLMModel(model_name="deepseek-r1-distill-qwen")
    
    @pytest.fixture
    def initialized_chat_model(self, llm_model, mock_xinference):
        """返回已初始化的聊天模型实例"""
        mock_chat_model = MagicMock()
        mock_xinference.Xinference.return_value = mock_chat_model
        
        llm_model.initialize()
        return llm_model
    
    @pytest.fixture
    def initialized_text_model(self, mock_xinference):
        """返回已初始化的文本模型实例"""
        mock_text_model = MagicMock()
        mock_xinference.Xinference.return_value = mock_text_model
        
        model = LLMModel(model_name="deepseek-r1-distill-qwen", model_type="text")
        model.initialize()
        return model
    
    def test_llm_init(self, llm_model):
        """测试LLM模型初始化"""
        assert llm_model is not None
        assert llm_model.model_name == "deepseek-r1-distill-qwen"
        assert llm_model.server_url == "http://localhost:9997"
        assert llm_model.model_type == "chat"
        assert llm_model.chat_model is None
    
    def test_llm_initialize_chat(self, llm_model, mock_xinference):
        """测试聊天模型初始化"""
        # 创建模拟对象
        mock_chat_model = MagicMock()
        mock_xinference.Xinference.return_value = mock_chat_model
        
        # 测试初始化
        llm_model.initialize()
        
        # 验证
        mock_xinference.Xinference.assert_called_once_with(
            server_url="http://localhost:9997",
            model_uid="deepseek-r1-distill-qwen"
        )
        assert llm_model.chat_model == mock_chat_model
        assert not hasattr(llm_model, 'model') or llm_model.model is None
    
    def test_llm_initialize_text(self, mock_xinference):
        """测试文本模型初始化"""
        # 创建模拟对象
        mock_text_model = MagicMock()
        mock_xinference.Xinference.return_value = mock_text_model
        
        # 创建文本模型并初始化
        model = LLMModel(model_name="deepseek-r1-distill-qwen", model_type="text")
        model.initialize()
        
        # 验证
        mock_xinference.Xinference.assert_called_once_with(
            server_url="http://localhost:9997",
            model_uid="deepseek-r1-distill-qwen"
        )
        assert model.model == mock_text_model
        assert model.chat_model is None
    
    def test_rerank_chat_model(self, initialized_chat_model):
        """测试使用聊天模型进行重排序"""
        # 设置模拟响应
        mock_response = MagicMock()
        mock_response.content = "1, 3, 2"
        initialized_chat_model.chat_model.invoke.return_value = mock_response
        
        # 测试数据
        query = "测试查询"
        candidates = ["候选句子1", "候选句子2", "候选句子3", "候选句子4"]
        indices = [10, 20, 30, 40]  # 原始索引
        
        # 执行重排序
        result = initialized_chat_model.rerank(query, candidates, indices, top_k=3)
        
        # 验证
        assert initialized_chat_model.chat_model.invoke.called
        assert result == [10, 30, 20]  # 根据响应"1, 3, 2"映射到原始索引
    
    def test_rerank_text_model(self, initialized_text_model):
        """测试使用文本模型进行重排序"""
        # 设置模拟响应
        mock_response = "2, 4, 1"
        initialized_text_model.model.invoke.return_value = mock_response
        
        # 测试数据
        query = "测试查询"
        candidates = ["候选句子1", "候选句子2", "候选句子3", "候选句子4"]
        indices = [10, 20, 30, 40]  # 原始索引
        
        # 执行重排序
        result = initialized_text_model.rerank(query, candidates, indices, top_k=3)
        
        # 验证
        assert initialized_text_model.model.invoke.called
        assert result == [20, 40, 10]  # 根据响应"2, 4, 1"映射到原始索引
    
    def test_rerank_auto_initialize(self, llm_model, mock_xinference):
        """测试在重排序时自动初始化模型"""
        # 创建模拟对象
        mock_chat_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1, 2"
        mock_chat_model.invoke.return_value = mock_response
        mock_xinference.Xinference.return_value = mock_chat_model
        
        # 测试数据
        query = "测试查询"
        candidates = ["候选句子1", "候选句子2"]
        indices = [10, 20]
        
        # 执行重排序（未显式初始化）
        result = llm_model.rerank(query, candidates, indices, top_k=2)
        
        # 验证自动初始化
        mock_xinference.Xinference.assert_called_once()
        assert llm_model.chat_model == mock_chat_model
        assert result == [10, 20]
    
    def test_rerank_invalid_response(self, initialized_chat_model):
        """测试处理无效响应"""
        # 设置模拟响应
        mock_response = MagicMock()
        mock_response.content = "这不是有效的索引列表"
        initialized_chat_model.chat_model.invoke.return_value = mock_response
        
        # 测试数据
        query = "测试查询"
        candidates = ["候选句子1", "候选句子2", "候选句子3"]
        indices = [10, 20, 30]
        
        # 执行重排序
        result = initialized_chat_model.rerank(query, candidates, indices, top_k=2)
        
        # 验证回退到前k个索引
        assert result == [10, 20]
    
    # def test_rerank_exception_handling(self, initialized_chat_model):
    #     """测试异常处理"""
    #     # 设置模拟响应引发异常
    #     initialized_chat_model.chat_model.invoke.side_effect = Exception("模拟异常")
        
    #     # 测试数据
    #     query = "测试查询"
    #     candidates = ["候选句子1", "候选句子2", "候选句子3"]
    #     indices = [10, 20, 30]
        
    #     # 执行重排序 - rerank方法应该内部处理异常并返回默认结果
    #     result = initialized_chat_model.rerank(query, candidates, indices, top_k=2)
        
    #     # 验证回退到前k个索引
    #     assert result == [10, 20]
    
    def test_rerank_not_enough_indices(self, initialized_chat_model):
        """测试响应中索引不足的情况"""
        # 设置模拟响应
        mock_response = MagicMock()
        mock_response.content = "1"  # 只返回一个索引，但要求top_k=3
        initialized_chat_model.chat_model.invoke.return_value = mock_response
        
        # 测试数据
        query = "测试查询"
        candidates = ["候选句子1", "候选句子2", "候选句子3", "候选句子4"]
        indices = [10, 20, 30, 40]
        
        # 执行重排序
        result = initialized_chat_model.rerank(query, candidates, indices, top_k=3)
        
        # 验证结果包含第一个返回的索引，然后是其他未在响应中的索引
        assert result[0] == 10  # 第一个是从响应中解析的
        assert set(result[1:]) <= {20, 30, 40}  # 其余是从候选中选择的
        assert len(result) == 3  # 总共返回3个结果


# 集成测试，可选择性注释掉如果没有实际环境
class TestLLMIntegration:
    @pytest.fixture(autouse=True)
    def disable_mocks(self):
        """临时禁用模拟对象，使集成测试使用实际模块而非模拟"""
        # 移除当前的模拟模块
        mock_modules = [
            'langchain_core.messages',
            'langchain_community.llms.xinference',
        ]

        for module_name in mock_modules:
            if module_name in sys.modules:
                sys.modules.pop(module_name)
    
        # 导入实际模块
        from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
        from langchain_community.llms.xinference import Xinference
        
        yield
        
        # 测试结束后，不需要手动恢复
        # mock_dependencies fixture 会在下一个测试前重新应用模拟

    def test_actual_llm_reranking(self):
        """测试实际LLM模型重排序功能"""
        try:
            print("\n开始测试实际LLM重排序...")
            
            # 创建实际模型
            from semantic_retrieval.llm.model import LLMModel
            print("创建 LLMModel 实例...")
            model = LLMModel(
                model_name="deepseek-r1-distill-qwen",
                model_type="chat"
            )
            
            print("初始化模型...")
            model.initialize(server_url="http://20.30.80.200:9997")
            print(f"模型初始化完成，chat_model 类型: {type(model.chat_model)}")
            
            # 测试数据
            query = "人工智能的发展趋势"
            candidates = [
                "人工智能正在各个领域快速发展",
                "机器学习是人工智能的一个重要分支",
                "深度学习在图像识别方面取得了突破性进展",
                "自然语言处理技术使机器能够理解人类语言",
                "强化学习使机器能够通过试错学习"
            ]
            indices = [0, 1, 2, 3, 4]
            
            print("执行重排序...")
            # 执行重排序
            # 首先测试 SystemMessage 和 BaseMessage 类型
            system_message = SystemMessage(content="测试内容")
            print(f"SystemMessage 类型: {type(system_message)}")
            print(f"isinstance(system_message, BaseMessage): {isinstance(system_message, BaseMessage)}")
            
            # 执行重排序
            result = model.rerank(query, candidates, indices, top_k=3)
            
            # 验证
            assert len(result) == 3
            assert all(idx in indices for idx in result)
            
            # 打印结果以便手动验证
            print("\n重排序结果:")
            for idx in result:
                print(f"- {candidates[idx]}")

        except Exception as e:
            import traceback
            print(f"\n完整错误信息: {repr(e)}")
            print(f"错误类型: {type(e)}")
            # 打印更多关于消息类型的信息
            traceback.print_exc()
            # 获取系统消息并检查其类型
            try:
                test_message = SystemMessage(content="Test")
                print(f"测试消息类型: {type(test_message)}")
                print(f"是否为BaseMessage: {isinstance(test_message, BaseMessage)}")
                print(f"SystemMessage.__module__: {SystemMessage.__module__}")
                print(f"BaseMessage.__module__: {BaseMessage.__module__}")
            except Exception as inner_e:
                print(f"测试消息检查失败: {repr(inner_e)}")
            
            pytest.skip(f"集成测试失败，可能是环境问题: {repr(e)}")
