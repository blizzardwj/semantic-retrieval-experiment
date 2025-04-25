from abc import ABC, abstractmethod

class ModelRunner(ABC):
    """
    封装语义匹配模型，处理单行或批量数据。
    
    主要职责:
    - 加载和初始化语义匹配模型
    - 处理输入数据预处理和特征提取
    - 执行模型推理并返回匹配结果
    
    数据流向:
    - 输入: 从ParallelProcessor接收行数据
    - 输出: 向ParallelProcessor返回处理结果
    """
    
    def __init__(self, model_path, input_columns=None, device='cpu'):
        """
        初始化模型运行器。
        
        Args:
            model_path (str): 模型文件路径
            input_columns (list, optional): 作为模型输入的列名列表
            device (str): 运行设备，'cpu'或'cuda'
        """
        self.model_path = model_path
        self.input_columns = input_columns
        self.device = device
        self.model = None
        self.is_initialized = False
    
    @abstractmethod
    def _load_model(self):
        """
        加载语义匹配模型。
        
        Returns:
            model: 加载的模型对象
        """
        pass
    
    @abstractmethod
    def preprocess(self, row_data):
        """
        对输入行数据进行预处理。
        
        Args:
            row_data (dict): 行数据
            
        Returns:
            object: 预处理后的模型输入
        """
        pass
    
    @abstractmethod
    def predict(self, row_data):
        """
        对单行数据运行语义匹配预测。
        
        包含错误处理机制，确保模型故障不会中断整个处理流程。
        
        Args:
            row_data (dict): 行数据
            
        Returns:
            object: 模型预测结果
        """
        pass
    
    @abstractmethod
    def batch_predict(self, rows_data):
        """
        对多行数据批量运行预测，提高处理效率。
        
        Args:
            rows_data (list[dict]): 多行数据
            
        Returns:
            dict: 行索引到预测结果的映射
        """
        pass
    
    def initialize(self):
        """
        初始化模型资源。
        
        该方法应在模型使用前调用，用于加载模型和准备任何必要的资源。
        实现懒加载模式，避免在创建实例时就加载大型模型。
        
        Returns:
            self: 返回自身实例，支持方法链式调用
        """
        if not self.is_initialized:
            self.model = self._load_model()
            self.is_initialized = True
        return self
    
    def cleanup(self):
        """
        清理模型资源。
        
        该方法应在完成模型使用后调用，用于释放模型占用的内存和其他资源。
        
        Returns:
            bool: 清理是否成功
        """
        try:
            self.model = None
            self.is_initialized = False
            return True
        except Exception as e:
            print(f"清理模型资源时发生错误: {e}")
            return False
    
    def __enter__(self):
        """
        上下文管理器入口方法。
        
        自动调用initialize方法，并返回模型实例。
        
        Returns:
            self: 返回已初始化的模型实例
        """
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        上下文管理器退出方法。
        
        自动调用cleanup方法，释放模型资源。
        
        Args:
            exc_type: 异常类型(如果有)
            exc_val: 异常值(如果有)
            exc_tb: 异常堆栈跟踪(如果有)
            
        Returns:
            bool: 如果异常已处理则返回True，否则返回False
        """
        self.cleanup()
        return exc_type is None
