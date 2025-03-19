class RobustCheckpointManager:
    """
    管理实验检查点，支持可靠的断点续传。
    
    主要职责:
    - 追踪和记录已处理的数据行
    - 保存和加载检查点文件
    - 提供进度信息和中断恢复功能
    
    数据流向:
    - 输入: 从ParallelProcessor接收已处理的行索引和元数据
    - 输出: 向ParallelProcessor提供已处理的行索引，用于跳过重复处理
    """
    
    def __init__(self, checkpoint_dir="checkpoints", experiment_name="semantic_matching"):
        """
        初始化检查点管理器。
        
        Args:
            checkpoint_dir (str): 检查点文件目录
            experiment_name (str): 实验名称，用于检查点文件命名
        """
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        self.checkpoint_path = self._prepare_checkpoint_path()
        self.backup_path = f"{self.checkpoint_path}.backup"
    
    def _prepare_checkpoint_path(self):
        """
        准备检查点文件路径，确保目录存在。
        
        Returns:
            str: 检查点文件完整路径
        """
        pass
    
    def save_checkpoint(self, processed_indices, metadata=None):
        """
        保存当前处理进度到检查点文件。
        
        实现了双文件备份机制，防止写入过程中断导致检查点损坏。
        
        Args:
            processed_indices (set): 已处理行的索引集合
            metadata (dict, optional): 附加元数据信息，如时间戳、配置参数等
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    def load_checkpoint(self):
        """
        加载检查点，获取已处理的行索引。
        
        如果主检查点损坏，尝试从备份恢复。
        
        Returns:
            tuple: (已处理行索引集合, 检查点元数据)
        """
        pass
    
    def checkpoint_exists(self):
        """
        检查是否存在有效的检查点文件。
        
        Returns:
            bool: 是否存在可用检查点
        """
        pass
    
    def get_progress_info(self, total_rows):
        """
        获取当前处理进度信息。
        
        Args:
            total_rows (int): 总行数
            
        Returns:
            dict: 包含进度百分比、处理行数、剩余行数等信息
        """
        pass
