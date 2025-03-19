# 数据流动路径

## 1. 主要数据流向

### 原始数据流 (CSV → 处理结果)
```
ChunkedDataManager(读取) → ParallelProcessor → ModelRunner(处理) → 
ParallelProcessor(收集) → ChunkedDataManager(保存)
```

### 检查点流 (处理状态 → 持久化)
```
ParallelProcessor(跟踪进度) → RobustCheckpointManager(保存检查点) → 
RobustCheckpointManager(加载检查点) → ParallelProcessor(恢复进度)
```

### 控制流 (系统管理)
```
OptimizedSemanticMatchingSystem → ParallelProcessor → 其他组件
```

## 2. 核心协作过程

### 断点续处理协作
- ParallelProcessor启动时，从RobustCheckpointManager获取已处理的行索引
- 将这些索引传递给ChunkedDataManager，获取未处理的数据块
- 确保只处理新数据，避免重复工作

### 并行处理协作
- ParallelProcessor创建进程池，将数据块分配给多个工作进程
- 每个工作进程使用ModelRunner实例处理数据
- 处理结果返回给ParallelProcessor并合并

### 检查点保存协作
- 处理一定数量行后，ParallelProcessor调用RobustCheckpointManager保存检查点
- 同时可能调用ChunkedDataManager保存中间结果