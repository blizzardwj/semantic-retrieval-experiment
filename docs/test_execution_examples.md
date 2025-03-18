# 语义检索测试执行示例

本文档提供了如何执行语义检索测试的具体示例。

## 1. 环境准备

首先确保已安装所有必要的依赖：

```bash
# 检查已安装的包
PYTHONPATH=$PYTHONPATH:./src uv pip list

# 安装缺失的依赖
uv pip install -e .
```

## 2. 数据加载测试

测试数据加载功能是否正常工作：

```bash
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_data_loader.py -v
```

## 3. 语义匹配测试执行

### 3.1 Word2Vec语义匹配测试

```bash
# 运行Word2Vec语义匹配测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_matching.py::TestMatching::test_word2vec_matching -v
```

### 3.2 BGE模型语义匹配测试

```bash
# 运行BGE-large-zh模型语义匹配测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_matching.py::TestMatching::test_bge_large_matching -v

# 运行BGE-M3模型语义匹配测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_matching.py::TestMatching::test_bge_m3_matching -v
```

## 4. 语义检索测试执行

### 4.1 Word2Vec语义检索测试

```bash
# 运行Word2Vec语义检索测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_retrieval.py::TestRetrieval::test_word2vec_vector_store_retrieval -v
```

### 4.2 BGE模型语义检索测试

```bash
# 运行BGE-large-zh模型语义检索测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_retrieval.py::TestRetrieval::test_bge_large_vector_store_retrieval -v

# 运行BGE-M3模型语义检索测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_retrieval.py::TestRetrieval::test_bge_m3_vector_store_retrieval -v
```

### 4.3 LLM重排序语义检索测试

```bash
# 运行LLM重排序语义检索测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_retrieval.py::TestRetrieval::test_llm_reranked_vector_store_retrieval -v
```

## 5. 综合评估测试

```bash
# 运行评估器测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_evaluation.py -v
```

## 6. 端到端集成测试

```bash
# 运行端到端集成测试
PYTHONPATH=$PYTHONPATH:./src pytest tests/test_integration.py -v
```

## 7. 自定义测试脚本示例

以下是一个自定义测试脚本示例，可以用于比较不同方法的性能：

```python
# compare_methods.py
import pandas as pd
import matplotlib.pyplot as plt
from semantic_retrieval.data.data_loader import DataLoader
from semantic_retrieval.evaluation.evaluator import Evaluator
from semantic_retrieval.retrieval.word2vec import Word2VecApproach
from semantic_retrieval.retrieval.bge import BGEApproach
from semantic_retrieval.retrieval.llm_reranked import LLMRerankedBGEApproach

def compare_retrieval_methods(data_file):
    # 加载测试数据
    data_loader = DataLoader(data_file)
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 评估不同方法
    results = {}
    
    # 1. Word2Vec检索
    print("评估Word2Vec检索...")
    word2vec_retriever = Word2VecApproach(use_vector_store=True)
    results["Word2Vec"] = evaluator.evaluate_approach(
        approach_name="Word2Vec",
        retriever=word2vec_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 2. BGE-large检索
    print("评估BGE-large检索...")
    bge_large_retriever = BGEApproach(
        model_name="bge-large-zh-v1.5",
        server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["BGE-large"] = evaluator.evaluate_approach(
        approach_name="BGE-large",
        retriever=bge_large_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 3. BGE-M3检索
    print("评估BGE-M3检索...")
    bge_m3_retriever = BGEApproach(
        model_name="bge-m3",
        server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["BGE-M3"] = evaluator.evaluate_approach(
        approach_name="BGE-M3",
        retriever=bge_m3_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 4. LLM重排序检索
    print("评估LLM重排序检索...")
    llm_reranked_retriever = LLMRerankedBGEApproach(
        embedding_model="bge-large-zh-v1.5",
        embedding_server_url="http://20.30.80.200:9997",
        llm_model="deepseek-r1-distill-qwen",
        llm_server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["LLM重排序"] = evaluator.evaluate_approach(
        approach_name="LLM重排序",
        retriever=llm_reranked_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5,
        initial_top_k=10
    )
    
    # 比较不同方法
    comparison = evaluator.compare_approaches(results)
    
    # 可视化结果
    metrics = ["precision", "recall", "f1", "mrr", "ndcg"]
    df = pd.DataFrame(comparison)
    
    # 绘制柱状图
    ax = df.plot(kind="bar", figsize=(12, 8))
    plt.title("不同检索方法性能比较")
    plt.xlabel("评估指标")
    plt.ylabel("得分")
    plt.xticks(rotation=0)
    plt.legend(title="检索方法")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig("retrieval_comparison.png")
    print("结果已保存到 retrieval_comparison.png")
    
    return comparison

if __name__ == "__main__":
    # 使用测试数据集
    data_file = "data/test_dataset.csv"
    results = compare_retrieval_methods(data_file)
    print("\n最终比较结果:")
    print(pd.DataFrame(results))
```

运行自定义测试脚本：

```bash
PYTHONPATH=$PYTHONPATH:./src python scripts/compare_methods.py
```

## 8. 测试结果分析

测试完成后，可以分析以下几个方面的结果：

1. **语义匹配准确性**：比较不同模型在语义匹配任务上的准确率、精确率、召回率和F1分数
2. **检索性能**：比较不同模型在top-k检索中的命中率
3. **排序质量**：通过MRR和NDCG指标评估不同模型的排序质量
4. **计算效率**：记录不同模型的推理时间和资源消耗

## 9. 常见问题排查

1. **Xinference推理服务连接问题**：
   - 确认服务器URL是否正确（http://20.30.80.200:9997）
   - 检查网络连接是否正常
   - 验证服务器上是否已加载所需模型

2. **向量存储问题**：
   - 确保有足够的磁盘空间
   - 检查持久化目录的权限
   - 如果索引损坏，尝试删除并重新创建

3. **LLM重排序问题**：
   - 检查LLM模型响应格式
   - 确认模型是否正确解析查询
   - 查看LLM日志以排查潜在问题
