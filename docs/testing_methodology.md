# 语义检索测试方法文档

本文档详细说明了语义检索实验中使用的测试方法、数据格式以及评估指标。

## 1. 测试数据格式

### 1.1 基本数据结构

测试数据采用CSV格式，包含以下字段：

```
['id', 'sentence1', 'sentence2', 'label']
```

其中：
- `id`: 数据条目的唯一标识符
- `sentence1`: 查询句子
- `sentence2`: 候选句子
- `label`: 标签，表示sentence1和sentence2是否语义相关（1表示相关，0表示不相关）

### 1.2 示例数据

```python
test_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'sentence1': ["深度学习模型需要大量训练数据", "苹果是一种水果", "自然语言处理是AI的分支", "地球是太阳系中的行星", "这本书很有趣"],
    'sentence2': ["机器学习算法依赖数据质量", "汽车需要汽油才能行驶", "计算机视觉研究图像识别", "月球是地球的卫星", "那部电影很无聊"],
    'label': [1, 0, 0, 1, 0]
})
```

### 1.3 数据加载

使用`DataLoader`类加载测试数据，该类提供以下方法：
- `load_data()`: 加载CSV数据文件
- `get_sentence_pairs()`: 获取所有(sentence1, sentence2)对
- `get_sentence1_list()`: 获取所有sentence1列表
- `get_sentence2_list()`: 获取所有sentence2列表
- `get_labels()`: 获取所有标签列表

## 2. 语义匹配测试方法

语义匹配测试主要评估不同模型计算句子向量并进行相似度计算的能力。

### 2.1 Word2Vec语义匹配测试

#### 测试步骤：
1. 使用Word2Vec模型为sentence1和sentence2生成向量表示
2. 计算每对sentence1和sentence2向量之间的余弦相似度
3. 设定相似度阈值（例如0.5），将高于阈值的对标记为相关（1），低于阈值的对标记为不相关（0）
4. 将预测结果与ground truth标签进行比较，计算准确率、精确率、召回率和F1分数

#### 示例代码：
```python
def test_word2vec_semantic_matching():
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化Word2Vec模型
    word2vec_approach = Word2VecApproach()
    
    # 计算相似度
    similarities = []
    for s1, s2 in zip(sentence1_list, sentence2_list):
        s1_vector = word2vec_approach.get_sentence_vector(s1)
        s2_vector = word2vec_approach.get_sentence_vector(s2)
        similarity = word2vec_approach.calculate_similarity(s1_vector, s2_vector)
        similarities.append(similarity)
    
    # 根据阈值生成预测标签
    threshold = 0.5
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    
    # 评估结果
    accuracy = sum(p == gt for p, gt in zip(predictions, ground_truth)) / len(ground_truth)
    
    return {
        "accuracy": accuracy,
        "similarities": similarities,
        "predictions": predictions
    }
```

### 2.2 BGE模型语义匹配测试

#### 测试步骤：
1. 使用两种BGE模型（bge-large-zh-v1.5和bge-m3）分别为sentence1和sentence2生成向量表示
2. 计算每对sentence1和sentence2向量之间的余弦相似度
3. 设定相似度阈值，将高于阈值的对标记为相关（1），低于阈值的对标记为不相关（0）
4. 将预测结果与ground truth标签进行比较，计算评估指标

#### 示例代码：
```python
def test_bge_semantic_matching(model_name="bge-large-zh-v1.5"):
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化BGE模型
    server_url = "http://20.30.80.200:9997"
    bge_approach = BGEApproach(model_name=model_name, server_url=server_url)
    
    # 计算相似度
    similarities = []
    for s1, s2 in zip(sentence1_list, sentence2_list):
        s1_vector = bge_approach.get_sentence_vector(s1)
        s2_vector = bge_approach.get_sentence_vector(s2)
        similarity = bge_approach.calculate_similarity(s1_vector, s2_vector)
        similarities.append(similarity)
    
    # 根据阈值生成预测标签
    threshold = 0.5
    predictions = [1 if sim >= threshold else 0 for sim in similarities]
    
    # 评估结果
    accuracy = sum(p == gt for p, gt in zip(predictions, ground_truth)) / len(ground_truth)
    
    return {
        "model": model_name,
        "accuracy": accuracy,
        "similarities": similarities,
        "predictions": predictions
    }
```

## 3. 语义检索测试方法

语义检索测试评估不同模型在检索相关句子方面的性能。

### 3.1 Word2Vec语义检索测试

#### 测试步骤：
1. 使用Word2Vec模型为所有sentence2建立向量库
2. 对每个sentence1查询，检索向量库中最相似的top-k个结果
3. 评估top-1、top-3和top-5结果中是否包含对应的相关sentence2
4. 如果包含相关sentence2，则标记为1，否则标记为0

#### 示例代码：
```python
def test_word2vec_retrieval():
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化Word2Vec检索器
    retriever = Word2VecApproach(use_vector_store=True)
    retriever.index_sentences(sentence2_list)
    
    # 评估结果
    top_1_hits = []
    top_3_hits = []
    top_5_hits = []
    
    for i, query in enumerate(sentence1_list):
        # 检索top-5结果
        results, similarities, indices = retriever.retrieve(query, top_k=5)
        
        # 检查是否命中相关句子
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        
        for k, idx in enumerate(indices):
            if ground_truth[i] == 1 and idx == i:  # 如果ground truth为1且检索到对应的sentence2
                if k < 1:
                    hit_1 = 1
                if k < 3:
                    hit_3 = 1
                if k < 5:
                    hit_5 = 1
        
        top_1_hits.append(hit_1)
        top_3_hits.append(hit_3)
        top_5_hits.append(hit_5)
    
    # 计算平均命中率
    top_1_accuracy = sum(top_1_hits) / len(top_1_hits)
    top_3_accuracy = sum(top_3_hits) / len(top_3_hits)
    top_5_accuracy = sum(top_5_hits) / len(top_5_hits)
    
    return {
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "top_1_hits": top_1_hits,
        "top_3_hits": top_3_hits,
        "top_5_hits": top_5_hits
    }
```

### 3.2 BGE模型语义检索测试

#### 测试步骤：
1. 使用两种BGE模型（bge-large-zh-v1.5和bge-m3）分别为所有sentence2建立向量库
2. 对每个sentence1查询，检索向量库中最相似的top-k个结果
3. 评估top-1、top-3和top-5结果中是否包含对应的相关sentence2
4. 如果包含相关sentence2，则标记为1，否则标记为0

#### 示例代码：
```python
def test_bge_retrieval(model_name="bge-large-zh-v1.5"):
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化BGE检索器
    server_url = "http://20.30.80.200:9997"
    retriever = BGEApproach(
        model_name=model_name,
        server_url=server_url,
        use_vector_store=True
    )
    retriever.index_sentences(sentence2_list)
    
    # 评估结果
    top_1_hits = []
    top_3_hits = []
    top_5_hits = []
    
    for i, query in enumerate(sentence1_list):
        # 检索top-5结果
        results, similarities, indices = retriever.retrieve(query, top_k=5)
        
        # 检查是否命中相关句子
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        
        for k, idx in enumerate(indices):
            if ground_truth[i] == 1 and idx == i:  # 如果ground truth为1且检索到对应的sentence2
                if k < 1:
                    hit_1 = 1
                if k < 3:
                    hit_3 = 1
                if k < 5:
                    hit_5 = 1
        
        top_1_hits.append(hit_1)
        top_3_hits.append(hit_3)
        top_5_hits.append(hit_5)
    
    # 计算平均命中率
    top_1_accuracy = sum(top_1_hits) / len(top_1_hits)
    top_3_accuracy = sum(top_3_hits) / len(top_3_hits)
    top_5_accuracy = sum(top_5_hits) / len(top_5_hits)
    
    return {
        "model": model_name,
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "top_1_hits": top_1_hits,
        "top_3_hits": top_3_hits,
        "top_5_hits": top_5_hits
    }
```

### 3.3 LLM重排序语义检索测试

#### 测试步骤：
1. 使用BGE模型为所有sentence2建立向量库
2. 对每个sentence1查询，先检索向量库中最相似的top-10个结果
3. 使用LLM（deepseek-r1-distill-qwen）对检索结果进行重排序，获取top-5结果
4. 评估重排序后的top-1、top-3和top-5结果中是否包含对应的相关sentence2
5. 如果包含相关sentence2，则标记为1，否则标记为0

#### 示例代码：
```python
def test_llm_reranked_retrieval():
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化LLM重排序检索器
    server_url = "http://20.30.80.200:9997"
    retriever = LLMRerankedBGEApproach(
        embedding_model="bge-large-zh-v1.5",
        embedding_server_url=server_url,
        llm_model="deepseek-r1-distill-qwen",
        llm_server_url=server_url,
        use_vector_store=True
    )
    retriever.index_sentences(sentence2_list)
    
    # 评估结果
    top_1_hits = []
    top_3_hits = []
    top_5_hits = []
    
    for i, query in enumerate(sentence1_list):
        # 检索top-5结果（从初始top-10中重排序）
        results, similarities, indices = retriever.retrieve(query, top_k=5, initial_top_k=10)
        
        # 检查是否命中相关句子
        hit_1 = 0
        hit_3 = 0
        hit_5 = 0
        
        for k, idx in enumerate(indices):
            if ground_truth[i] == 1 and idx == i:  # 如果ground truth为1且检索到对应的sentence2
                if k < 1:
                    hit_1 = 1
                if k < 3:
                    hit_3 = 1
                if k < 5:
                    hit_5 = 1
        
        top_1_hits.append(hit_1)
        top_3_hits.append(hit_3)
        top_5_hits.append(hit_5)
    
    # 计算平均命中率
    top_1_accuracy = sum(top_1_hits) / len(top_1_hits)
    top_3_accuracy = sum(top_3_hits) / len(top_3_hits)
    top_5_accuracy = sum(top_5_hits) / len(top_5_hits)
    
    return {
        "top_1_accuracy": top_1_accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "top_1_hits": top_1_hits,
        "top_3_hits": top_3_hits,
        "top_5_hits": top_5_hits
    }
```

## 4. 综合评估方法

使用`Evaluator`类对不同方法的检索结果进行综合评估：

### 4.1 评估指标

- **准确率（Accuracy）**：检索结果中相关句子的比例
- **精确率（Precision）**：检索结果中相关句子的比例
- **召回率（Recall）**：成功检索到的相关句子占所有相关句子的比例
- **特异性（Specificity）**：检索结果中不相关句子的比例
- **F1分数**：精确率和召回率的调和平均值
- **AUC**：曲线下面积，是模型区分正负样本能力的度量
- **平均倒数排名（MRR）**：相关句子在检索结果中排名的倒数的平均值
- **归一化折损累积增益（NDCG）**：考虑排序位置的评估指标

### 4.2 方法比较

使用`Evaluator.compare_approaches`方法比较不同检索方法的性能：

```python
def compare_all_approaches():
    # 加载测试数据
    data_loader = DataLoader("test_dataset.csv")
    sentence1_list = data_loader.get_sentence1_list()
    sentence2_list = data_loader.get_sentence2_list()
    ground_truth = data_loader.get_labels()
    
    # 初始化评估器
    evaluator = Evaluator()
    
    # 评估不同方法
    results = {}
    
    # 1. Word2Vec检索
    word2vec_retriever = Word2VecApproach(use_vector_store=True)
    results["word2vec"] = evaluator.evaluate_approach(
        approach_name="word2vec",
        retriever=word2vec_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 2. BGE-large检索
    bge_large_retriever = BGEApproach(
        model_name="bge-large-zh-v1.5",
        server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["bge_large"] = evaluator.evaluate_approach(
        approach_name="bge_large",
        retriever=bge_large_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 3. BGE-M3检索
    bge_m3_retriever = BGEApproach(
        model_name="bge-m3",
        server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["bge_m3"] = evaluator.evaluate_approach(
        approach_name="bge_m3",
        retriever=bge_m3_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5
    )
    
    # 4. LLM重排序检索
    llm_reranked_retriever = LLMRerankedBGEApproach(
        embedding_model="bge-large-zh-v1.5",
        embedding_server_url="http://20.30.80.200:9997",
        llm_model="deepseek-r1-distill-qwen",
        llm_server_url="http://20.30.80.200:9997",
        use_vector_store=True
    )
    results["llm_reranked"] = evaluator.evaluate_approach(
        approach_name="llm_reranked",
        retriever=llm_reranked_retriever,
        query_sentences=sentence1_list,
        sentence2_list=sentence2_list,
        ground_truth_labels=ground_truth,
        top_k=5,
        initial_top_k=10
    )
    
    # 比较不同方法
    comparison = evaluator.compare_approaches(results)
    
    return comparison
```

## 5. 测试执行流程

1. 准备测试数据集，确保包含必要的字段
2. 执行语义匹配测试，评估不同模型的匹配性能
3. 执行语义检索测试，评估不同模型的检索性能
4. 使用评估器进行综合评估，比较不同方法的性能
5. 可视化评估结果，生成比较图表

## 6. 注意事项

1. 确保Xinference模型推理服务可用，并且配置了正确的URL（http://20.30.80.200:9997）
2. 对于LLM重排序方法，需要确保LLM模型（deepseek-r1-distill-qwen）在Xinference模型推理服务上可用
3. 向量存储使用持久化目录，确保有足够的磁盘空间
4. 对于大规模测试，可能需要调整批处理大小和超时设置
