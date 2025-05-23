# 通过向量相似度检索，查找语料库中与query 最相似句子

## 实验目标

本实验旨在评估不同文本嵌入模型在向量空间中对句子对语义表示能力的表现。具体目标包括：
- 构建包含超过 10 万条句子对及其相似性标签的语料库；
- 使用指定嵌入模型将句子映射为向量，并构建对应的向量索引；
- 验证语义相似句对在向量空间中距离显著小于非相似句对的假设；
- 基于检索结果统计并比较不同模型的准确率、召回率等评价指标。

## 实验方法

方法1：基于向量相似度的检索实验
1. 准备语料库表格，其中包含两列句子 (A 列和 B 列) 及相似标签 (true/false)。
2. 使用指定嵌入模型对 A 列所有句子生成向量，构建向量索引库。
3. 遍历 B 列中每个句子作为 query：
   3.1 将 query 嵌入成向量；  
   3.2 在索引库中检索 top‑k 相似向量；  
   3.3 判断检索结果中是否存在 label 为 true 的句子；  
      - 若存在，则该条 query 检索成功；  
      - 若不存在且该对真实标签为 false，则判为正确未检出；  
      - 其它情况视为检索错误。
4. 统计所有 query 的检索结果，计算方法1的准确率、召回率等指标。
5. 实验可通过调整阈值、top‑k 参数评估模型稳定性。

## 实验方案优缺点

优点：
- 模式直观：直接模拟向量检索场景，接近实际应用。  
- 评价维度丰富：可灵活调节 top‑k 和相似度阈值，观察效果变化。

缺点：
- 计算开销大：大规模语料库下索引构建与多次检索耗时显著。  
- 单向检索：只检索 A→B，未对称评估，可能遗漏双向相似性信息。

## 实验方法2（不在此实现）

方法2：基于句子对的二分类模型

直接使用模型为句子对的两个句子生成嵌入向量，然后计算两句话的相似度，最后利用二分类模型测评的方法，根据ground truth 计算 F1 score，AUC值等指标。

## 实验方案2优缺点（略）