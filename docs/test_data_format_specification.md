# 语义检索测试数据格式规范

本文档详细说明了语义检索实验中使用的测试数据格式规范。

## 1. 基本数据格式

### 1.1 CSV文件格式

测试数据应以CSV格式存储，包含以下必要字段：

| 字段名 | 类型 | 说明 |
|-------|------|------|
| id | 整数 | 数据条目的唯一标识符 |
| sentence1 | 字符串 | 查询句子 |
| sentence2 | 字符串 | 候选句子 |
| label | 整数 | 标签，表示sentence1和sentence2是否语义相关（1表示相关，0表示不相关） |

### 1.2 文件编码

- 文件编码应使用UTF-8
- 文件应包含标题行
- 字段分隔符为逗号(,)

### 1.3 示例数据

```csv
id,sentence1,sentence2,label
1,深度学习模型需要大量训练数据,机器学习算法依赖数据质量,1
2,苹果是一种水果,汽车需要汽油才能行驶,0
3,自然语言处理是AI的分支,计算机视觉研究图像识别,0
4,地球是太阳系中的行星,月球是地球的卫星,1
5,这本书很有趣,那部电影很无聊,0
```

## 2. 数据要求

### 2.1 数据量要求

- **最小数据量**：测试数据集应至少包含50对句子
- **推荐数据量**：为获得可靠的评估结果，建议使用200-500对句子
- **平衡性**：正例（label=1）和负例（label=0）应尽量平衡，比例接近1:1

### 2.2 句子要求

- **句子长度**：每个句子应包含5-50个词或字符
- **语言**：支持中文、英文或混合语言
- **内容多样性**：句子应覆盖多个领域和主题，以测试模型的泛化能力
- **质量**：句子应语法正确，没有明显的拼写错误

### 2.3 标签要求

- **二元标签**：目前仅支持二元标签（0或1）
- **标注准则**：
  - 标签1：两个句子在语义上相关或表达相似的概念
  - 标签0：两个句子在语义上不相关或表达不同的概念

## 3. 高级数据格式

### 3.1 多级相关性标签（可选）

对于更精细的评估，可以使用多级相关性标签：

| 标签值 | 相关性级别 |
|-------|----------|
| 0 | 完全不相关 |
| 1 | 轻微相关 |
| 2 | 相关 |
| 3 | 高度相关 |

使用多级标签时，需要修改评估脚本以适应非二元标签。

### 3.2 分组查询（可选）

对于评估不同类型的查询性能，可以添加查询分组信息：

```csv
id,sentence1,sentence2,label,group
1,深度学习模型需要大量训练数据,机器学习算法依赖数据质量,1,tech
2,苹果是一种水果,汽车需要汽油才能行驶,0,general
3,自然语言处理是AI的分支,计算机视觉研究图像识别,0,tech
4,地球是太阳系中的行星,月球是地球的卫星,1,science
5,这本书很有趣,那部电影很无聊,0,entertainment
```

## 4. 数据加载与处理

### 4.1 DataLoader类

项目中的`DataLoader`类用于加载和处理测试数据：

```python
from semantic_retrieval.data.data_loader import DataLoader

# 初始化数据加载器
data_loader = DataLoader("path/to/test_dataset.csv")

# 加载数据
df = data_loader.load_data()

# 获取句子对
sentence_pairs = data_loader.get_sentence_pairs()

# 获取sentence1列表
sentence1_list = data_loader.get_sentence1_list()

# 获取sentence2列表
sentence2_list = data_loader.get_sentence2_list()

# 获取标签列表
labels = data_loader.get_labels()
```

### 4.2 数据验证

在使用数据前，应进行以下验证：

1. 检查数据是否包含所有必要字段
2. 验证标签值是否有效
3. 检查是否有缺失值
4. 确认句子长度是否合理

```python
def validate_dataset(file_path):
    """验证数据集格式是否符合要求"""
    try:
        # 加载数据
        df = pd.read_csv(file_path)
        
        # 检查必要字段
        required_columns = ['id', 'sentence1', 'sentence2', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误：缺少必要字段 {missing_columns}")
            return False
        
        # 检查标签值
        invalid_labels = df[~df['label'].isin([0, 1])]['label'].unique()
        if len(invalid_labels) > 0:
            print(f"错误：发现无效的标签值 {invalid_labels}")
            return False
        
        # 检查缺失值
        missing_values = df[required_columns].isnull().sum()
        if missing_values.sum() > 0:
            print(f"错误：发现缺失值\n{missing_values}")
            return False
        
        # 检查句子长度
        short_sentences = df[(df['sentence1'].str.len() < 5) | (df['sentence2'].str.len() < 5)]
        if len(short_sentences) > 0:
            print(f"警告：发现{len(short_sentences)}个过短的句子")
        
        long_sentences = df[(df['sentence1'].str.len() > 200) | (df['sentence2'].str.len() > 200)]
        if len(long_sentences) > 0:
            print(f"警告：发现{len(long_sentences)}个过长的句子")
        
        # 检查标签分布
        label_counts = df['label'].value_counts()
        print(f"标签分布：\n{label_counts}")
        
        # 计算正负例比例
        pos_ratio = label_counts.get(1, 0) / len(df)
        print(f"正例比例：{pos_ratio:.2f}")
        
        if pos_ratio < 0.3 or pos_ratio > 0.7:
            print("警告：正负例分布不平衡")
        
        print("数据集验证通过！")
        return True
    
    except Exception as e:
        print(f"验证过程中出错：{e}")
        return False
```

## 5. 数据集示例

### 5.1 小型测试数据集

用于单元测试和快速验证：

```python
test_data = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'sentence1': ["深度学习模型需要大量训练数据", "苹果是一种水果", "自然语言处理是AI的分支", "地球是太阳系中的行星", "这本书很有趣"],
    'sentence2': ["机器学习算法依赖数据质量", "汽车需要汽油才能行驶", "计算机视觉研究图像识别", "月球是地球的卫星", "那部电影很无聊"],
    'label': [1, 0, 0, 1, 0]
})
```

### 5.2 中型评估数据集

用于模型性能评估：

```python
# 生成中型评估数据集示例
def generate_medium_dataset(output_path, size=100):
    """生成中型评估数据集"""
    import random
    import pandas as pd
    
    # 定义领域和对应的句子模板
    domains = {
        "技术": [
            "人工智能正在改变世界各个领域",
            "深度学习模型需要大量训练数据",
            "机器学习算法依赖数据质量和特征工程",
            "神经网络是深度学习的核心组件",
            "自然语言处理让机器能够理解人类语言",
            "计算机视觉技术可以识别图像中的物体",
            "强化学习是机器学习中的一种方法",
            "大语言模型已经取得了长足的进步",
            "知识图谱可以表示实体之间的关系",
            "语义检索可以帮助我们找到相关的信息"
        ],
        "科学": [
            "地球是太阳系中的第三颗行星",
            "月球是地球的唯一天然卫星",
            "太阳是太阳系的中心天体",
            "光速是宇宙中的最高速度",
            "量子力学描述了微观世界的规律",
            "相对论改变了我们对时空的理解",
            "DNA是遗传信息的载体",
            "元素周期表包含了所有已知元素",
            "黑洞是引力极强的天体",
            "宇宙大爆炸理论解释了宇宙的起源"
        ],
        "日常": [
            "这本书很有趣",
            "那部电影很无聊",
            "今天天气真好",
            "我喜欢吃水果",
            "运动对健康有益",
            "音乐可以改善心情",
            "睡眠对身体恢复很重要",
            "阅读可以增长知识",
            "旅行可以开阔视野",
            "朋友之间应该互相帮助"
        ]
    }
    
    # 生成数据
    data = []
    id_counter = 1
    
    while len(data) < size:
        # 随机选择领域
        domain = random.choice(list(domains.keys()))
        sentences = domains[domain]
        
        # 随机选择两个句子
        s1 = random.choice(sentences)
        
        # 决定是生成正例还是负例
        if random.random() < 0.5:  # 生成正例
            # 从同一领域选择另一个句子
            s2 = random.choice([s for s in sentences if s != s1])
            label = 1
        else:  # 生成负例
            # 从其他领域选择句子
            other_domain = random.choice([d for d in domains.keys() if d != domain])
            s2 = random.choice(domains[other_domain])
            label = 0
        
        # 添加到数据集
        data.append({
            'id': id_counter,
            'sentence1': s1,
            'sentence2': s2,
            'label': label
        })
        
        id_counter += 1
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"已生成{size}条数据，保存至{output_path}")
    
    return df
```

## 6. 数据集分割

对于大型数据集，建议进行训练/验证/测试分割：

```python
def split_dataset(file_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """将数据集分割为训练集、验证集和测试集"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 第一次分割：分离出测试集
    train_val, test = train_test_split(
        df, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=df['label']  # 确保标签分布一致
    )
    
    # 第二次分割：分离训练集和验证集
    train, val = train_test_split(
        train_val,
        test_size=val_ratio/(train_ratio+val_ratio),
        random_state=random_state,
        stratify=train_val['label']
    )
    
    # 保存分割后的数据集
    base_name = file_path.rsplit('.', 1)[0]
    train.to_csv(f"{base_name}_train.csv", index=False)
    val.to_csv(f"{base_name}_val.csv", index=False)
    test.to_csv(f"{base_name}_test.csv", index=False)
    
    print(f"数据集分割完成：")
    print(f"训练集：{len(train)}条")
    print(f"验证集：{len(val)}条")
    print(f"测试集：{len(test)}条")
    
    return train, val, test
```

## 7. 数据增强（可选）

对于小型数据集，可以考虑使用数据增强技术：

```python
def augment_dataset(file_path, output_path, augmentation_factor=2):
    """对数据集进行简单的数据增强"""
    import pandas as pd
    import random
    import jieba
    import nltk
    from nltk.corpus import wordnet
    
    # 加载数据
    df = pd.read_csv(file_path)
    original_size = len(df)
    
    # 定义简单的同义词替换函数
    def replace_with_synonym(sentence, language='zh'):
        if language == 'zh':
            # 中文分词
            words = list(jieba.cut(sentence))
            # 随机选择一个词进行替换
            if len(words) > 3:
                idx = random.randint(0, len(words)-1)
                # 这里应该使用中文同义词库，简化起见，我们只做小改动
                words[idx] = words[idx] + "的"
                return ''.join(words)
            return sentence
        else:
            # 英文同义词替换
            words = nltk.word_tokenize(sentence)
            if len(words) > 3:
                idx = random.randint(0, len(words)-1)
                synonyms = []
                for syn in wordnet.synsets(words[idx]):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                if synonyms:
                    words[idx] = random.choice(synonyms)
                return ' '.join(words)
            return sentence
    
    # 增强数据
    augmented_data = []
    id_counter = df['id'].max() + 1
    
    for _ in range(augmentation_factor - 1):
        for _, row in df.iterrows():
            # 只对正例进行增强
            if row['label'] == 1:
                # 随机决定增强哪个句子
                if random.random() < 0.5:
                    s1 = replace_with_synonym(row['sentence1'])
                    s2 = row['sentence2']
                else:
                    s1 = row['sentence1']
                    s2 = replace_with_synonym(row['sentence2'])
                
                augmented_data.append({
                    'id': id_counter,
                    'sentence1': s1,
                    'sentence2': s2,
                    'label': 1
                })
                
                id_counter += 1
    
    # 合并原始数据和增强数据
    augmented_df = pd.DataFrame(augmented_data)
    final_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # 保存增强后的数据集
    final_df.to_csv(output_path, index=False)
    
    print(f"数据增强完成：")
    print(f"原始数据：{original_size}条")
    print(f"增强数据：{len(augmented_data)}条")
    print(f"最终数据：{len(final_df)}条")
    
    return final_df
```

## 8. 数据集版本控制

为确保测试结果的可重复性，建议对测试数据集进行版本控制：

1. 为每个数据集分配唯一的版本号
2. 记录数据集的基本统计信息（大小、标签分布等）
3. 在测试结果中引用数据集版本号
4. 保存数据集的SHA256哈希值，用于验证完整性

```python
def generate_dataset_metadata(file_path):
    """生成数据集元数据"""
    import pandas as pd
    import hashlib
    import json
    from datetime import datetime
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 计算SHA256哈希值
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    
    # 生成元数据
    metadata = {
        "filename": file_path,
        "version": datetime.now().strftime("%Y%m%d"),
        "created_at": datetime.now().isoformat(),
        "size": len(df),
        "columns": list(df.columns),
        "label_distribution": df['label'].value_counts().to_dict(),
        "positive_ratio": float(df['label'].mean()),
        "sha256": file_hash
    }
    
    # 保存元数据
    metadata_path = file_path.rsplit('.', 1)[0] + '_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"元数据已保存至{metadata_path}")
    
    return metadata
```
