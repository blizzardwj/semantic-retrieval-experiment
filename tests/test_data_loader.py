import pytest
import pandas as pd
import os
from semantic_retrieval.data.data_loader import DataLoader

"""
CSV file schema:  ['id', 'sentence1', 'sentence2', 'label']
"""
class TestDataLoader:
    @pytest.fixture
    def test_data(self):
        """Fixture to create test data"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'sentence1': ["深度学习模型需要大量训练数据", "苹果是一种水果", "自然语言处理是AI的分支", "地球是太阳系中的行星", "这本书很有趣"],
            'sentence2': ["机器学习算法依赖数据质量", "汽车需要汽油才能行驶", "计算机视觉研究图像识别", "月球是地球的卫星", "那部电影很无聊"],
            'label': [1, 0, 0, 1, 0]
        })
    
    @pytest.fixture
    def data_file(self, tmp_path, test_data):
        """Fixture to create a temporary test data file
        
        Args:
            tmp_path: Built-in pytest fixture that provides a temporary directory path
            test_data: Test data fixture
            
        Returns:
            Path to the temporary test data file
        """
        file_path = tmp_path / "test_dataset.csv"
        test_data.to_csv(file_path, index=False)
        return file_path
        
    @pytest.fixture
    def data_loader(self, data_file):
        """Fixture to create a DataLoader instance with dynamic file"""
        return DataLoader(str(data_file))
    
    def test_load_data(self, data_loader, test_data):
        """Test dataset loading functionality"""
        df = data_loader.load_data()
        assert df is not None
        assert len(df) == len(test_data)
        assert list(df.columns) == ['id', 'sentence1', 'sentence2', 'label']
    
    @pytest.mark.parametrize("index,excepted_sentence", [
        (0, ("深度学习模型需要大量训练数据", "机器学习算法依赖数据质量")),
        (1, ("苹果是一种水果", "汽车需要汽油才能行驶"))
    ])
    def test_get_sentence_pairs(self, data_loader, index, excepted_sentence):
        """Test getting sentence pairs"""
        pairs = data_loader.get_sentence_pairs()
        assert len(pairs) == 5
        assert pairs[index] == excepted_sentence
    
    @pytest.mark.parametrize("index,expected_sentence", [
        (0, "深度学习模型需要大量训练数据"),
        (1, "苹果是一种水果"),
    ])
    def test_get_sentence1(self, data_loader, index, expected_sentence):
        """Test getting specific sentence1 by index"""
        sentences = data_loader.get_sentence1_list()
        assert sentences[index] == expected_sentence
        
    @pytest.mark.parametrize("index,expected_sentence", [
        (0, "机器学习算法依赖数据质量"),
        (1, "汽车需要汽油才能行驶"),
    ])
    def test_get_sentence2(self, data_loader, index, expected_sentence):
        """Test getting specific sentence2 by index"""
        sentences = data_loader.get_sentence2_list()
        assert sentences[index] == expected_sentence
        
    def test_get_labels(self, data_loader):
        """Test getting labels"""
        labels = data_loader.get_labels()
        assert len(labels) == 5
        assert labels == [1, 0, 0, 1, 0]
        
    def test_invalid_file_path(self):
        """Test behavior with invalid file path"""
        loader = DataLoader("invalid/path/to/file.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()
            
    def test_empty_file(self, tmp_path):
        """Test behavior with empty file"""
        empty_file = tmp_path / "empty.csv"
        empty_file.touch()  # Create empty file
        loader = DataLoader(str(empty_file))
        with pytest.raises(Exception):  # Adjust based on expected behavior
            loader.load_data()
            
    def test_malformed_data(self, tmp_path):
        """Test behavior with malformed data"""
        malformed_file = tmp_path / "malformed.csv"
        with open(malformed_file, 'w') as f:
            f.write("id,sentence1\n1,test\n")  # Missing required columns
        
        loader = DataLoader(str(malformed_file))
        with pytest.raises(Exception):  # Adjust based on expected behavior
            # This should fail because the file doesn't have all required columns
            pairs = loader.get_sentence_pairs()
