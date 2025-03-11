import pytest
from semantic_retrieval.data.data_loader import DataLoader

class TestDataLoader:
    @pytest.fixture
    def data_loader(self):
        """Fixture to create a DataLoader instance"""
        return DataLoader("tests/test_data/test_dataset.csv")
    
    def test_load_data(self, data_loader):
        """Test dataset loading functionality"""
        df = data_loader.load_data()
        assert df is not None
        assert len(df) == 5
        assert list(df.columns) == ['id', 'sentence1', 'sentence2', 'label']
        
    def test_get_sentence_pairs(self, data_loader):
        """Test getting sentence pairs"""
        pairs = data_loader.get_sentence_pairs()
        assert len(pairs) == 5
        assert pairs[0] == ("深度学习模型需要大量训练数据", "机器学习算法依赖数据质量")
        assert pairs[1] == ("苹果是一种水果", "汽车需要汽油才能行驶")
        
    def test_get_sentence1_list(self, data_loader):
        """Test getting sentence1 list"""
        sentences = data_loader.get_sentence1_list()
        assert len(sentences) == 5
        assert sentences[0] == "深度学习模型需要大量训练数据"
        assert sentences[1] == "苹果是一种水果"
        
    def test_get_sentence2_list(self, data_loader):
        """Test getting sentence2 list"""
        sentences = data_loader.get_sentence2_list()
        assert len(sentences) == 5
        assert sentences[0] == "机器学习算法依赖数据质量"
        assert sentences[1] == "汽车需要汽油才能行驶"
        
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
