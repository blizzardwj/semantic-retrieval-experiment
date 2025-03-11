"""
Data loader for loading and preprocessing the CSV dataset.
"""

class DataLoader:
    """Responsible for loading and preprocessing the CSV dataset."""
    
    def __init__(self, file_path):
        """Initialize with the path to the CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing the dataset
        """
        self.file_path = file_path
        self.data = None
        
    def load_data(self):
        """Load data from CSV file."""
        import pandas as pd
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def get_sentence_pairs(self):
        """Get all sentence1 and sentence2 pairs."""
        if self.data is None:
            self.load_data()
        return list(zip(self.data['sentence1'], self.data['sentence2']))
    
    def get_sentence1_list(self):
        """Get list of all sentence1 entries."""
        if self.data is None:
            self.load_data()
        return self.data['sentence1'].tolist()
    
    def get_sentence2_list(self):
        """Get list of all sentence2 entries."""
        if self.data is None:
            self.load_data()
        return self.data['sentence2'].tolist()
    
    def get_labels(self):
        """Get ground truth labels."""
        if self.data is None:
            self.load_data()
        return self.data['label'].tolist()

