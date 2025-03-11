"""
Implementation for BGE embedding models.
"""
from semantic_retrieval.embedding.base import EmbeddingModel

class BGEEmbedding(EmbeddingModel):
    """Implementation for BGE embedding models."""
    
    def __init__(self, model_name):
        """Initialize with model name.
        
        Args:
            model_name (str): Name of the BGE model to use (e.g., "bge_large_zh-v1.5" or "bge-m3")
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def initialize(self):
        """Initialize and load BGE model."""
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
    
    def embed(self, text):
        """Embed a single text using BGE."""
        import torch
        
        if self.model is None or self.tokenizer is None:
            self.initialize()
            
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as sentence embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]
    
    def embed_batch(self, texts, batch_size=32):
        """Embed a batch of texts using BGE."""
        import numpy as np
        import torch
        from tqdm import tqdm
        
        if self.model is None or self.tokenizer is None:
            self.initialize()
            
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use CLS token embedding as sentence embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)
