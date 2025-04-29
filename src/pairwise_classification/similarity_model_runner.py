# -*- coding: utf-8 -*-
"""ModelRunner implementation for calculating sentence similarity."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any, Dict, List
import pandas as pd

from src.runner_worker.model_runner import ModelRunner
# Assuming embedding models have an 'embed_query' method
# from semantic_retrieval.embedding.base import BaseEmbedding # Use the actual base class or protocol if available

from src.utils import build_logger

logger = build_logger(__name__)

class SimilarityModelRunner(ModelRunner):
    """
    A ModelRunner that calculates cosine similarity between two sentences 
    using a provided embedding model.
    """

    def __init__(self, embedding_model: Any, input_columns: List[str] = ['sentence1', 'sentence2'], **kwargs):
        """
        Initialize the runner.

        Args:
            embedding_model: An initialized embedding model instance with an 'embed_query' method.
            input_columns: List containing the names of the two columns holding the sentences.
        """
        if len(input_columns) != 2:
            raise ValueError("input_columns must contain exactly two column names for sentence1 and sentence2")
        
        # Store the model directly; no separate loading needed as it's passed in initialized.
        self.model = embedding_model
        self.is_initialized = True # Mark as initialized since model is passed in ready
        self.input_columns = input_columns
        # Pass other potential ModelRunner args like device to super if needed, though not used here.
        # super().__init__(model_path=None, input_columns=input_columns, **kwargs) 
        # We bypass super().__init__ slightly as we manage the model directly.
        self.model_path = getattr(embedding_model, 'model_path', 'N/A') # Store path if available
        self.device = getattr(embedding_model, 'device', 'N/A') # Store device if available

    def _load_model(self):
        """Model is already loaded and passed during init."""
        return self.model

    def preprocess(self, row_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Extracts the two sentences from the row data.
        """
        try:
            sentence1 = row_data[self.input_columns[0]]
            sentence2 = row_data[self.input_columns[1]]
            if not isinstance(sentence1, str) or not isinstance(sentence2, str):
                # Handle potential non-string data (e.g., NaN) gracefully
                # print(f"Warning: Non-string input detected in row. Columns: {self.input_columns}, Values: [{sentence1}, {sentence2}]")
                return "", "" # Return empty strings to avoid embedding errors
            return sentence1, sentence2
        except KeyError as e:
            raise KeyError(f"Missing required input column: {e}. Expected: {self.input_columns}") from e
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return "", ""

    def predict(self, row_data: Dict[str, Any]) -> float:
        """
        Calculates the cosine similarity between two sentences in the row data.
        Returns 0.0 if sentences are empty or an error occurs.
        """
        try:
            sentence1, sentence2 = self.preprocess(row_data)
            
            # Handle cases where preprocessing might return empty strings (e.g., due to NaNs)
            if not sentence1 or not sentence2:
                return 0.0

            embedding1 = np.array(self.model.embed_query(sentence1)).reshape(1, -1)
            embedding2 = np.array(self.model.embed_query(sentence2)).reshape(1, -1)
            
            # Handle potential errors during embedding (e.g., model issues)
            if embedding1.size == 0 or embedding2.size == 0:
                print(f"Warning: Embedding failed for sentences: ['{sentence1}', '{sentence2}']")
                return 0.0

            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(max(0, min(1, similarity))) # Ensure value is between 0 and 1

        except Exception as e:
            # Log the error and the problematic row data for debugging
            # Consider more robust logging in a production environment
            print(f"Error during prediction for columns {self.input_columns}: {e}. Row data sample: {{k: str(v)[:50] for k, v in row_data.items()}}")
            return 0.0 # Return a default value on error

    def batch_predict(self, rows_data: List[Dict[str, Any]]) -> Dict[int, float]:
        """
        Processes a batch of rows by calling predict on each.
        Note: This is not optimized for batch embedding if the underlying model supports it.
        Returns mapping from original index (assumed to be list index) to result.
        """
        # TODO: Implement optimized batch prediction if self.model has an `embed_documents` method.
        results = {}
        for i, row in enumerate(rows_data):
            results[i] = self.predict(row)
        return results

    def initialize(self):
        """Model is initialized externally and passed in."""
        # Ensure the model is still valid/available if needed
        if self.model is None:
            raise RuntimeError("Embedding model was not provided or has been cleaned up.")
        self.is_initialized = True
        return self

    def cleanup(self):
        """
        Cleans up resources. Assumes the embedding model passed in
        might be managed externally, so doesn't delete it by default.
        Set model to None to indicate cleanup.
        """
        # If the embedding model's lifecycle is tied ONLY to this runner,
        # you might add model-specific cleanup here.
        # For now, we just release the reference.
        # print(f"Cleaning up SimilarityModelRunner for {self.model_path}")
        self.model = None
        self.is_initialized = False
        return True
