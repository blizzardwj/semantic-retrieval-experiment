"""
Evaluates retrieval results against ground truth.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime


class Evaluator:
    """Evaluates retrieval results against ground truth."""
    
    def evaluate(self, 
        retrieved_indices_list: List[List[int]] , 
        ground_truth_labels: List[Any], 
        sentence2_indices: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval results against ground truth.
        
        Args:
            retrieved_indices_list (list): List of lists, where each inner list contains
                                          the indices of retrieved sentences for a query
            ground_truth_labels (list): Ground truth labels. Can be a 1D list (same labels for all queries)
                                       or a 2D list (different labels for each query)
            sentence2_indices (list): Indices of sentence2 entries
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            "precision": [],
            "recall": [],
            "f1": [],
            "mrr": [],  # Mean Reciprocal Rank
            "ndcg": []  # Normalized Discounted Cumulative Gain
        }
        
        # Check if ground_truth_labels is 2D (different labels for each query)
        is_2d_labels = isinstance(ground_truth_labels[0], list)
        
        # For each query's retrieved results
        for i, retrieved_indices in enumerate(retrieved_indices_list):
            # Get the appropriate ground truth labels for this query
            query_labels = ground_truth_labels[i] if is_2d_labels else ground_truth_labels
            
            # Get ground truth labels for retrieved indices
            retrieved_labels = [query_labels[idx] for idx in retrieved_indices]
            
            # Calculate metrics
            true_positives = sum(retrieved_labels)
            false_positives = len(retrieved_labels) - true_positives
            
            # Find all positive examples in the ground truth
            positive_indices = [j for j, label in enumerate(query_labels) if label == 1]
            total_positives = len(positive_indices)
            
            # Calculate precision, recall, F1
            precision = true_positives / len(retrieved_labels) if len(retrieved_labels) > 0 else 0
            recall = true_positives / total_positives if total_positives > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            # Calculate MRR
            mrr = 0
            for j, idx in enumerate(retrieved_indices):
                if query_labels[idx] == 1:
                    mrr = 1 / (j + 1)
                    break
            
            # Calculate NDCG
            dcg = 0
            idcg = 0
            
            # Calculate DCG
            for j, idx in enumerate(retrieved_indices):
                rel = query_labels[idx]
                # Using binary relevance (0 or 1)
                dcg += rel / np.log2(j + 2)  # j+2 because j is 0-indexed
            
            # Calculate IDCG (ideal DCG)
            # Sort by relevance (1s first, then 0s)
            ideal_relevance = sorted([query_labels[idx] for idx in retrieved_indices], reverse=True)
            for j, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(j + 2)
            
            # Calculate NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # Append metrics
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["mrr"].append(mrr)
            metrics["ndcg"].append(ndcg)
        
        # Average metrics across all queries
        for metric in metrics:
            metrics[metric] = sum(metrics[metric]) / len(metrics[metric]) if metrics[metric] else 0
        
        return metrics
    
    def compare_approaches(self, 
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Compare evaluation results from different approaches.
        
        Args:
            results (dict): Results from different approaches
            
        Returns:
            dict: Comparison of approaches
        """
        comparison = {}
        for approach_name, approach_results in results.items():
            comparison[approach_name] = approach_results["evaluation"]
        
        return comparison
    
    def evaluate_approach(self, 
        approach_name: str,
        retriever: Any,
        query_sentences: List[str],
        sentence2_list: List[str],
        ground_truth_labels: List[Any],
        top_k: int = 5,
        initial_top_k: Optional[int] = None
    ):
        """
        Evaluate a single retrieval approach.
        
        Args:
            approach_name (str): Name of the approach
            retriever: Retrieval approach instance
            query_sentences (list): List of query sentences
            sentence2_list (list): List of sentence2 entries
            ground_truth_labels (list): Ground truth labels. Can be a 1D list (same labels for all queries)
                                       or a 2D list (different labels for each query)
            top_k (int): Number of top results to return
            initial_top_k (int, optional): Initial top_k for approaches with reranking
            
        Returns:
            dict: Evaluation results
        """
        # Initialize retriever if needed
        if hasattr(retriever, 'index_sentences') and callable(retriever.index_sentences):
            retriever.index_sentences(sentence2_list)
        
        # Store retrieved results
        retrieved_indices_list = []
        retrieved_results_list = []  # Keep this name for backward compatibility
        retrieved_similarities_list = []
        
        # For each query
        for query_sentence in query_sentences:
            # Retrieve results
            try:
                # Try with initial_top_k if provided
                if initial_top_k is not None:
                    try:
                        results, similarities, indices = retriever.retrieve(
                            query_sentence=query_sentence,
                            top_k=top_k,
                            initial_top_k=initial_top_k
                        )
                    except TypeError:
                        # Fallback if initial_top_k is not supported
                        results, similarities, indices = retriever.retrieve(
                            query_sentence=query_sentence,
                            top_k=top_k
                        )
                else:
                    # No initial_top_k
                    results, similarities, indices = retriever.retrieve(
                        query_sentence=query_sentence,
                        top_k=top_k
                    )
            except TypeError:
                # Try with positional arguments
                results, similarities, indices = retriever.retrieve(
                    query_sentence,
                    top_k
                )
            
            # Store results
            retrieved_indices_list.append(indices)
            retrieved_results_list.append(results)
            retrieved_similarities_list.append(similarities)
        
        # Create sentence2 indices
        sentence2_indices = list(range(len(sentence2_list)))
        
        # Evaluate retrieval results
        evaluation_metrics = self.evaluate(
            retrieved_indices_list=retrieved_indices_list,
            ground_truth_labels=ground_truth_labels,
            sentence2_indices=sentence2_indices
        )
        
        # Store results
        results = {
            "approach_name": approach_name,
            "evaluation": evaluation_metrics,
            "retrieved_indices": retrieved_indices_list,
            "retrieved_results": retrieved_results_list,  # Keep this name for backward compatibility
            "retrieved_similarities": retrieved_similarities_list
        }
        
        return results
    
    def evaluate_multiple_approaches(self,
        retrievers: Dict[str, Any],
        query_sentences: List[str],
        sentence2_list: List[str],
        ground_truth_labels: List[Any],
        top_k: int = 5,
        initial_top_k: Optional[Dict[str, int]] = None
    ):
        """
        Evaluate multiple retrieval approaches.
        
        Args:
            retrievers (dict): Dictionary of retriever instances with approach names as keys
            query_sentences (list): List of query sentences
            sentence2_list (list): List of sentence2 entries
            ground_truth_labels (list): Ground truth labels. Can be a 1D list (same labels for all queries)
                                       or a 2D list (different labels for each query)
            top_k (int): Number of top results to return
            initial_top_k (dict, optional): Dictionary of initial top_k values for approaches with reranking
            
        Returns:
            dict: Evaluation results for all approaches
        """
        results = {}
        
        # For each approach
        for approach_name, retriever in retrievers.items():
            # Get initial_top_k for this approach if specified
            approach_initial_top_k = None
            if initial_top_k is not None and approach_name in initial_top_k:
                approach_initial_top_k = initial_top_k[approach_name]
            
            # Evaluate approach
            approach_results = self.evaluate_approach(
                approach_name=approach_name,
                retriever=retriever,
                query_sentences=query_sentences,
                sentence2_list=sentence2_list,
                ground_truth_labels=ground_truth_labels,
                top_k=top_k,
                initial_top_k=approach_initial_top_k
            )
            
            # Store results
            results[approach_name] = approach_results
        
        return results
    
    def save_results(self, 
        results: Dict[str, Dict[str, Any]], 
        output_dir: str = "evaluation_results"
    ) -> str:
        """Save evaluation results to file.
        
        Args:
            results (dict): Evaluation results
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved results file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Prepare results for serialization (convert numpy arrays to lists)
        serializable_results = {}
        for approach_name, approach_results in results.items():
            serializable_approach = {}
            for key, value in approach_results.items():
                if key == "retrieved_similarities":
                    # Convert list of numpy arrays to list of lists
                    serializable_approach[key] = [[float(sim) for sim in sims] for sims in value]
                elif key == "evaluation":
                    # Convert metrics dict with potential numpy values to dict with Python types
                    serializable_approach[key] = {k: float(v) for k, v in value.items()}
                else:
                    serializable_approach[key] = value
            serializable_results[approach_name] = serializable_approach
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Dict[str, Any]]:
        """Load evaluation results from file.
        
        Args:
            filepath (str): Path to results file
            
        Returns:
            dict: Evaluation results
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    def visualize_comparison(self, 
        comparison: Dict[str, Dict[str, float]], 
        output_path: Optional[str] = None
    ) -> None:
        """Visualize comparison of different approaches.
        
        Args:
            comparison (dict): Comparison of approaches
            output_path (str, optional): Path to save the visualization
        """
        # Extract metrics and approach names
        metrics = list(next(iter(comparison.values())).keys())
        approaches = list(comparison.keys())
        
        # Set up the figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
        
        # If there's only one metric, axes will not be an array
        if len(metrics) == 1:
            axes = [axes]
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [comparison[approach][metric] for approach in approaches]
            
            # Create bar chart
            axes[i].bar(approaches, values)
            axes[i].set_title(f"{metric.upper()}")
            axes[i].set_ylim(0, 1)  # Most metrics are between 0 and 1
            
            # Add values on top of bars
            for j, value in enumerate(values):
                axes[i].text(j, value + 0.01, f"{value:.3f}", ha='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
