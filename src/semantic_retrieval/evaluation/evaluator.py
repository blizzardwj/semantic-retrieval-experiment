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
        retrieved_indices_list, 
        ground_truth_labels, 
        sentence2_indices
    ):
        """Evaluate retrieval results against ground truth.
        
        Args:
            retrieved_indices_list (list): List of lists, where each inner list contains
                                          the indices of retrieved sentences for a query
            ground_truth_labels (list): Ground truth labels
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
        
        # For each query's retrieved results
        for retrieved_indices in retrieved_indices_list:
            # Get ground truth labels for retrieved indices
            retrieved_labels = [ground_truth_labels[i] for i in retrieved_indices]
            
            # Calculate metrics
            true_positives = sum(retrieved_labels)
            false_positives = len(retrieved_labels) - true_positives
            
            # Find all positive examples in the ground truth
            positive_indices = [i for i, label in enumerate(ground_truth_labels) if label == 1]
            total_positives = len(positive_indices)
            
            # Calculate precision, recall, F1
            precision = true_positives / len(retrieved_labels) if len(retrieved_labels) > 0 else 0
            recall = true_positives / total_positives if total_positives > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            
            # Calculate MRR
            mrr = 0
            for i, idx in enumerate(retrieved_indices):
                if ground_truth_labels[idx] == 1:
                    mrr = 1 / (i + 1)
                    break
            
            # Calculate NDCG
            dcg = 0
            idcg = 0
            
            # Calculate DCG
            for i, idx in enumerate(retrieved_indices):
                rel = ground_truth_labels[idx]
                # Using binary relevance (0 or 1)
                dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed
            
            # Calculate IDCG (ideal DCG)
            # Sort by relevance (1s first, then 0s)
            ideal_relevance = sorted([ground_truth_labels[i] for i in retrieved_indices], reverse=True)
            for i, rel in enumerate(ideal_relevance):
                idcg += rel / np.log2(i + 2)
            
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
    
    def compare_approaches(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
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
                         ground_truth_labels: List[int],
                         top_k: int = 5,
                         initial_top_k: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a single retrieval approach.
        
        Args:
            approach_name (str): Name of the approach
            retriever: Retrieval approach instance
            query_sentences (list): List of query sentences
            sentence2_list (list): List of sentence2 entries
            ground_truth_labels (list): Ground truth labels
            top_k (int): Number of top results to return
            initial_top_k (int, optional): Initial top_k for approaches with reranking
            
        Returns:
            dict: Evaluation results
        """
        # Index sentences if not already indexed
        if hasattr(retriever, 'index_sentences') and callable(retriever.index_sentences):
            retriever.index_sentences(sentence2_list)
        
        # Retrieve results for each query
        retrieved_indices_list = []
        retrieved_results_list = []
        retrieved_similarities_list = []
        
        for query in query_sentences:
            if initial_top_k is not None and hasattr(retriever, 'retrieve') and callable(retriever.retrieve):
                results, similarities, indices = retriever.retrieve(query, top_k=top_k, initial_top_k=initial_top_k)
            else:
                results, similarities, indices = retriever.retrieve(query, top_k=top_k)
            
            retrieved_results_list.append(results)
            retrieved_similarities_list.append(similarities)
            retrieved_indices_list.append(indices)
        
        # Evaluate retrieval results
        evaluation_metrics = self.evaluate(
            retrieved_indices_list=retrieved_indices_list,
            ground_truth_labels=ground_truth_labels,
            sentence2_indices=list(range(len(sentence2_list)))
        )
        
        # Compile results
        results = {
            "approach_name": approach_name,
            "retrieved_results": retrieved_results_list,
            "retrieved_similarities": retrieved_similarities_list,
            "retrieved_indices": retrieved_indices_list,
            "evaluation": evaluation_metrics
        }
        
        return results
    
    def evaluate_multiple_approaches(self,
                                   retrievers: Dict[str, Any],
                                   query_sentences: List[str],
                                   sentence2_list: List[str],
                                   ground_truth_labels: List[int],
                                   top_k: int = 5,
                                   initial_top_k: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple retrieval approaches.
        
        Args:
            retrievers (dict): Dictionary of retriever instances with approach names as keys
            query_sentences (list): List of query sentences
            sentence2_list (list): List of sentence2 entries
            ground_truth_labels (list): Ground truth labels
            top_k (int): Number of top results to return
            initial_top_k (dict, optional): Dictionary of initial top_k values for approaches with reranking
            
        Returns:
            dict: Evaluation results for all approaches
        """
        results = {}
        
        for approach_name, retriever in retrievers.items():
            init_top_k = initial_top_k.get(approach_name) if initial_top_k else None
            
            approach_results = self.evaluate_approach(
                approach_name=approach_name,
                retriever=retriever,
                query_sentences=query_sentences,
                sentence2_list=sentence2_list,
                ground_truth_labels=ground_truth_labels,
                top_k=top_k,
                initial_top_k=init_top_k
            )
            
            results[approach_name] = approach_results
        
        return results
    
    def save_results(self, results: Dict[str, Dict[str, Any]], output_dir: str = "evaluation_results") -> str:
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
    
    def visualize_comparison(self, comparison: Dict[str, Dict[str, float]], 
                           output_path: Optional[str] = None) -> None:
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
