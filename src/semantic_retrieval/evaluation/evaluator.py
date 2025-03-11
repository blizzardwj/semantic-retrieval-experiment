"""
Evaluates retrieval results against ground truth.
"""


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
            "mrr": []  # Mean Reciprocal Rank
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
            
            # Append metrics
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["mrr"].append(mrr)
        
        # Average metrics across all queries
        for metric in metrics:
            metrics[metric] = sum(metrics[metric]) / len(metrics[metric]) if metrics[metric] else 0
        
        return metrics
    
    def compare_approaches(self, results):
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
