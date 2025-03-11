"""
Orchestrates the experiment and compares approaches.
"""
from semantic_retrieval.data.data_loader import DataLoader
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.evaluation.evaluator import Evaluator

class ExperimentRunner:
    """Orchestrates the experiment and compares approaches."""
    
    def __init__(self, 
        data_loader: DataLoader, 
        approaches: list[RetrievalApproach], 
        evaluator: Evaluator
    ):
        """Initialize with data loader, approaches, and evaluator.
        
        Args:
            data_loader (DataLoader): Data loader instance
            approaches (list): List of RetrievalApproach instances
            evaluator (Evaluator): Evaluator instance
        """
        self.data_loader = data_loader
        self.approaches = approaches
        self.evaluator = evaluator
    
    def run_experiment(self, query_sentences=None, top_k=5):
        """Run the experiment and compare approaches.
        
        Args:
            query_sentences (list, optional): List of query sentences. If None, uses all sentence1 entries.
            top_k (int): Number of top results to retrieve
            
        Returns:
            dict: Results from different approaches
        """
        import time
        
        # Load data
        self.data_loader.load_data()
        sentence1_list = self.data_loader.get_sentence1_list()
        sentence2_list = self.data_loader.get_sentence2_list()
        labels = self.data_loader.get_labels()
        
        # Set query sentences if not provided
        if query_sentences is None:
            query_sentences = sentence1_list
        
        # Results dictionary
        results = {}
        
        # Run each approach
        for approach in self.approaches:
            print(f"Running approach: {approach.name}")
            
            # Index sentence2 in the approach
            start_time = time.time()
            approach.index_sentences(sentence2_list)
            indexing_time = time.time() - start_time
            
            # Retrieve for each query sentence
            approach_results = []
            retrieval_times = []
            
            for query_sentence in query_sentences:
                start_time = time.time()
                retrieved = approach.retrieve(query_sentence, top_k)
                retrieval_time = time.time() - start_time
                
                approach_results.append(retrieved)
                retrieval_times.append(retrieval_time)
            
            # Evaluate results
            evaluation = self.evaluator.evaluate(
                [result[2] for result in approach_results],  # indices
                labels,
                list(range(len(sentence2_list)))
            )
            
            # Store results
            results[approach.name] = {
                "retrieved": approach_results,
                "evaluation": evaluation,
                "indexing_time": indexing_time,
                "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times)
            }
        
        return results
    
    def compare_results(self, results):
        """Compare results from different approaches.
        
        Args:
            results (dict): Results from different approaches
            
        Returns:
            dict: Comparison of approaches
        """
        # Comparison dictionary
        comparison = {
            "metrics": self.evaluator.compare_approaches(results),
            "timing": {}
        }
        
        # Compare timing
        for approach_name, approach_results in results.items():
            comparison["timing"][approach_name] = {
                "indexing_time": approach_results["indexing_time"],
                "avg_retrieval_time": approach_results["avg_retrieval_time"]
            }
        
        return comparison
    
    def visualize_comparison(self, comparison):
        """Visualize the comparison of approaches.
        
        Args:
            comparison (dict): Comparison of approaches
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Extract approach names and metrics
        approaches = list(comparison["metrics"].keys())
        metrics = ["precision", "recall", "f1", "mrr"]
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
        
        # Bar positions
        x = np.arange(len(approaches))
        width = 0.2
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [comparison["metrics"][approach][metric] for approach in approaches]
            axes[i].bar(x, values, width=width, label=metric)
            axes[i].set_title(metric.capitalize())
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(approaches, rotation=45, ha="right")
            axes[i].set_ylim(0, 1)
            
            # Add values on top of bars
            for j, value in enumerate(values):
                axes[i].text(j, value + 0.05, f"{value:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig("approach_comparison_metrics.png")
        
        # Create a figure for timing comparison
        plt.figure(figsize=(12, 6))
        
        # Extract timing data
        indexing_times = [comparison["timing"][approach]["indexing_time"] for approach in approaches]
        retrieval_times = [comparison["timing"][approach]["avg_retrieval_time"] for approach in approaches]
        
        # Plot timing data
        x = np.arange(len(approaches))
        width = 0.35
        
        plt.bar(x - width/2, indexing_times, width, label='Indexing Time (s)')
        plt.bar(x + width/2, retrieval_times, width, label='Avg. Retrieval Time (s)')
        
        plt.xlabel('Approaches')
        plt.ylabel('Time (seconds)')
        plt.title('Timing Comparison')
        plt.xticks(x, approaches, rotation=45, ha="right")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("approach_comparison_timing.png")
        
        plt.show()
