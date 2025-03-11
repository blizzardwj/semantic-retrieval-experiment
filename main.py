#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the semantic retrieval experiment.
"""

def main():
    """Main function to run the experiment."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run comparative semantic retrieval experiment')
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top results to retrieve')
    parser.add_argument('--sample-size', type=int, default=None, help='Number of query sentences to sample')
    args = parser.parse_args()
    
    # Initialize data loader
    data_loader = DataLoader(args.data)
    
    # Initialize approaches
    word2vec_approach = Word2VecApproach()
    bge_approach = BGEApproach()
    llm_reranked_approach = LLMRerankedBGEApproach()
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Initialize experiment runner
    experiment_runner = ExperimentRunner(
        data_loader,
        [word2vec_approach, bge_approach, llm_reranked_approach],
        evaluator
    )
    
    # Load data
    data_loader.load_data()
    
    # Sample query sentences if specified
    query_sentences = None
    if args.sample_size is not None:
        import random
        sentence1_list = data_loader.get_sentence1_list()
        query_sentences = random.sample(sentence1_list, min(args.sample_size, len(sentence1_list)))
    
    # Run experiment
    results = experiment_runner.run_experiment(query_sentences, args.top_k)
    
    # Compare results
    comparison = experiment_runner.compare_results(results)
    
    # Print comparison
    print("\nMetrics Comparison:")
    for approach, metrics in comparison["metrics"].items():
        print(f"\n{approach}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\nTiming Comparison:")
    for approach, timing in comparison["timing"].items():
        print(f"\n{approach}:")
        print(f"  Indexing Time: {timing['indexing_time']:.4f} seconds")
        print(f"  Avg. Retrieval Time: {timing['avg_retrieval_time']:.4f} seconds")
    
    # Visualize comparison
    experiment_runner.visualize_comparison(comparison)


if __name__ == "__main__":
    main()

