# Project Directory Structure

This document outlines the overall directory structure of the semantic-retrieval-experiment project.

## Main Project Structure

```
semantic-retrieval-experiment/
├── README.md                  # Project documentation and usage instructions
├── requirements.txt           # Package dependencies
├── setup.py                   # Installation configuration
├── .gitignore                 # Files to exclude from version control
├── LICENSE                    # Project license
├── pyproject.toml             # Modern Python project configuration
├── main.py                    # Entry point for running the experiment
├── examples/                  # Example usage scripts
├── notebooks/                 # Jupyter notebooks for analysis
├── docs/                      # Documentation files
├── data/                      # Sample and test data
├── tests/                     # Test suite
└── src/                       # Source code
```

## Source Code Structure

```
src/
└── semantic_retrieval/        # Main package
    ├── __init__.py            # Package initialization and version
    ├── config.py              # Configuration settings and defaults
    ├── data/                  # Data management module
    │   ├── __init__.py
    │   └── data_loader.py     # DataLoader implementation
    │
    ├── embedding/             # Embedding models module
    │   ├── __init__.py
    │   ├── base.py            # Abstract EmbeddingModel base class
    │   ├── fasttext.py        # FastTextEmbedding implementation
    │   └── bge.py             # BGEEmbedding implementation
    │
    ├── similarity/            # Similarity computation module
    │   ├── __init__.py
    │   └── calculator.py      # SimilarityCalculator implementation
    │
    ├── llm/                   # LLM integration module
    │   ├── __init__.py
    │   └── model.py           # LLMModel implementation
    │
    ├── retrieval/             # Retrieval approaches module
    │   ├── __init__.py
    │   ├── base.py            # Abstract RetrievalApproach base class
    │   ├── word2vec.py        # Word2VecApproach implementation
    │   ├── bge.py             # BGEApproach implementation
    │   └── llm_reranked.py    # LLMRerankedBGEApproach implementation
    │
    ├── evaluation/            # Evaluation module
    │   ├── __init__.py
    │   └── evaluator.py       # Evaluator implementation
    │
    └── experiment/            # Experiment orchestration module
        ├── __init__.py
        ├── runner.py          # ExperimentRunner implementation
        └── visualization.py   # Results visualization utilities
```

## Tests Structure

```
tests/
├── __init__.py
├── conftest.py                # Test fixtures and configuration
├── test_data/                 # Test data files
│   └── test_dataset.csv
├── test_data_loader.py        # Tests for data loading
├── test_embedding.py          # Tests for embedding models
├── test_similarity.py         # Tests for similarity calculation
├── test_llm.py                # Tests for LLM integration
├── test_retrieval.py          # Tests for retrieval approaches
├── test_evaluation.py         # Tests for evaluation metrics
└── test_experiment.py         # Tests for experiment orchestration
```

## Documentation Structure

```
docs/
├── architecture.md            # System architecture overview
├── api_reference.md           # API documentation for all modules
├── installation.md            # Installation instructions
├── usage.md                   # Usage guide
├── development.md             # Development guide for contributors
├── directory_structure.md     # This file - project directory structure
└── examples/                  # Documented code examples
```

## Examples Structure

```
examples/
├── simple_experiment.py       # Basic experiment with all approaches
├── custom_model.py            # How to add a custom embedding model
├── performance_comparison.py  # In-depth comparison script
└── visualize_results.py       # Advanced visualization examples
```

## Notebooks Structure

```
notebooks/
├── experiment_walkthrough.ipynb  # Step-by-step experiment demonstration
├── results_analysis.ipynb        # Analysis of experiment results
└── model_comparison.ipynb        # Detailed model comparison visualizations
```
