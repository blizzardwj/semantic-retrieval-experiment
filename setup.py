from setuptools import setup, find_packages

setup(
    name="semantic-retrieval-experiment",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=1.9.0",
        "transformers>=4.12.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
    },
    python_requires=">=3.8",
    author="",
    author_email="",
    description="A project for experimenting with semantic retrieval techniques",
    keywords="semantic, retrieval, embeddings, search",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
