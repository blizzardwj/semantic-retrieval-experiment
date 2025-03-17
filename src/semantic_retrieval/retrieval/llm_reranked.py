"""
Implementation of Approach 3: BGE embedding with LLM reranking.
"""
from semantic_retrieval.retrieval.base import RetrievalApproach
from semantic_retrieval.embedding.bge import BGEEmbedding
from semantic_retrieval.llm.model import LLMModel
from typing import List, Tuple, Optional
from semantic_retrieval.retrieval.base import InitialRetrievalStrategy, DirectComparisonStrategy, VectorStoreStrategy


class LLMRerankedBGEApproach(RetrievalApproach):
    """Implementation of Approach 3: BGE embedding with LLM reranking."""
    
    def __init__(self, 
        embedding_model: str = "bge-large-zh-v1.5",
        embedding_server_url: str = "http://20.30.80.200:9997",
        llm_model: str = "deepseek-r1-distill-qwen", 
        llm_server_url: str = "http://20.30.80.200:9997",
        persist_directory: Optional[str] = None, 
        use_vector_store: bool = False
    ):
        """Initialize LLM-reranked BGE approach.
        
        Args:
            embedding_model (str): Name of the BGE model to use
            embedding_server_url (str): URL of the BGE model server
            llm_model (str): Name of the LLM model to use
            llm_server_url (str): URL of the LLM model server
            persist_directory (str, optional): Directory to persist vector store. If None, 
                an in-memory vector store will be used.
            use_vector_store (bool): Whether to use vector store for initial retrieval
        """
        super().__init__("LLM-Reranked BGE")
        self.embedding_model = BGEEmbedding(embedding_model)
        self.llm_model = LLMModel(llm_model)  # This could be changed to a different LLM
        self.embedding_server_url = embedding_server_url
        self.llm_server_url = llm_server_url
        self.persist_directory = persist_directory
        self.sentence_list = None
        self.use_vector_store = use_vector_store
        
        # Set the initial retrieval strategy based on the use_vector_store flag
        self.retrieval_strategy = self._create_retrieval_strategy(use_vector_store)
    
    def _create_retrieval_strategy(self, use_vector_store: bool) -> InitialRetrievalStrategy:
        """Helper method to create the appropriate retrieval strategy.
        
        Args:
            use_vector_store (bool): Whether to use vector store
            
        Returns:
            InitialRetrievalStrategy: The selected retrieval strategy
        """
        if use_vector_store:
            return VectorStoreStrategy(self.persist_directory)
        else:
            return DirectComparisonStrategy()
    
    def set_retrieval_strategy(self, strategy: InitialRetrievalStrategy) -> None:
        """Set the initial retrieval strategy.
        
        Args:
            strategy: The retrieval strategy to use
        """
        self.retrieval_strategy = strategy
        # Re-initialize if we already have sentences
        if self.sentence_list:
            self.retrieval_strategy.initialize(self.sentence_list, self.embedding_model)
    
    def index_sentences(self, sentence_list: List[str], use_vector_store: Optional[bool] = None) -> None:
        """Index sentence list using the current retrieval strategy or optionally switch strategy.
        
        Args:
            sentence_list (list): List of sentence entries
            use_vector_store (bool, optional): If provided, switches to vector store strategy (True)
                                              or direct comparison strategy (False)
        """
        self.sentence_list = sentence_list
        self.embedding_model.initialize(server_url=self.embedding_server_url)
        self.llm_model.initialize(server_url=self.llm_server_url)
        
        # Check if we need to change the strategy
        if use_vector_store is not None and use_vector_store != self.use_vector_store:
            self.use_vector_store = use_vector_store
            self.retrieval_strategy = self._create_retrieval_strategy(use_vector_store)
        
        # Initialize the strategy with the sentences
        self.retrieval_strategy.initialize(sentence_list, self.embedding_model)
    
    def retrieve(self, query_sentence: str, top_k: int = 5, initial_top_k: int = 10) -> Tuple[List[str], List[float], List[int]]:
        """Retrieve top_k similar sentences using initial retrieval and LLM reranking.
        
        Args:
            query_sentence (str): Query sentence
            top_k (int): Number of top results to return
            initial_top_k (int): Number of initial candidates for LLM reranking
            
        Returns:
            tuple: (top_sentences, top_similarities, top_indices)
        """
        # First get initial_top_k candidates using the current retrieval strategy
        initial_top_sentences, initial_top_similarities, initial_top_indices = self.retrieval_strategy.retrieve(
            query_sentence, self.embedding_model, initial_top_k
        )
        
        # Use LLM to rerank
        reranked_indices = self.llm_model.rerank(
            query_sentence, initial_top_sentences, initial_top_indices, top_k
        )
        
        # Get final top_k sentences and their similarities
        # reranked_indices 已经是映射到原始语料库的索引，直接使用它
        final_top_indices = reranked_indices[:top_k]
        
        # 为了获取相似度和句子，需要找到这些索引在 initial_top_indices 中的位置
        final_top_similarities = []
        final_top_sentences = []
        
        for idx in final_top_indices:
            # 查找索引在 initial_top_indices 中的位置
            if idx in initial_top_indices:
                pos = initial_top_indices.index(idx)
                final_top_similarities.append(initial_top_similarities[pos])
                final_top_sentences.append(initial_top_sentences[pos])
            else:
                # 如果找不到对应的索引（可能是LLM返回了不在候选集中的索引），
                # 则使用原始语料库中的句子（如果可用）
                if self.sentence_list and 0 <= idx < len(self.sentence_list):
                    final_top_sentences.append(self.sentence_list[idx])
                    # 由于没有相似度分数，使用一个默认值（例如0）
                    final_top_similarities.append(0.0)
                else:
                    # 如果索引无效，跳过这个结果
                    continue
        
        return final_top_sentences, final_top_similarities, final_top_indices
    
    def batch_retrieve(self, query_sentences: List[str], top_k: int = 5, initial_top_k: int = 10) -> List[Tuple[List[str], List[float], List[int]]]:
        """Retrieve top_k similar sentences for multiple queries.
        
        Args:
            query_sentences (list): List of query sentences
            top_k (int): Number of top results to return for each query
            initial_top_k (int): Number of initial candidates for LLM reranking
            
        Returns:
            list: List of tuples (top_sentences, top_similarities, top_indices) for each query
        """
        results = []
        
        for query in query_sentences:
            query_result = self.retrieve(query, top_k, initial_top_k)
            results.append(query_result)
        
        return results
