"""
Interfaces with a large language model for reranking.
"""

from typing import List, Optional, Union, Any, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

class LLMModel():
    """Interfaces with a large language model for reranking."""

    def __init__(self, model_name: str, model_type: str = "chat"):
        """Initialize with model name and model type.
        
        Args:
            model_name (str): Name of the LLM model in XInference
            model_type (str): Type of model ('chat' or 'text')
        """
        self.model_name = model_name    # the same as model_uid in Xinference
        self.model_type = model_type
        self.server_url = None
        self.chat_model = None
        self.model = None
    
    def initialize(self, server_url: str = "http://localhost:9997") -> None:
        """Initialize and load the XInference model via langchain.
        
        Args:
            server_url (str): URL of the XInference server
            
        Raises:
            ConnectionError: If connection to XInference server fails
            ValueError: If model_type is invalid
        """
        self.server_url = server_url
        
        try:
            from langchain_community.llms.xinference import Xinference
            
            if self.model_type == "chat":
                # Create a chat model using XInference
                self.chat_model = Xinference(
                    server_url=self.server_url,
                    model_uid=self.model_name
                )
            elif self.model_type == "text":
                # Create a text completion model using XInference
                self.model = Xinference(
                    server_url=self.server_url,
                    model_uid=self.model_name
                )
            else:
                raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'chat' or 'text'")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize LLM model: {e}")
    
    def _ensure_initialized(self) -> None:
        """Ensure the model is initialized before use.
        
        Raises:
            ValueError: If server_url is not set
        """
        if not self.server_url:
            raise ValueError("Server URL not set. Call initialize() with server_url first.")
            
        if not self.chat_model and not self.model:
            self.initialize(self.server_url)
    
    def rerank(self, query_sentence: str, candidate_sentences: List[str], 
              candidate_indices: List[int], top_k: int = 5) -> List[int]:
        """Rerank candidate sentences using XInference LLM.
        
        Args:
            query_sentence (str): Query sentence
            candidate_sentences (list): List of candidate sentences
            candidate_indices (list): Original indices of candidate sentences
            top_k (int): Number of top results to return
            
        Returns:
            list: Indices of top_k reranked candidates
            
        Raises:
            ValueError: If model is not initialized properly
        """
        self._ensure_initialized()
        
        # Format candidates for the prompt
        formatted_candidates = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(candidate_sentences)])
        
        try:
            # Create system and human messages for the chat model
            if self.model_type == "chat":
                system_message = SystemMessage(content=f"""
                You are a semantic search expert. Your task is to find the sentences that are semantically 
                most similar to a given query. Analyze the semantic similarity between the query and each candidate.
                Return ONLY the numbers of the top {top_k} most semantically similar sentences in order of relevance,
                separated by commas without any explanation.
                """)
                
                human_message = HumanMessage(content=f"""
                Query: "{query_sentence}"
                
                Candidate sentences:
                {formatted_candidates}
                
                Return the numbers of the top {top_k} most semantically similar sentences.
                """)
                
                # Get response from the chat model
                try:
                    response = self.chat_model.invoke([system_message, human_message])
                    # 处理返回值可能是字符串或具有 content 属性的对象的情况
                    if hasattr(response, 'content'):
                        result = response.content.strip()
                    else:
                        result = str(response).strip()
                except Exception as e:
                    print(f"Error invoking LLM: {e}")
                    # Fallback to returning the first top_k indices
                    return self._fallback_indices(candidate_indices, top_k)
            else:
                # For text completion model
                prompt = f"""
                I need to find the top {top_k} sentences that are semantically most similar to the following query:
                Query: "{query_sentence}"
                
                Here are the candidate sentences:
                {formatted_candidates}
                
                Please analyze the semantic similarity between the query and each candidate.
                Return ONLY the numbers of the top {top_k} most semantically similar sentences in order of relevance,
                separated by commas without any explanation.
                """
                
                response = self.model.invoke(prompt)
                result = response.strip()
            
            # Parse the response to get the indices
            try:
                final_indices = self._parse_llm_response(result, candidate_indices, top_k)
                return final_indices
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                return self._fallback_indices(candidate_indices, top_k)
        except Exception as e:
            print(f"Error in reranking process: {e}")
            return self._fallback_indices(candidate_indices, top_k)
    
    def _parse_llm_response(self, result: str, candidate_indices: List[int], 
                           top_k: int) -> List[int]:
        """Parse LLM response to extract indices.
        
        Args:
            result (str): LLM response string
            candidate_indices (List[int]): Original indices
            top_k (int): Number of top results to return
            
        Returns:
            List[int]: Parsed indices
        """
        import re
        numbers = re.findall(r'\d+', result)
        
        # Convert to integers and adjust for 0-based indexing
        reranked_indices = [int(num) - 1 for num in numbers[:top_k]]
        
        # Map back to original indices
        final_indices = [candidate_indices[idx] for idx in reranked_indices 
                        if idx < len(candidate_indices)]
        
        # If we couldn't parse enough indices, fall back to the first top_k
        if len(final_indices) < top_k:
            remaining = top_k - len(final_indices)
            for i in range(len(candidate_indices)):
                if i not in reranked_indices and len(final_indices) < top_k:
                    final_indices.append(candidate_indices[i])
                    remaining -= 1
                if remaining == 0:
                    break
        
        return final_indices[:top_k]
    
    def _fallback_indices(self, candidate_indices: List[int], top_k: int) -> List[int]:
        """Return fallback indices when LLM fails.
        
        Args:
            candidate_indices (List[int]): Original indices
            top_k (int): Number of top results to return
            
        Returns:
            List[int]: First top_k indices
        """
        return [candidate_indices[i] for i in range(min(top_k, len(candidate_indices)))]


"""
LLM model config:
    server_url="http://20.30.80.200:9997/v1",
    model_name="deepseek-r1-distill-qwen"
"""