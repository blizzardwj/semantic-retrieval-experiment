"""
Interfaces with a large language model for reranking.
"""
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, SystemMessage

class LLMModel(ABC):
    """Interfaces with a large language model for reranking."""

    def __init__(self, model_name, server_url="http://localhost:9997", model_type="chat"):
        """Initialize with model name and server URL.
        
        Args:
            model_name (str): Name of the LLM model in XInference
            server_url (str): URL of the XInference server
            model_type (str): Type of model ('chat' or 'text')
        """
        super().__init__(model_name)
        self.server_url = server_url
        self.model_name = model_name
        self.model_type = model_type
        self.chat_model = None
    
    def initialize(self):
        """Initialize and load the XInference model via langchain."""
        from langchain_community.llms.xinference import Xinference
        if self.model_type == "chat":
            # Create a chat model using XInference
            self.chat_model = Xinference(
                server_url=self.server_url,
                model_uid=self.model_name
            )
        else:
            # Create a text completion model using XInference
            self.model = Xinference(
                server_url=self.server_url,
                model_uid=self.model_name
            )
    
    def rerank(self, query_sentence, candidate_sentences, candidate_indices, top_k=5):
        """Rerank candidate sentences using XInference LLM.
        
        Args:
            query_sentence (str): Query sentence
            candidate_sentences (list): List of candidate sentences
            candidate_indices (list): Original indices of candidate sentences
            top_k (int): Number of top results to return
            
        Returns:
            list: Indices of top_k reranked candidates
        """
        if not self.chat_model and not self.model:
            self.initialize()
        
        # Format candidates for the prompt
        formatted_candidates = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(candidate_sentences)])
        
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
            response = self.chat_model.invoke([system_message, human_message])
            result = response.content.strip()
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
            # Extract numbers from the response
            import re
            numbers = re.findall(r'\d+', result)
            
            # Convert to integers and adjust for 0-based indexing
            reranked_indices = [int(num) - 1 for num in numbers[:top_k]]
            
            # Map back to original indices
            final_indices = [candidate_indices[idx] for idx in reranked_indices if idx < len(candidate_indices)]
            
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
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            # Fallback to returning the first top_k indices
            return [candidate_indices[i] for i in range(min(top_k, len(candidate_indices)))]

"""
LLM model config:
    server_url="http://20.30.80.200:9997/v1",
    model_name="deepseek-r1-distill-qwen"
"""