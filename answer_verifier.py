import time
from typing import Dict, Any, List, Tuple
from langchain_openai import OpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)

class AnswerVerifier:
    def __init__(self):
        self.llm = OpenAI(temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        
    def verify_answer(self, question: str, answer: str, sources: List[Any]) -> Tuple[bool, float, float]:
        """
        Verify the answer against the sources and calculate faithfulness score
        
        Args:
            question: The original question
            answer: The generated answer
            sources: List of source documents
            
        Returns:
            Tuple of (is_faithful, latency, faithfulness_score)
        """
        start_time = time.time()
        
        # Check if answer is empty or default response
        if not answer or "I cannot find this information" in answer:
            logger.warning("Empty or default response detected")
            return False, time.time() - start_time, 0.0
            
        # Extract key information from sources
        source_texts = [source.page_content for source in sources if hasattr(source, 'page_content')]
        
        if not source_texts:
            logger.warning("No source texts found for verification")
            return False, time.time() - start_time, 0.0
        
        # Verify answer against source information
        faithfulness_score = self._calculate_faithfulness(answer, source_texts)
        
        # Log the verification results
        logger.info(f"Faithfulness score: {faithfulness_score:.2f}")
        logger.info(f"Number of sources used: {len(source_texts)}")
        
        is_faithful = faithfulness_score > 0.7
        
        latency = time.time() - start_time
        logger.info(f"Verification latency: {latency:.2f} seconds")
        
        return is_faithful, latency, faithfulness_score
    
    def _calculate_faithfulness(self, answer: str, source_texts: List[str]) -> float:
        """Calculate faithfulness score based on text content"""
        try:
            # Create embeddings for answer and source text
            answer_embedding = self.embeddings.embed_query(answer)
            source_embeddings = [self.embeddings.embed_query(text) for text in source_texts]
            
            # Calculate similarity scores
            similarities = []
            for source_embedding in source_embeddings:
                similarity = self._cosine_similarity(answer_embedding, source_embedding)
                similarities.append(similarity)
            
            # Return highest similarity score
            max_similarity = max(similarities) if similarities else 0.0
            logger.info(f"Max similarity score: {max_similarity:.2f}")
            return max_similarity
            
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {str(e)}")
            return 0.0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0 