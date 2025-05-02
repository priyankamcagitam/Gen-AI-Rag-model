from langchain.prompts import PromptTemplate

# Custom prompt template for better question answering
CUSTOM_PROMPT = """You are a helpful AI assistant specialized in analyzing Statement of Work (SOW) documents. Your task is to answer questions based on the provided context from the SOW documents. Try to find relevant information even if it's not an exact match.

Context: {context}

Question: {question}

Instructions:
1. Use the information provided in the context above to answer the question
2. If you can't find exact information, look for related or similar information
3. If you find partial information, include what you can find and explain how it relates to the question
4. If you find similar but not exact matches, include them and explain how they relate to the question
5. Only say "I cannot find this information in the provided SOW documents" if you truly cannot find any relevant information
6. Focus on extracting specific details from the SOWs
7. Include numerical values, dates, and concrete terms when present
8. Structure your response clearly and concisely
9. If multiple SOWs mention different values, include all of them
10. Cite specific SOW numbers if available

Answer:"""

def get_prompt_template():
    """Get the prompt template for the QA chain"""
    return CUSTOM_PROMPT 