from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

class AnswerGenerator:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.qa_chain = None
        self.llm = ChatOpenAI(temperature=0.1)
        
    def setup_qa_chain(self, prompt_template: str):
        """Setup the QA chain with the given prompt template"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 8}  # Increase number of retrieved documents
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """Generate an answer for the given question"""
        if not self.qa_chain:
            raise Exception("QA chain not initialized. Call setup_qa_chain first.")
            
        # Get answer from QA chain
        result = self.qa_chain({"query": question})
        
        # Process the answer
        answer = result["result"]
        sources = result["source_documents"]
        
        # If no answer found, try to get more context
        if "I cannot find this information" in answer:
            # Try to get more documents with a broader search
            docs = self.vector_store.similarity_search(
                question,
                k=10  # Get more documents
            )
            
            if docs:
                # Update the answer with the new context
                result = self.qa_chain({"query": question})
                answer = result["result"]
                sources = result["source_documents"]
        
        return {
            "answer": answer,
            "sources": sources
        } 