import os
from typing import Dict, Any
from dotenv import load_dotenv
import time
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from data_processor import DataProcessor
from prompts import get_prompt_template
from example_questions import get_example_questions
from metrics_handler import MetricsHandler
from answer_generator import AnswerGenerator
from answer_verifier import AnswerVerifier

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="SOW RAG System API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SOWRAGSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.metrics_handler = MetricsHandler()
        self.answer_generator = None
        self.answer_verifier = AnswerVerifier()
        self.current_session_metrics = {
            "queries": [],
            "quality_metrics": {
                "average_faithfulness_score": 0,
                "total_faithful_answers": 0
            }
        }
        
    def setup_qa_chain(self):
        """Setup the QA chain with enhanced retrieval and prompting"""
        print("Setting up QA chain...")
        
        # Get prompt template
        prompt = get_prompt_template()
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator(self.data_processor.vector_store)
        self.answer_generator.setup_qa_chain(prompt)
        
        # Update metrics with QA configuration
        self.metrics_handler.update_system_config({
            "qa_config": {
                "retriever_k": 4,
                "retriever_fetch_k": 8,
                "temperature": 0.1,
                "model": "gpt-3.5-turbo"
            }
        })
        print("QA chain setup complete")
        
    def initialize(self):
        """Initialize the entire RAG system"""
        print("\nInitializing RAG system...")
        
        # Process data
        print("\nStep 1: Loading documents...")
        doc_count = self.data_processor.load_documents()
        if doc_count == 0:
            raise Exception("No documents were loaded. Please check the documents directory.")
        print(f"Loaded {doc_count} documents")
        
        print("\nStep 2: Splitting documents...")
        chunk_count = self.data_processor.split_documents()
        if chunk_count == 0:
            raise Exception("No document chunks were created. Please check the document splitting process.")
        print(f"Split into {chunk_count} chunks")
        
        print("\nStep 3: Creating embeddings...")
        vector_store = self.data_processor.create_embeddings()
        if not vector_store:
            raise Exception("Vector store was not created successfully.")
        print("Embeddings created successfully")
        
        # Verify vector store
        print("\nStep 4: Verifying vector store...")
        test_query = "What are the deliverables?"
        test_results = vector_store.similarity_search(test_query, k=1)
        if not test_results:
            print("Warning: Initial test query returned no results. Checking system state...")
            print(f"Vector store exists: {bool(self.data_processor.vector_store)}")
            print(f"Number of documents: {len(self.data_processor.documents)}")
            print("\nTrying alternative test query...")
            test_query = "What is the project duration?"
            test_results = vector_store.similarity_search(test_query, k=1)
            if not test_results:
                raise Exception("Vector store test queries returned no results. Please check document processing.")
        
        print(f"Vector store verification successful. Found {len(test_results)} results for test query.")
        
        # Setup QA chain
        print("\nStep 5: Setting up QA chain...")
        self.setup_qa_chain()
        print("QA chain setup complete")
        
        # Test QA chain
        print("\nStep 6: Testing QA chain...")
        test_result = self.query(test_query)
        if not test_result or "I cannot find this information" in test_result["answer"]:
            print("\nQA chain test failed. Checking system state...")
            print(f"Vector store exists: {bool(self.data_processor.vector_store)}")
            print(f"Number of documents: {len(self.data_processor.documents)}")
            print(f"Test query: {test_query}")
            print(f"Test result: {test_result}")
            raise Exception("QA chain test query returned no valid results.")
        
        print("QA chain test successful.")
        
        print("\n" + "="*50)
        print("RAG System Initialized Successfully!")
        print("="*50 + "\n")
        
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with enhanced response format"""
        if not self.answer_generator:
            raise Exception("RAG system not initialized. Call initialize() first.")
        
        # Start timing total response
        start_time = time.time()
        
        # Generate answer using the answer generator (actual query time)
        query_start_time = time.time()
        result = self.answer_generator.generate_answer(question)
        query_time = time.time() - query_start_time
        
        # Verify the answer
        is_faithful, latency, faithfulness_score = self.answer_verifier.verify_answer(
            question,
            result["answer"],
            result["sources"]
        )
        
        # Calculate total response time (includes all steps)
        response_time = time.time() - start_time
        
        # Update metrics using metrics handler
        self.metrics_handler.add_query_metrics(
            question=question,
            answer=result["answer"],
            sources=result["sources"],
            response_time=response_time,
            query_time=query_time,  # Only the time spent in answer generation
            latency=latency,
            faithfulness_score=faithfulness_score
        )
        
        # Update current session metrics
        self.current_session_metrics = self.metrics_handler.metrics
        
        # Add detailed metrics to the result
        result["faithfulness_score"] = faithfulness_score
        result["latency"] = latency
        result["response_time"] = response_time  # Total time including all steps
        result["query_time"] = query_time  # Only time spent generating answer
        
        return result
    
    def save_metrics(self):
        """Save system metrics"""
        return self.metrics_handler.save_metrics()
    
    def get_current_metrics(self):
        """Get current session metrics"""
        return self.current_session_metrics

# Create global RAG system instance
rag_system = SOWRAGSystem()

# Pydantic models for API
class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]

# API endpoints
@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

@app.get("/metrics")
async def get_metrics():
    """Get current session metrics"""
    return rag_system.get_current_metrics()

@app.get("/questions")
async def get_questions():
    """Get all example questions"""
    return {"questions": get_example_questions()}

@app.post("/query")
async def query(question: Question, background_tasks: BackgroundTasks):
    """Query the RAG system"""
    try:
        result = rag_system.query(question.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def main():
    # Initialize the RAG system
    print("Initializing RAG system...")
    rag_system.initialize()
    
    # Get example questions
    example_questions = get_example_questions()
    
    # Run example queries
    print("\nExample queries and answers:")
    for question in example_questions:
        print(f"\nQuestion: {question}")
        result = rag_system.query(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {len(result['sources'])}")
    
    # Save metrics
    rag_system.save_metrics()
    
    # Start the API server
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 