from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime
import glob
from rag_system import SOWRAGSystem
import logging
import sys
import asyncio
from contextlib import asynccontextmanager
import time

# Configure logging to print to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system = None
initialization_complete = False

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system, initialization_complete
    
    print("\n" + "="*50)
    print("Initializing RAG System...")
    print("="*50 + "\n")
    
    try:
        # Create a new instance of SOWRAGSystem
        rag_system = SOWRAGSystem()
        
        # Process data
        print("Loading documents...")
        doc_count = rag_system.data_processor.load_documents()
        print(f"Loaded {doc_count} documents")
        
        if doc_count == 0:
            raise Exception("No documents were loaded. Please check the documents directory.")
        
        print("Splitting documents...")
        chunk_count = rag_system.data_processor.split_documents()
        print(f"Split into {chunk_count} chunks")
        
        if chunk_count == 0:
            raise Exception("No document chunks were created. Please check the document splitting process.")
        
        print("Creating embeddings...")
        vector_store = rag_system.data_processor.create_embeddings()
        print("Embeddings created")
        
        # Verify vector store
        if not vector_store:
            raise Exception("Vector store was not created successfully.")
        
        # Test vector store
        print("\nTesting vector store...")
        test_query = "What are the deliverables?"
        test_results = vector_store.similarity_search(test_query, k=1)
        if not test_results:
            # Try to get more documents
            print("No results found in initial test. Trying to get more documents...")
            docs = rag_system.data_processor.documents
            if not docs:
                raise Exception("No documents available for testing.")
            
            # Print sample of documents
            print("\nSample of available documents:")
            for i, doc in enumerate(docs[:3]):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
            
            # Try different query
            print("\nTrying different test query...")
            test_query = "What is the project duration?"
            test_results = vector_store.similarity_search(test_query, k=1)
            if not test_results:
                raise Exception("Vector store test query returned no results. Please check document processing.")
        
        print(f"Vector store test successful. Found {len(test_results)} results.")
        
        # Setup QA chain
        print("\nSetting up QA chain...")
        rag_system.setup_qa_chain()
        print("QA chain setup complete")
        
        # Test QA chain
        print("\nTesting QA chain...")
        test_result = rag_system.query(test_query)
        if not test_result or "I cannot find this information" in test_result["answer"]:
            print("\nQA chain test failed. Checking system state...")
            print(f"Vector store exists: {bool(rag_system.data_processor.vector_store)}")
            print(f"Number of documents: {len(rag_system.data_processor.documents)}")
            print(f"Test query: {test_query}")
            print(f"Test result: {test_result}")
            raise Exception("QA chain test query returned no valid results.")
        
        print("QA chain test successful.")
        
        print("\n" + "="*50)
        print("RAG System Initialized Successfully!")
        print("="*50 + "\n")
        
        initialization_complete = True
        return True
    except Exception as e:
        print("\n" + "="*50)
        print(f"Error Initializing RAG System: {str(e)}")
        print("="*50 + "\n")
        logger.error(f"Error Initializing RAG System: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Startup: Initialize the RAG system
    asyncio.create_task(initialize_rag_system_async())
    yield
    # Shutdown: Clean up resources if needed
    pass

# Create FastAPI app with lifespan
app = FastAPI(title="SOW RAG System API", lifespan=lifespan)

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

class Question(BaseModel):
    text: str

class Answer(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metrics: Dict[str, Any]

@app.get("/status")
async def get_status():
    """Get the current status of the RAG system"""
    global rag_system, initialization_complete
    
    if not initialization_complete:
        return {
            "status": "initializing",
            "message": "RAG system is still initializing. Please try again in a few moments."
        }
    
    if not rag_system:
        return {
            "status": "error",
            "message": "RAG system failed to initialize."
        }
    
    return {
        "status": "ready",
        "message": "RAG system is ready to accept queries."
    }

@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")

@app.get("/metrics")
async def get_metrics():
    """Get all available metrics files"""
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    metrics_files = glob.glob("metrics/*.json")
    metrics_data = []
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics_data.append({
                    "filename": os.path.basename(file_path),
                    "timestamp": data.get("timestamp", ""),
                    "total_queries": len(data.get("queries", [])),
                    "average_faithfulness": data.get("quality_metrics", {}).get("average_faithfulness_score", 0)
                })
        except Exception as e:
            logger.error(f"Error reading {file_path}: {str(e)}")
    
    return metrics_data

@app.get("/metrics/{filename}")
async def get_metric_file(filename: str):
    """Get specific metrics file"""
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    file_path = os.path.join("metrics", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Metrics file not found")
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading metrics file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading metrics file: {str(e)}")

@app.get("/questions")
async def get_questions():
    """Get all example questions"""
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    from example_questions import get_example_questions
    return {"questions": get_example_questions()}

@app.post("/query")
async def query(question: Question, background_tasks: BackgroundTasks):
    """Query the RAG system"""
    global rag_system, initialization_complete
    
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    if not rag_system:
        logger.error("RAG system not initialized")
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Start timing the entire request
        request_start_time = time.time()
        
        logger.info(f"Processing query: {question.text}")
        
        # Verify vector store before querying
        if not rag_system.data_processor.vector_store:
            raise HTTPException(status_code=500, detail="Vector store not initialized")
        
        # Get the answer
        result = rag_system.query(question.text)
        
        # Calculate total request time
        request_time = time.time() - request_start_time
        
        # Add detailed timing metrics to the response
        result["metrics"] = {
            "request_time": request_time,  # Total time including HTTP overhead
            "response_time": result.get("response_time", 0.0),  # Total processing time
            "query_time": result.get("query_time", 0.0),  # Time spent generating answer
            "faithfulness_score": result.get("faithfulness_score", 0.0),
            "latency": result.get("latency", 0.0),
            "sources_count": len(result.get("sources", [])),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save metrics in the background
        background_tasks.add_task(rag_system.save_metrics)
        
        logger.info(f"Query processed successfully in {request_time:.2f} seconds")
        logger.info(f"Detailed timings: query={result['metrics']['query_time']:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/sow-types")
async def get_sow_types():
    """Get all available SOW types"""
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    from generate_sow import SOWGenerator
    generator = SOWGenerator()
    return {"sow_types": generator.sow_types}

@app.post("/generate-sow")
async def generate_sow(sow_type: Optional[str] = None):
    """Generate a new SOW"""
    if not initialization_complete:
        raise HTTPException(
            status_code=503,
            detail="RAG system is still initializing. Please try again in a few moments."
        )
    
    try:
        from generate_sow import SOWGenerator
        generator = SOWGenerator()
        filename = generator.generate_sow(sow_type)
        logger.info(f"Generated SOW: {filename}")
        return {"message": f"Generated SOW: {filename}"}
    except Exception as e:
        logger.error(f"Error generating SOW: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating SOW: {str(e)}")

async def initialize_rag_system_async():
    """Initialize the RAG system asynchronously"""
    success = initialize_rag_system()
    if not success:
        logger.error("Failed to initialize RAG system. The API will not function correctly.")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*50)
    print("Server Starting...")
    print("="*50)
    print("\nTry accessing the application at:")
    print("1. http://localhost:8000")
    print("2. http://127.0.0.1:8000")
    print("3. http://[your-ip-address]:8000")
    print("\nPress Ctrl+C to stop the server")
    print("\n" + "="*50 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    ) 