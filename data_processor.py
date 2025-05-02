import os
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from datetime import datetime

class DataProcessor:
    def __init__(self):
        self.documents = None
        self.vector_store = None
        self.collection_name = "sow_documents"  # Fixed collection name
        
    def load_documents(self):
        """Load and process SOW documents"""
        print("Loading documents...")
        
        # Get all .docx files excluding temporary files
        doc_files = []
        for root, _, files in os.walk('documents'):
            for file in files:
                if file.endswith('.docx') and not file.startswith('~$'):
                    doc_files.append(os.path.join(root, file))
        
        if not doc_files:
            print("No valid .docx files found in documents directory")
            return 0
            
        print(f"Found {len(doc_files)} valid .docx files")
        
        # Load each document
        self.documents = []
        for file_path in doc_files:
            try:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                # Add metadata to each document
                for doc in docs:
                    # Ensure the document has content
                    if not doc.page_content.strip():
                        print(f"Warning: Empty document content in {file_path}")
                        continue
                        
                    # Extract SOW type from the title
                    sow_type = "Unknown"
                    if "Statement of Work - " in doc.page_content:
                        sow_type = doc.page_content.split("Statement of Work - ")[1].split("\n")[0]
                    
                    # Extract key sections
                    deliverables = []
                    milestones = []
                    terms = []
                    
                    current_section = None
                    for line in doc.page_content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                            
                        if "Deliverables" in line:
                            current_section = "deliverables"
                        elif "Project Milestones" in line:
                            current_section = "milestones"
                        elif "Terms and Conditions" in line:
                            current_section = "terms"
                        elif current_section and line.startswith('•'):
                            if current_section == "deliverables":
                                deliverables.append(line[1:].strip())
                            elif current_section == "milestones":
                                milestones.append(line[1:].strip())
                            elif current_section == "terms":
                                terms.append(line[1:].strip())
                    
                    doc.metadata.update({
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "timestamp": datetime.now().isoformat(),
                        "sow_type": sow_type,
                        "deliverables": "|".join(deliverables),
                        "milestones": "|".join(milestones),
                        "terms": "|".join(terms)
                    })
                    self.documents.append(doc)
                print(f"Loaded: {os.path.basename(file_path)} with {len(docs)} chunks")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        print(f"Successfully loaded {len(self.documents)} document chunks")
        return len(self.documents)
        
    def split_documents(self):
        """Split documents into chunks with optimized parameters"""
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,  # Smaller chunks for better precision
            chunk_overlap=150,  # More overlap for better context
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            is_separator_regex=False
        )
        self.documents = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.documents)} chunks")
        
        # Print sample of first few chunks to verify content
        print("\nSample of first few chunks:")
        for i, doc in enumerate(self.documents[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
        
        return len(self.documents)
        
    def create_embeddings(self):
        """Create embeddings and vector store with improved configuration"""
        print("Creating embeddings...")
        
        # Initialize embeddings with better model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",  # Use the same model as before
            model_kwargs={'device': 'cpu'}
        )
        
        # Check if vector store exists and has documents
        if os.path.exists("./chroma_db"):
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings,
                collection_name=self.collection_name
            )
            
            # Verify if the vector store has documents
            doc_count = self.vector_store._collection.count()
            if doc_count > 0:
                print(f"Found existing vector store with {doc_count} documents")
                return self.vector_store
            else:
                print("Existing vector store is empty, creating new one...")
        
        # Create new vector store
        print("Creating new vector store...")
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=embeddings,
            persist_directory="./chroma_db",
            collection_name=self.collection_name,
            collection_metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,  # Reduced for faster construction
                "hnsw:search_ef": 200  # Reduced for faster search
            }
        )
        
        # Persist the vector store
        self.vector_store.persist()
        print(f"Vector store created and persisted with collection: {self.collection_name}")
        
        # Verify the vector store
        print("\nVerifying vector store:")
        print(f"Number of documents in store: {self.vector_store._collection.count()}")
        
        # Skip verification tests for faster initialization
        return self.vector_store 