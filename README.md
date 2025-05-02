# SOW RAG System

A Retrieval-Augmented Generation (RAG) system specialized in analyzing and answering questions about Statement of Work (SOW) documents.

## Project Structure

```
custom-rag/
├── documents/           # SOW documents in various formats
├── metrics/            # Evaluation metrics and results
├── static/             # Frontend static files
├── templates/          # System templates
│   ├── sow_template.json       # SOW document structure template
│   ├── answer_template.json    # Answer formatting template
│   └── evaluation_template.json # Evaluation metrics template
├── api.py              # FastAPI server implementation
├── rag_system.py       # Main RAG system implementation
├── data_processor.py   # Document processing utilities
├── prompts.py          # Prompt templates
├── metrics_handler.py  # Metrics tracking and evaluation
├── answer_generator.py # Answer generation logic
├── answer_verifier.py  # Answer verification system
├── ground_truth.py     # Ground truth for evaluation
├── evaluate_rag.py     # Evaluation script
└── requirements.txt    # Project dependencies
```

## Template System

The system uses three main templates to ensure consistency and quality:

### 1. SOW Template (`templates/sow_template.json`)
- Defines the structure of SOW documents
- Contains placeholders for project-specific information
- Standardizes document format across the system
- Includes sections for:
  - Project information
  - Objectives
  - Scope
  - Timeline
  - Terms and conditions

### 2. Answer Template (`templates/answer_template.json`)
- Structures how answers are formatted
- Defines required components:
  - Direct answer
  - Supporting details
  - Source references
  - Additional context
- Includes metrics tracking
- Specifies formatting requirements

### 3. Evaluation Template (`templates/evaluation_template.json`)
- Defines evaluation criteria
- Includes metrics for:
  - Quality (faithfulness, completeness, relevance)
  - Performance (response time, query time, latency)
  - Source usage (number of sources, relevance scores)

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

3. Place SOW documents in the `documents/` directory

## Running the System

1. Start the RAG system:
```bash
python rag_system.py
```

2. Access the web interface at `http://localhost:8000`

3. Run evaluation:
```bash
python evaluate_rag.py
```

## Features

- Document Processing
  - Supports multiple document formats (JSON, DOCX, XLSX)
  - Automatic text extraction and chunking
  - Vector embeddings for semantic search

- Question Answering
  - Context-aware responses
  - Source citation
  - Confidence scoring
  - Performance metrics

- Evaluation
  - Ground truth comparison
  - Quality metrics
  - Performance tracking
  - Source relevance scoring

## API Endpoints

- `GET /`: Web interface
- `GET /questions`: Get example questions
- `POST /query`: Submit a question
- `GET /metrics`: Get system metrics

## Evaluation Metrics

The system tracks several metrics:

### Quality Metrics
- Faithfulness Score (threshold: 0.7)
- Completeness (threshold: 0.8)
- Relevance (threshold: 0.6)

### Performance Metrics
- Response Time (threshold: 3.0s)
- Query Time (threshold: 2.0s)
- Latency (threshold: 1.0s)

### Source Metrics
- Number of Sources (min: 1, max: 5)
- Source Relevance (threshold: 0.5)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that processes and manages 20 Statement of Work (SOW) documents. The system enables efficient document retrieval and question-answering capabilities.

## Features
- Document processing and chunking
- Vector embeddings using HuggingFace's sentence transformers
- ChromaDB vector store for efficient similarity search and persistence
- OpenAI-powered question answering
- Support for DOCX document format

## Requirements

### 1. Python Environment
- Python 3.8 or higher
- pip package manager

### 2. Dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```

### 3. Credentials and Costs

#### Required Credentials
1. **OpenAI API Key (Chargeable)**
   - Required for question-answering functionality
   - Cost: Pay-per-use based on token usage
   - Pricing: 
     - GPT-3.5-turbo: $0.0015 per 1K tokens input, $0.002 per 1K tokens output
   - How to get: Sign up at https://platform.openai.com/
   - Note: You can set usage limits in your OpenAI account

2. **HuggingFace (Free)**
   - Used for sentence-transformers model
   - No API key required
   - Model: `sentence-transformers/all-mpnet-base-v2`
   - Runs locally on your machine

3. **Local System Requirements (Free)**
   - Python environment
   - Disk space requirements:
     - Document storage: ~1-2MB per SOW
     - Vector embeddings: ~100-200MB
     - ChromaDB storage: ~50-100MB
   - RAM: 8GB+ recommended
   - No additional costs

### 4. Environment Variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure
```
.
├── documents/              # Directory containing SOW documents
├── output/                # Directory containing verification metrics
├── generate_sow.py        # Script to generate dummy SOW documents
├── rag_system.py          # Main RAG system implementation
├── verify_answers.py      # Script for answer verification and metrics
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## Usage

### 1. Generate SOW Documents
```bash
python generate_sow.py
```
This will create 20 dummy SOW documents in the `documents` directory.

### 2. Run the RAG System
```bash
python rag_system.py
```
This will:
- Load and process the SOW documents
- Create embeddings
- Set up the vector store
- Run example queries

### 3. Custom Queries
You can modify the `main()` function in `rag_system.py` to ask different questions or create a new script that uses the `SOWRAGSystem` class.

## Example Queries
The system can answer questions about:
- Project types
- Project durations
- Deliverables
- Terms and conditions
- Budget information
- Client details

## Technical Details

### Document Processing
- Uses `docx2txt` for DOCX file processing
- Implements chunking with overlap for better context
- Preserves document structure and formatting

### Embeddings
- Uses HuggingFace's `sentence-transformers/all-mpnet-base-v2` model
- Creates vector representations of document chunks
- Enables semantic search capabilities

### Vector Store
- Implements ChromaDB for efficient similarity search and persistence
- Stores document embeddings and metadata
- Enables fast retrieval of relevant content
- Provides persistent storage of embeddings
- Supports metadata filtering and hybrid search

### Question Answering
- Uses OpenAI's language model for generation
- Implements retrieval-augmented generation
- Provides context-aware answers

## Cost Management

### OpenAI Usage Optimization
1. **Token Usage**
   - Monitor token usage in OpenAI dashboard
   - Set up usage alerts
   - Implement token limits

2. **Query Optimization**
   - Keep questions concise
   - Use appropriate chunk sizes
   - Implement caching for frequent queries

3. **Free Alternatives**
   - For embedding: Continue using free HuggingFace models
   - For vector store: ChromaDB is free and provides persistence
   - For document processing: All local tools are free

## Troubleshooting

### Common Issues
1. **Module Not Found Errors**
   - Ensure all dependencies are installed correctly
   - Check Python version compatibility

2. **API Key Issues**
   - Verify OpenAI API key in `.env` file
   - Ensure the key has sufficient credits
   - Check OpenAI API status at https://status.openai.com/

3. **Document Processing Errors**
   - Check document format (must be DOCX)
   - Verify document permissions

### Performance Optimization
- Adjust chunk size and overlap in `rag_system.py`
- Modify embedding model based on requirements
- Fine-tune ChromaDB retrieval parameters
- Configure similarity search thresholds

## Future Enhancements
1. Support for additional document formats
2. Web interface for querying
3. Document similarity search
4. Export functionality for processed documents
5. Batch processing capabilities

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License.

## Answer Verification and Accuracy

### 1. Verification Methods

#### Automated Verification
- Uses test questions with known answers
- Compares model outputs with expected answers
- Calculates similarity scores
- Provides quantitative accuracy metrics

#### Manual Verification
- Review source documents
- Check retrieved context
- Verify against domain knowledge
- Cross-reference with multiple sources

### 2. Improving Accuracy

#### Document Processing
- Optimize chunk size (recommended: 1000-2000 characters)
- Adjust chunk overlap (recommended: 200-400 characters)
- Implement smart chunking based on document structure

#### Retrieval Optimization
- Fine-tune ChromaDB search parameters
- Implement hybrid search (keyword + semantic)
- Add metadata filtering
- Use relevance scoring
- Configure similarity thresholds

#### Prompt Engineering
- Make questions specific and clear
- Add context to questions
- Use few-shot examples
- Implement chain-of-thought prompting

#### Post-processing
- Add answer validation
- Implement fact-checking
- Use confidence scoring
- Add source attribution

### 3. Monitoring and Evaluation

#### Metrics to Track
- Answer similarity scores
- Retrieval precision and recall
- Response time
- Token usage
- User feedback

#### Continuous Improvement
- Regular accuracy testing
- Update test questions
- Monitor error patterns
- Implement feedback loops

## Answer Verification and Metrics

### 1. Metrics Storage
The system automatically stores verification metrics in JSON format in the `output` directory. Each verification run creates a timestamped file (e.g., `verification_metrics_YYYYMMDD_HHMMSS.json`) containing:

#### Overall Metrics
- Similarity scores (mean and standard deviation)
- BLEU scores
- METEOR scores
- Precision, Recall, and F1 scores
- Key points coverage

#### Per-Question Analysis
For each test question, the metrics file includes:
- Question and expected answer
- Model's actual response
- Detailed similarity metrics
- Missing and extra words analysis
- Source document tracking
- Key points coverage

### 2. Running Verification
```bash
python verify_answers.py
```
This will:
- Run test questions through the RAG system
- Compare responses with expected answers
- Generate comprehensive metrics
- Save results to the output directory

### 3. Interpreting Metrics

#### Similarity Scores
- Range: 0.0 to 1.0
- Higher is better
- Measures semantic similarity between expected and actual answers

#### BLEU and METEOR
- Industry-standard NLP metrics
- Evaluate text generation quality
- Consider word order and synonyms

#### Precision and Recall
- Precision: Accuracy of provided information
- Recall: Completeness of provided information
- F1: Harmonic mean of precision and recall

#### Key Points Coverage
- Percentage of essential information captured
- Identifies missing critical details
- Helps in system optimization

### 4. Using Metrics for Improvement

#### System Optimization
- Adjust retriever parameters based on metrics
- Fine-tune chunk sizes and overlap
- Modify prompt templates
- Update test cases

#### Quality Monitoring
- Track metrics over time
- Identify performance trends
- Set quality thresholds
- Implement automated alerts 