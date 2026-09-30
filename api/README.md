# Hybrid RAG API - FastAPI Backend

Production-grade FastAPI backend for the Hybrid RAG chatbot system.

## Architecture

The API provides a complete Hybrid RAG pipeline with:
- **Dense Retrieval**: Pinecone vector search with Gemini embeddings
- **Sparse Retrieval**: BM25 keyword search  
- **RRF Fusion**: Reciprocal Rank Fusion to combine results
- **LangGraph Orchestration**: Multi-step pipeline with retry logic
- **Conversational Memory**: Thread-based memory using LangGraph checkpointing

## Endpoints

### `GET /health`
Health check endpoint returning:
- Service status
- Configuration (LLM model, embedding model, index names, top-k values)
- Graph initialization status

### `POST /query`
Main query endpoint for the RAG pipeline.

**Request:**
```json
{
  "question": "What is agentic AI?",
  "thread_id": "user-123"
}
```

**Response:**
```json
{
  "question": "What is agentic AI?",
  "answer": "Agentic AI refers to...",
  "sources": [
    {
      "chunk_id": "Ebook-Agentic-AI_p3_c1",
      "page_num": 3,
      "label": "core_concept",
      "text": "Agentic AI is a paradigm..."
    }
  ],
  "status": "answered",
  "metadata": {
    "retrieval_attempt": 0,
    "dense_count": 10,
    "sparse_count": 10,
    "fused_count": 5,
    "top_rrf_score": 0.0456,
    "dense_scores": [...],
    "sparse_scores": [...],
    "rrf_scores": [...]
  }
}
```

**Status values:**
- `answered`: Normal answer returned
- `low_relevance`: Retrieved chunks scored below threshold, fallback answer
- `invalid_query`: Question is off-topic or too short

**Metadata includes:**
- `retrieval_attempt`: 0 (first attempt) or 1 (retry)
- Counts for dense, sparse, and fused results
- Top RRF score
- Individual scores from dense retrieval (top 5)
- Individual scores from sparse retrieval (top 5)  
- All RRF fusion scores

## Setup

### Prerequisites
All dependencies should already be installed from `requirements.txt`:
```bash
pip install fastapi uvicorn
```

### Environment Variables
The API uses `.env` at the project root (already configured in `src/config.py`):
```env
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_REGION=your_region
PINECONE_INDEX_NAME=your_index_name
```

## Running the API

### Development Mode (with auto-reload)
```bash
cd D:\RAG_Project
uvicorn api.main:app --reload --port 8000
```

### Production Mode
```bash
cd D:\RAG_Project
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The server will start at: `http://localhost:8000`

### Startup Process
On startup, the API will:
1. Load configuration from `.env`
2. Load BM25 corpus from `data/processed/bm25_corpus.jsonl`
3. Build BM25 index (121 documents)
4. Initialize Pinecone client
5. Initialize Groq LLM client
6. Compile LangGraph pipeline

This takes ~5-10 seconds. Once you see `✓ RAG pipeline ready`, the API is ready for requests.

## Testing the API

### Interactive API Documentation
FastAPI provides automatic interactive docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Using the Test Client
We've provided a comprehensive test client:

```bash
# Make sure the server is running first
uvicorn api.main:app --reload --port 8000

# In another terminal:
python api/test_client.py
```

The test client will:
1. Check API health
2. Run 3 test queries
3. Display full responses with:
   - Generated answers
   - Source chunks
   - Dense/Sparse/RRF scores
   - Retrieval metadata

### Using curl

**Health check:**
```bash
curl http://localhost:8000/health
```

**Query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is agentic AI?",
    "thread_id": "test-session"
  }'
```

### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is agentic AI?",
        "thread_id": "user-123"
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Status: {result['status']}")
print(f"Sources: {len(result['sources'])} chunks")
print(f"Top RRF Score: {result['metadata']['top_rrf_score']:.4f}")
```

## Conversational Memory

The API maintains conversation history per `thread_id`:

```python
# First question
requests.post("http://localhost:8000/query", json={
    "question": "What is agentic AI?",
    "thread_id": "user-123"
})

# Follow-up (same thread_id)
requests.post("http://localhost:8000/query", json={
    "question": "How does it work?",
    "thread_id": "user-123"
})
```

Each unique `thread_id` maintains its own conversation context via LangGraph's `MemorySaver`.

## CORS Configuration

The API is currently configured to allow all origins for development:
```python
allow_origins=["*"]
```

For production, update `api/main.py` to specify allowed origins:
```python
allow_origins=[
    "http://localhost:3000",
    "https://yourdomain.com"
]
```

## Performance Notes

- **First request**: ~2-5 seconds (includes Pinecone query, BM25 search, RRF fusion, LLM generation)
- **Subsequent requests**: ~2-4 seconds
- **Memory**: ~500MB RAM (BM25 index + LangGraph state)
- **Concurrent requests**: Single worker handles ~10 concurrent users

For production with more users, use multiple workers:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Troubleshooting

### "RAG pipeline not initialized"
The graph failed to build at startup. Check:
1. `.env` file exists at project root with all required keys
2. `data/processed/bm25_corpus.jsonl` exists
3. Pinecone index exists and is accessible
4. API keys are valid

### "Validation Error" on /query
Check request format:
- `question` is required (3-500 characters)
- `thread_id` is optional (defaults to "default")

### Connection errors
Make sure the server is running on port 8000:
```bash
uvicorn api.main:app --reload --port 8000
```

## Next Steps

### Frontend Integration
The API is ready for a frontend chatbot UI. Example React/Next.js integration:

```javascript
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: userInput,
    thread_id: sessionId
  })
});

const data = await response.json();
// Display data.answer, data.sources, data.metadata
```

### Deployment
For production deployment:
1. Set environment-specific origins in CORS
2. Use a process manager (PM2, systemd)
3. Set up reverse proxy (nginx)
4. Configure SSL/HTTPS
5. Use multiple workers for scaling
6. Add request rate limiting
7. Set up logging and monitoring

## API Response Example

```json
{
  "question": "What is agentic AI?",
  "answer": "Agentic AI represents a paradigm shift in artificial intelligence...",
  "sources": [
    {
      "chunk_id": "Ebook-Agentic-AI_p3_c1",
      "page_num": 3,
      "label": "core_concept",
      "text": "Agentic AI is a class of AI systems that can perceive their environment..."
    }
  ],
  "status": "answered",
  "metadata": {
    "retrieval_attempt": 0,
    "dense_count": 10,
    "sparse_count": 10,
    "fused_count": 5,
    "top_rrf_score": 0.0456,
    "dense_scores": [
      {"chunk_id": "...", "score": 0.8234, "page_num": 3}
    ],
    "sparse_scores": [
      {"chunk_id": "...", "score": 12.4567, "page_num": 3}
    ],
    "rrf_scores": [
      {"chunk_id": "...", "rrf_score": 0.0456, "page_num": 3}
    ]
  }
}
```

## Files

- `api/main.py` - FastAPI application with health and query endpoints
- `api/test_client.py` - Comprehensive test client
- `api/README.md` - This file
