"""
api/main.py

FastAPI backend for the Hybrid RAG chatbot.

Endpoints:
  GET  /health       - Health check endpoint
  POST /query        - Main query endpoint (returns answer + chunks + scores)

Run from project root:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import QueryRequest, QueryResponse, SourceChunk
from src.config import get_settings
from src.graph.graph_build import build_graph

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────────────────
graph = None


# ── Lifespan context manager ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: Load BM25 index and compile LangGraph once.
    Shutdown: Clean up resources.
    """
    global graph

    logger.info("=" * 80)
    logger.info("  FASTAPI STARTUP - Loading RAG Pipeline")
    logger.info("=" * 80)

    cfg = get_settings()
    logger.info(f"LLM Model       : {cfg.llm_model}")
    logger.info(f"Embedding Model : {cfg.embedding_model}")
    logger.info(f"Pinecone Index  : {cfg.pinecone_index_name}")
    logger.info(f"Dense top-k     : {cfg.dense_top_k}")
    logger.info(f"BM25 top-k      : {cfg.bm25_top_k}")
    logger.info(f"RRF k           : {cfg.rrf_k}")
    logger.info(f"Final top-k     : {cfg.final_top_k}")

    # Build graph (loads BM25 corpus, compiles LangGraph)
    logger.info("Building LangGraph RAG pipeline...")
    graph = build_graph()
    logger.info("✓ RAG pipeline ready")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("Shutting down FastAPI app")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hybrid RAG API",
    description="Production-grade Hybrid RAG system with Dense (Pinecone) + Sparse (BM25) + RRF fusion",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns system status and configuration.
    """
    cfg = get_settings()

    return {
        "status": "healthy",
        "service": "Hybrid RAG API",
        "version": "1.0.0",
        "graph_loaded": graph is not None,
        "config": {
            "llm_model": cfg.llm_model,
            "embedding_model": cfg.embedding_model,
            "pinecone_index": cfg.pinecone_index_name,
            "dense_top_k": cfg.dense_top_k,
            "bm25_top_k": cfg.bm25_top_k,
            "rrf_k": cfg.rrf_k,
            "final_top_k": cfg.final_top_k,
            "relevance_threshold": cfg.relevance_threshold,
        }
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint.

    Processes a question through the hybrid RAG pipeline and returns:
      - Generated answer
      - Source chunks (with text, page numbers, labels)
      - Retrieval scores (dense, sparse, RRF)
      - Pipeline status
      - Retrieval metadata

    Request body:
      - question: str (required, 3-500 chars)
      - thread_id: str (optional, default="default")

    Response:
      - question: str
      - answer: str
      - sources: list[SourceChunk]
      - status: str
      - metadata: dict (retrieval scores, attempt count, etc.)
    """
    if graph is None:
        logger.error("Graph not initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Check server logs."
        )

    logger.info(f"Received query: '{request.question}' (thread_id={request.thread_id})")

    try:
        # Run the graph
        result = graph.invoke(
            {"question": request.question},
            config={"configurable": {"thread_id": request.thread_id}},
        )

        # Extract data
        answer = result.get("answer", "")
        status = result.get("status", "unknown")
        fused_chunks = result.get("fused_chunks", [])
        dense_results = result.get("dense_results", [])
        sparse_results = result.get("sparse_results", [])
        retrieval_attempt = result.get("retrieval_attempt", 0)

        # Build source chunks
        sources = [
            SourceChunk(
                chunk_id=chunk["chunk_id"],
                page_num=chunk["page_num"],
                label=chunk["label"],
                text=chunk["text"],
            )
            for chunk in fused_chunks
        ]

        # Build metadata with retrieval scores
        metadata = {
            "retrieval_attempt": retrieval_attempt,
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_chunks),
            "top_rrf_score": fused_chunks[0]["rrf_score"] if fused_chunks else 0.0,
            "dense_scores": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                    "page_num": chunk["page_num"],
                }
                for chunk in dense_results[:5]  # Top 5
            ],
            "sparse_scores": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "score": chunk["score"],
                    "page_num": chunk["page_num"],
                }
                for chunk in sparse_results[:5]  # Top 5
            ],
            "rrf_scores": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "rrf_score": chunk["rrf_score"],
                    "page_num": chunk["page_num"],
                }
                for chunk in fused_chunks
            ],
        }

        logger.info(
            f"Query completed: status={status}, "
            f"answer_len={len(answer)}, "
            f"sources={len(sources)}, "
            f"attempt={retrieval_attempt}"
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            status=status,
            metadata=metadata,
        )

    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (for local testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
