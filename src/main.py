"""
src/main.py

FastAPI application entry point for the RAG API.

The lifespan context manager builds the LangGraph once at startup and stores
it in app.state so every request shares the same compiled graph + BM25 index
(no file I/O per query).

Endpoint
--------
POST /query
    Body:  {"question": "..."}
    Returns: {"question": "...", "answer": "...", "sources": [...], "status": "..."}
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.graph.graph_build import build_graph
from src.schema import QueryRequest, QueryResponse, SourceChunk

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: build and compile the LangGraph (loads BM25 corpus into memory).
    Shutdown: nothing to clean up (no persistent connections).
    """
    logger.info("=== RAG API startup ===")
    cfg = get_settings()
    logger.info("Config: LLM=%s, embedding=%s, index=%s", cfg.llm_model, cfg.embedding_model, cfg.pinecone_index_name)

    # Build graph (loads BM25 corpus, compiles graph)
    app.state.graph = build_graph()
    logger.info("Graph ready, API accepting requests")

    yield

    logger.info("=== RAG API shutdown ===")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG API — Agentic AI Ebook",
    description="Hybrid retrieval (dense + sparse) with LangGraph orchestration",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check."""
    return {"status": "ok", "message": "RAG API is running"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """
    Process a user question through the RAG pipeline.

    Flow:
      validate → retrieve (dense + sparse) → RRF fuse → relevance check
        → retry (if low relevance on first attempt) → generate answer | fallback
    """
    logger.info("POST /query: %s (thread_id=%s)", req.question, req.thread_id)

    # Invoke graph with thread_id for conversational memory
    graph = app.state.graph
    result = graph.invoke(
        {"question": req.question},
        config={"configurable": {"thread_id": req.thread_id}},
    )

    # Extract sources from fused_chunks (empty if invalid_query or fallback)
    sources: list[SourceChunk] = []
    for chunk in result.get("fused_chunks", []):
        sources.append(
            SourceChunk(
                chunk_id=chunk["chunk_id"],
                page_num=chunk["page_num"],
                label=chunk["label"],
                text=chunk["text"],
            )
        )

    response = QueryResponse(
        question=req.question,
        answer=result.get("answer", ""),
        sources=sources,
        status=result.get("status", "answered"),
    )

    logger.info("POST /query complete: status=%s, sources=%d", response.status, len(response.sources))
    return response
