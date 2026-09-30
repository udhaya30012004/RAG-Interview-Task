"""
src/schema.py

Pydantic models for FastAPI request and response bodies.
Kept separate from config so the API contract is easy to find and change.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body of POST /query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's question about the Agentic AI ebook.",
        examples=["What is agentic AI?"],
    )
    thread_id: str = Field(
        default="default",
        description="Conversation thread ID for memory (default: 'default'). Use a unique ID per user session.",
        examples=["user-123", "session-abc"],
    )


class SourceChunk(BaseModel):
    """A single retrieved chunk returned alongside the answer."""

    chunk_id: str = Field(..., description="Unique chunk identifier, e.g. Ebook-Agentic-AI_p3_c1")
    page_num: int = Field(..., description="Source page number (1-indexed)")
    label:    str = Field(..., description="Structural label: introduction | core_concept | case_study | conclusion")
    text:     str = Field(..., description="The chunk text that was used to generate the answer")


class QueryResponse(BaseModel):
    """Body returned by POST /query."""

    question: str          = Field(..., description="The original question echoed back")
    answer:   str          = Field(..., description="LLM-generated answer grounded in retrieved chunks")
    sources:  list[SourceChunk] = Field(..., description="Top chunks used to generate the answer")
    status:   str          = Field(
        ...,
        description=(
            "Pipeline outcome: "
            "'answered' — normal answer returned; "
            "'low_relevance' — retrieved chunks scored below threshold, fallback answer; "
            "'invalid_query' — question is off-topic or too short"
        ),
    )
    metadata: dict = Field(
        default_factory=dict,
        description=(
            "Retrieval metadata including: "
            "retrieval_attempt (0 or 1), "
            "dense_count, sparse_count, fused_count, "
            "top_rrf_score, "
            "dense_scores (top 5 with chunk_id, score, page_num), "
            "sparse_scores (top 5 with chunk_id, score, page_num), "
            "rrf_scores (all fused chunks with chunk_id, rrf_score, page_num)"
        ),
    )
