"""
src/config.py

Single source of truth for all runtime settings.
Loaded once at startup via get_settings() — every other module imports from here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API keys ───────────────────────────────────────────────────────────────
    google_api_key:   str = Field(..., description="Gemini API key")
    groq_api_key:     str = Field(..., description="Groq API key")
    pinecone_api_key: str = Field(..., description="Pinecone API key")

    # ── Pinecone ───────────────────────────────────────────────────────────────
    pinecone_region:     str = Field(..., description="Pinecone index region")
    pinecone_index_name: str = Field(..., description="Pinecone index name")

    # ── Embedding ──────────────────────────────────────────────────────────────
    embedding_model: str = "gemini-embedding-001"
    embedding_dim:   int = 768

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_model:       str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.0

    # ── Retrieval ──────────────────────────────────────────────────────────────
    dense_top_k:    int   = 10    # candidates fetched from Pinecone before RRF
    bm25_top_k:     int   = 10    # candidates fetched from BM25 before RRF
    rrf_k:          int   = 60    # RRF constant
    final_top_k:    int   = 5     # chunks passed to LLM after RRF
    relevance_threshold: float = 0.01  # min RRF score to be considered relevant

    # ── Paths ──────────────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).parent.parent
    chunks_path:  Path = Path(__file__).parent.parent / "data" / "processed" / "chunks.jsonl"
    bm25_path:    Path = Path(__file__).parent.parent / "data" / "processed" / "bm25_corpus.jsonl"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
