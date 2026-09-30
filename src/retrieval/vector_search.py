"""
src/retrieval/vector_search.py

Dense retrieval via Pinecone + Gemini embeddings.

query_dense() embeds the question with gemini-embedding-001 (task_type
RETRIEVAL_QUERY) and returns the top-k most similar chunks from the index.
"""

from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone

from src.config import get_settings

# ── module-level singletons initialised on first use ─────────────────────────
_gemini_client: genai.Client | None = None
_pinecone_index = None


def _get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        cfg = get_settings()
        _gemini_client = genai.Client(api_key=cfg.google_api_key)
    return _gemini_client


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        cfg = get_settings()
        pc = Pinecone(api_key=cfg.pinecone_api_key)
        _pinecone_index = pc.Index(cfg.pinecone_index_name)
    return _pinecone_index


def _embed_query(text: str) -> list[float]:
    """Embed a user question with task_type=RETRIEVAL_QUERY."""
    cfg = get_settings()
    client = _get_gemini_client()
    response = client.models.embed_content(
        model=cfg.embedding_model,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",          # different from ingestion task
            output_dimensionality=cfg.embedding_dim,
        ),
    )
    return response.embeddings[0].values


def query_dense(question: str, top_k: int | None = None) -> list[dict]:
    """
    Embed *question* and query Pinecone for the nearest neighbours.

    Parameters
    ----------
    question : natural-language user question
    top_k    : number of candidates to return (defaults to settings.dense_top_k)

    Returns
    -------
    list of dicts, each with keys:
        chunk_id, score, text, source, page_num, label
    """
    cfg = get_settings()
    k   = top_k if top_k is not None else cfg.dense_top_k

    vector   = _embed_query(question)
    index    = _get_pinecone_index()
    response = index.query(
        vector=vector,
        top_k=k,
        include_metadata=True,
    )

    results: list[dict] = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append(
            {
                "chunk_id": match.id,
                "score":    match.score,        # cosine similarity in [0, 1]
                "text":     meta.get("text", ""),
                "source":   meta.get("source", ""),
                "page_num": int(meta.get("page_num", 0)),
                "label":    meta.get("label", "core_concept"),
            }
        )

    return results
