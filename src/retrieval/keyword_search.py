"""
src/retrieval/keyword_search.py

Sparse retrieval via BM25 (rank-bm25).

The BM25 index is built from bm25_corpus.jsonl at API startup and stored in
the FastAPI lifespan state so it lives for the lifetime of the process —
no file I/O per query.

query_sparse() accepts the pre-built index + corpus records and returns the
top-k chunks ranked by BM25 score.
"""

from rank_bm25 import BM25Okapi

from src.config import get_settings


def build_bm25_index(records: list[dict]) -> BM25Okapi:
    """
    Build a BM25Okapi index from the pre-tokenised corpus records.

    Parameters
    ----------
    records : list of dicts loaded from bm25_corpus.jsonl
              each record must have a "tokens" key (list[str])

    Returns
    -------
    BM25Okapi  ready for get_scores() calls
    """
    tokenised = [r["tokens"] for r in records]
    return BM25Okapi(tokenised)


def query_sparse(
    question: str,
    bm25_index: BM25Okapi,
    records: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Score every corpus document against *question* and return top-k.

    Parameters
    ----------
    question   : natural-language user question (tokenised with .lower().split())
    bm25_index : BM25Okapi built by build_bm25_index()
    records    : the same list[dict] passed to build_bm25_index()
    top_k      : number of candidates to return (defaults to settings.bm25_top_k)

    Returns
    -------
    list of dicts, each with keys:
        chunk_id, score, text, source, page_num, label
    """
    cfg = get_settings()
    k   = top_k if top_k is not None else cfg.bm25_top_k

    tokens = question.lower().split()
    scores = bm25_index.get_scores(tokens)

    # Pair each score with its record index, sort descending, take top-k
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results: list[dict] = []
    for idx in ranked:
        r = records[idx]
        results.append(
            {
                "chunk_id": r["chunk_id"],
                "score":    float(scores[idx]),   # raw BM25 score
                "text":     r["text"],
                "source":   r["source"],
                "page_num": r["page_num"],
                "label":    r["label"],
            }
        )

    return results
