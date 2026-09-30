"""
src/retrieval/fusion.py

Reciprocal Rank Fusion (RRF) over dense + sparse candidate lists.

rrf_fuse() merges the two ranked lists into a single ranked list and
returns the top final_top_k chunks for the LLM prompt.

Formula (per chunk, per list):
    score += 1 / (k + rank)          (rank is 1-indexed)
where k=60 is the RRF constant (default in literature).
"""

from src.config import get_settings


def rrf_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    rrf_k: int | None = None,
    final_top_k: int | None = None,
) -> list[dict]:
    """
    Merge dense and sparse candidate lists with Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_results  : output of query_dense()  — ordered best-first
    sparse_results : output of query_sparse() — ordered best-first
    rrf_k          : RRF constant (defaults to settings.rrf_k = 60)
    final_top_k    : how many chunks to return (defaults to settings.final_top_k = 5)

    Returns
    -------
    list of dicts ordered by descending RRF score, each with keys:
        chunk_id, rrf_score, text, source, page_num, label
    """
    cfg = get_settings()
    k   = rrf_k      if rrf_k      is not None else cfg.rrf_k
    top = final_top_k if final_top_k is not None else cfg.final_top_k

    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id → chunk metadata (last write wins; both lists carry the same data)
    chunk_meta: dict[str, dict]  = {}

    for ranked_list in (dense_results, sparse_results):
        for rank, chunk in enumerate(ranked_list, start=1):      # rank is 1-indexed
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_meta[cid] = chunk   # store / overwrite metadata

    # Sort by descending RRF score and take top final_top_k
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    fused: list[dict] = []
    for cid in sorted_ids[:top]:
        meta = chunk_meta[cid]
        fused.append(
            {
                "chunk_id":  cid,
                "rrf_score": rrf_scores[cid],
                "text":      meta["text"],
                "source":    meta["source"],
                "page_num":  meta["page_num"],
                "label":     meta["label"],
            }
        )

    return fused
