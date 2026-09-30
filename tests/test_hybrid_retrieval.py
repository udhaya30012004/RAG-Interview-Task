"""
tests/test_hybrid_retrieval.py

Comprehensive Hybrid Retrieval Evaluation Suite

Tests the complete hybrid retrieval pipeline:
  1. Dense retrieval (Pinecone vector search with Gemini embeddings)
  2. Sparse retrieval (BM25 keyword search)
  3. RRF fusion (Reciprocal Rank Fusion)

Metrics computed:
  - Precision@5, Recall@5, F1@5 (per method and fused)
  - MRR (Mean Reciprocal Rank)
  - NDCG@5 (Normalized Discounted Cumulative Gain)
  - Retrieval overlap (how much dense and sparse agree)
  - RRF lift (improvement from fusion vs individual methods)

Output: d:/RAG_Project/results/hybrid_retrieval_quality.txt

Run from project root:
    python -m tests.test_hybrid_retrieval
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.retrieval.vector_search import query_dense
from src.retrieval.keyword_search import query_sparse, build_bm25_index
from src.retrieval.fusion import rrf_fuse

logging.basicConfig(level=logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD TEST CASES & BM25 INDEX
# ══════════════════════════════════════════════════════════════════════════════

def load_test_cases(test_data_path: Path) -> List[Dict]:
    """Load test questions from test_data.txt."""
    test_cases = []
    with test_data_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) != 3:
                print(f"  [WARN] Skipping malformed line {line_num}")
                continue
            difficulty, question, keywords_str = parts
            keywords = [kw.strip() for kw in keywords_str.split(",")]
            test_cases.append({
                "id": len(test_cases) + 1,
                "difficulty": difficulty.strip(),
                "question": question.strip(),
                "relevant_keywords": keywords,
            })
    return test_cases


def load_bm25_index(bm25_path: Path):
    """Load BM25 corpus and build index."""
    records = []
    with bm25_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    index = build_bm25_index(records)
    return index, records


# ══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_relevance_score(chunk_text: str, keywords: List[str]) -> float:
    """Score chunk relevance based on keyword presence."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords) if keywords else 0.0


def label_relevant_chunks(chunks: List[Dict], keywords: List[str], threshold: float = 0.15) -> Set[str]:
    """Label which chunks are relevant based on keyword overlap."""
    relevant = set()
    for chunk in chunks:
        score = compute_relevance_score(chunk["text"], keywords)
        if score >= threshold:
            relevant.add(chunk["chunk_id"])
    return relevant


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@k = (# relevant in top-k) / k"""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_in_top_k / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@k = (# relevant in top-k) / (total # relevant)"""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for cid in top_k if cid in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def f1_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """F1@k = harmonic mean of precision@k and recall@k"""
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """MRR = 1 / (rank of first relevant chunk)"""
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """NDCG@k (Normalized Discounted Cumulative Gain)"""
    if len(relevant_ids) == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 if cid in relevant_ids else 0.0) / math.log2(i + 1)
        for i, cid in enumerate(top_k, start=1)
    )
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_overlap(ids_a: List[str], ids_b: List[str], k: int = 5) -> float:
    """Compute overlap between two ranked lists at top-k."""
    top_a = set(ids_a[:k])
    top_b = set(ids_b[:k])
    if len(top_a) == 0 and len(top_b) == 0:
        return 1.0
    union = top_a | top_b
    intersection = top_a & top_b
    return len(intersection) / len(union) if len(union) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(bm25_index, bm25_records, test_cases: List[Dict], output_path: Path) -> None:
    """Run all test cases and compute aggregate metrics."""

    cfg = get_settings()
    results = []
    aggregate = defaultdict(list)

    print("\n" + "═" * 80)
    print("  HYBRID RETRIEVAL QUALITY EVALUATION")
    print("═" * 80)
    print(f"  Test cases    : {len(test_cases)}")
    print(f"  Dense         : Pinecone + Gemini ({cfg.embedding_model})")
    print(f"  Sparse        : BM25 keyword search")
    print(f"  Fusion        : Reciprocal Rank Fusion (RRF)")
    print(f"  Dense top-k   : {cfg.dense_top_k}")
    print(f"  BM25 top-k    : {cfg.bm25_top_k}")
    print(f"  RRF k         : {cfg.rrf_k}")
    print(f"  Final top-k   : {cfg.final_top_k}")
    print(f"  Eval @ k      : 5")
    print("═" * 80 + "\n")

    for i, test in enumerate(test_cases, start=1):
        q_id = test["id"]
        question = test["question"]
        difficulty = test["difficulty"]
        keywords = test["relevant_keywords"]

        print(f"[{i}/{len(test_cases)}] Q{q_id} ({difficulty.upper()}): {question[:60]}...")

        # ── Run all three retrieval methods ────────────────────────────────────
        dense_results = query_dense(question, top_k=cfg.dense_top_k)
        sparse_results = query_sparse(question, bm25_index, bm25_records, top_k=cfg.bm25_top_k)
        fused_results = rrf_fuse(dense_results, sparse_results, rrf_k=cfg.rrf_k, final_top_k=cfg.final_top_k)

        # Extract IDs
        dense_ids = [c["chunk_id"] for c in dense_results]
        sparse_ids = [c["chunk_id"] for c in sparse_results]
        fused_ids = [c["chunk_id"] for c in fused_results]

        # Label relevance using ALL candidates from dense + sparse
        all_candidates = dense_results + sparse_results
        relevant_ids = label_relevant_chunks(all_candidates, keywords, threshold=0.15)

        # ── Metrics @ k=5 ──────────────────────────────────────────────────────
        k = 5

        # Dense metrics
        dense_p = precision_at_k(dense_ids, relevant_ids, k)
        dense_r = recall_at_k(dense_ids, relevant_ids, k)
        dense_f1 = f1_at_k(dense_ids, relevant_ids, k)
        dense_mrr = mean_reciprocal_rank(dense_ids, relevant_ids)
        dense_ndcg = ndcg_at_k(dense_ids, relevant_ids, k)

        # Sparse metrics
        sparse_p = precision_at_k(sparse_ids, relevant_ids, k)
        sparse_r = recall_at_k(sparse_ids, relevant_ids, k)
        sparse_f1 = f1_at_k(sparse_ids, relevant_ids, k)
        sparse_mrr = mean_reciprocal_rank(sparse_ids, relevant_ids)
        sparse_ndcg = ndcg_at_k(sparse_ids, relevant_ids, k)

        # Fused metrics
        fused_p = precision_at_k(fused_ids, relevant_ids, k)
        fused_r = recall_at_k(fused_ids, relevant_ids, k)
        fused_f1 = f1_at_k(fused_ids, relevant_ids, k)
        fused_mrr = mean_reciprocal_rank(fused_ids, relevant_ids)
        fused_ndcg = ndcg_at_k(fused_ids, relevant_ids, k)

        # Overlap & lift
        overlap = compute_overlap(dense_ids, sparse_ids, k=k)
        f1_lift = fused_f1 - max(dense_f1, sparse_f1)

        # Store
        result_entry = {
            "id": q_id,
            "difficulty": difficulty,
            "question": question,
            "relevant_count": len(relevant_ids),
            # Dense
            "dense_p@5": dense_p,
            "dense_r@5": dense_r,
            "dense_f1@5": dense_f1,
            "dense_mrr": dense_mrr,
            "dense_ndcg@5": dense_ndcg,
            # Sparse
            "sparse_p@5": sparse_p,
            "sparse_r@5": sparse_r,
            "sparse_f1@5": sparse_f1,
            "sparse_mrr": sparse_mrr,
            "sparse_ndcg@5": sparse_ndcg,
            # Fused
            "fused_p@5": fused_p,
            "fused_r@5": fused_r,
            "fused_f1@5": fused_f1,
            "fused_mrr": fused_mrr,
            "fused_ndcg@5": fused_ndcg,
            # Analysis
            "overlap@5": overlap,
            "f1_lift": f1_lift,
            "top_rrf_score": fused_results[0]["rrf_score"] if fused_results else 0.0,
        }
        results.append(result_entry)

        # Aggregate by difficulty
        aggregate[difficulty].append(result_entry)
        aggregate["all"].append(result_entry)

        print(f"      Dense  : P@5={dense_p:.3f} R@5={dense_r:.3f} F1@5={dense_f1:.3f}")
        print(f"      Sparse : P@5={sparse_p:.3f} R@5={sparse_r:.3f} F1@5={sparse_f1:.3f}")
        print(f"      Fused  : P@5={fused_p:.3f} R@5={fused_r:.3f} F1@5={fused_f1:.3f} (lift={f1_lift:+.3f})")

    print("\n" + "─" * 80 + "\n")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def avg_metrics(entries: List[Dict]) -> Dict[str, float]:
        if not entries:
            return {}
        return {
            # Dense
            "dense_p@5": sum(e["dense_p@5"] for e in entries) / len(entries),
            "dense_r@5": sum(e["dense_r@5"] for e in entries) / len(entries),
            "dense_f1@5": sum(e["dense_f1@5"] for e in entries) / len(entries),
            "dense_mrr": sum(e["dense_mrr"] for e in entries) / len(entries),
            "dense_ndcg@5": sum(e["dense_ndcg@5"] for e in entries) / len(entries),
            # Sparse
            "sparse_p@5": sum(e["sparse_p@5"] for e in entries) / len(entries),
            "sparse_r@5": sum(e["sparse_r@5"] for e in entries) / len(entries),
            "sparse_f1@5": sum(e["sparse_f1@5"] for e in entries) / len(entries),
            "sparse_mrr": sum(e["sparse_mrr"] for e in entries) / len(entries),
            "sparse_ndcg@5": sum(e["sparse_ndcg@5"] for e in entries) / len(entries),
            # Fused
            "fused_p@5": sum(e["fused_p@5"] for e in entries) / len(entries),
            "fused_r@5": sum(e["fused_r@5"] for e in entries) / len(entries),
            "fused_f1@5": sum(e["fused_f1@5"] for e in entries) / len(entries),
            "fused_mrr": sum(e["fused_mrr"] for e in entries) / len(entries),
            "fused_ndcg@5": sum(e["fused_ndcg@5"] for e in entries) / len(entries),
            # Analysis
            "overlap@5": sum(e["overlap@5"] for e in entries) / len(entries),
            "f1_lift": sum(e["f1_lift"] for e in entries) / len(entries),
        }

    agg_all = avg_metrics(aggregate["all"])
    agg_easy = avg_metrics(aggregate.get("easy", []))
    agg_medium = avg_metrics(aggregate.get("medium", []))
    agg_hard = avg_metrics(aggregate.get("hard", []))

    # ── Write report ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  HYBRID RETRIEVAL QUALITY EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test cases   : {len(test_cases)} questions\n\n")
        f.write(f"RETRIEVAL METHODS\n")
        f.write(f"  Dense      : Pinecone vector search + Gemini {cfg.embedding_model} (top-{cfg.dense_top_k})\n")
        f.write(f"  Sparse     : BM25 keyword search (top-{cfg.bm25_top_k})\n")
        f.write(f"  Fusion     : Reciprocal Rank Fusion (RRF k={cfg.rrf_k}, final top-{cfg.final_top_k})\n\n")

        f.write("=" * 80 + "\n")
        f.write("  AGGREGATE METRICS (ALL QUESTIONS)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"DENSE RETRIEVAL ({len(aggregate['all'])} questions)\n")
        f.write(f"  Precision@5  : {agg_all['dense_p@5']:.4f}\n")
        f.write(f"  Recall@5     : {agg_all['dense_r@5']:.4f}\n")
        f.write(f"  F1@5         : {agg_all['dense_f1@5']:.4f}\n")
        f.write(f"  MRR          : {agg_all['dense_mrr']:.4f}\n")
        f.write(f"  NDCG@5       : {agg_all['dense_ndcg@5']:.4f}\n\n")

        f.write(f"SPARSE RETRIEVAL (BM25)\n")
        f.write(f"  Precision@5  : {agg_all['sparse_p@5']:.4f}\n")
        f.write(f"  Recall@5     : {agg_all['sparse_r@5']:.4f}\n")
        f.write(f"  F1@5         : {agg_all['sparse_f1@5']:.4f}\n")
        f.write(f"  MRR          : {agg_all['sparse_mrr']:.4f}\n")
        f.write(f"  NDCG@5       : {agg_all['sparse_ndcg@5']:.4f}\n\n")

        f.write(f"FUSED RETRIEVAL (RRF)\n")
        f.write(f"  Precision@5  : {agg_all['fused_p@5']:.4f}\n")
        f.write(f"  Recall@5     : {agg_all['fused_r@5']:.4f}\n")
        f.write(f"  F1@5         : {agg_all['fused_f1@5']:.4f}\n")
        f.write(f"  MRR          : {agg_all['fused_mrr']:.4f}\n")
        f.write(f"  NDCG@5       : {agg_all['fused_ndcg@5']:.4f}\n\n")

        f.write(f"ANALYSIS\n")
        f.write(f"  Avg overlap@5 (Dense vs Sparse) : {agg_all['overlap@5']:.4f}\n")
        f.write(f"  Avg F1 lift from fusion          : {agg_all['f1_lift']:+.4f}\n\n")

        # Per-difficulty breakdown
        f.write("=" * 80 + "\n")
        f.write("  BREAKDOWN BY DIFFICULTY\n")
        f.write("=" * 80 + "\n\n")

        for diff_name, diff_agg in [("EASY", agg_easy), ("MEDIUM", agg_medium), ("HARD", agg_hard)]:
            if not diff_agg:
                continue
            f.write(f"{diff_name} ({len(aggregate.get(diff_name.lower(), []))} questions)\n")
            f.write(f"  Dense F1@5   : {diff_agg['dense_f1@5']:.4f}\n")
            f.write(f"  Sparse F1@5  : {diff_agg['sparse_f1@5']:.4f}\n")
            f.write(f"  Fused F1@5   : {diff_agg['fused_f1@5']:.4f}\n")
            f.write(f"  F1 lift      : {diff_agg['f1_lift']:+.4f}\n\n")

        # Per-question results
        f.write("=" * 80 + "\n")
        f.write("  PER-QUESTION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Q{r['id']} ({r['difficulty'].upper()})\n")
            f.write(f"  Question     : {r['question']}\n")
            f.write(f"  Relevant     : {r['relevant_count']} chunks (keyword-based)\n\n")

            f.write(f"  Dense        : P@5={r['dense_p@5']:.4f} R@5={r['dense_r@5']:.4f} " +
                   f"F1@5={r['dense_f1@5']:.4f} MRR={r['dense_mrr']:.4f} NDCG@5={r['dense_ndcg@5']:.4f}\n")

            f.write(f"  Sparse       : P@5={r['sparse_p@5']:.4f} R@5={r['sparse_r@5']:.4f} " +
                   f"F1@5={r['sparse_f1@5']:.4f} MRR={r['sparse_mrr']:.4f} NDCG@5={r['sparse_ndcg@5']:.4f}\n")

            f.write(f"  Fused        : P@5={r['fused_p@5']:.4f} R@5={r['fused_r@5']:.4f} " +
                   f"F1@5={r['fused_f1@5']:.4f} MRR={r['fused_mrr']:.4f} NDCG@5={r['fused_ndcg@5']:.4f}\n")

            f.write(f"  Overlap@5    : {r['overlap@5']:.4f}\n")
            f.write(f"  F1 lift      : {r['f1_lift']:+.4f}\n")
            f.write(f"  Top RRF score: {r['top_rrf_score']:.4f}\n")
            f.write("\n")

        # Metric definitions
        f.write("=" * 80 + "\n")
        f.write("  METRIC DEFINITIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Precision@5  : What fraction of top-5 retrieved chunks are relevant?\n")
        f.write("Recall@5     : What fraction of all relevant chunks are in top-5?\n")
        f.write("F1@5         : Harmonic mean of Precision@5 and Recall@5\n")
        f.write("MRR          : Mean Reciprocal Rank — 1/(rank of first relevant chunk)\n")
        f.write("NDCG@5       : Normalized Discounted Cumulative Gain — rewards relevant chunks at top\n")
        f.write("Overlap@5    : Jaccard similarity between dense and sparse top-5 results\n")
        f.write("F1 lift      : Improvement in F1@5 from fusion over best individual method\n\n")
        f.write("Relevance labeling: keyword-based (15%+ keyword overlap = relevant)\n")
        f.write("Higher is better for all metrics (range: 0.0 to 1.0)\n\n")

    print(f"✓ Report written to: {output_path}\n")
    print("SUMMARY")
    print("─" * 80)
    print(f"  Dense F1@5       : {agg_all['dense_f1@5']:.4f}")
    print(f"  Sparse F1@5      : {agg_all['sparse_f1@5']:.4f}")
    print(f"  Fused F1@5       : {agg_all['fused_f1@5']:.4f}")
    print(f"  Avg F1 lift      : {agg_all['f1_lift']:+.4f}")
    print(f"  Avg overlap      : {agg_all['overlap@5']:.4f}")
    print("─" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = get_settings()

    print("\nLoading BM25 corpus...")
    bm25_index, bm25_records = load_bm25_index(cfg.bm25_path)
    print(f"✓ BM25 index ready: {len(bm25_records)} documents\n")

    # Load test cases
    test_data_path = Path(__file__).parent / "test_data.txt"
    if not test_data_path.exists():
        print(f"  [ERROR] Test data file not found: {test_data_path}")
        return

    print(f"Loading test cases from: {test_data_path}")
    test_cases = load_test_cases(test_data_path)

    if not test_cases:
        print("  [ERROR] No test cases loaded. Check test_data.txt format.")
        return

    print(f"✓ Loaded {len(test_cases)} test cases\n")

    # Count by difficulty
    difficulty_counts = defaultdict(int)
    for tc in test_cases:
        difficulty_counts[tc["difficulty"]] += 1

    print("Test case distribution:")
    for diff in ["easy", "medium", "hard"]:
        print(f"  {diff.upper():<8} : {difficulty_counts[diff]} questions")
    print()

    output_path = Path("results/hybrid_retrieval_quality.txt")
    run_evaluation(bm25_index, bm25_records, test_cases, output_path)


if __name__ == "__main__":
    main()
