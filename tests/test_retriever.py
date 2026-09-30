"""
tests/test_retriever.py

Retrieval Quality Evaluation Suite

Tests the hybrid retrieval system (dense + sparse + RRF) against a curated
set of 10 questions with human-labeled relevant chunks.

Metrics computed:
  - Precision@k (how many retrieved chunks are relevant)
  - Recall@k (what % of all relevant chunks were retrieved)
  - F1@k (harmonic mean of precision and recall)
  - MRR (Mean Reciprocal Rank — position of first relevant chunk)
  - NDCG@k (Normalized Discounted Cumulative Gain — rank quality)

Output: d:/RAG_Project/results/retriever_quality.txt

Run from project root:
    python -m tests.test_retriever
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from datetime import datetime
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.graph_build import build_graph
from src.config import get_settings

logging.basicConfig(level=logging.WARNING)  # suppress info logs for cleaner output


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES — loaded from tests/test_data.txt
# ══════════════════════════════════════════════════════════════════════════════

def load_test_cases(test_data_path: Path) -> List[Dict]:
    """
    Load test questions from test_data.txt.

    Format: difficulty|question|keyword1,keyword2,keyword3
    Lines starting with # are comments.
    """
    test_cases = []

    with test_data_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) != 3:
                print(f"  [WARN] Skipping malformed line {line_num}: {line[:60]}")
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


# ══════════════════════════════════════════════════════════════════════════════
# RELEVANCE SCORING
# ══════════════════════════════════════════════════════════════════════════════

def compute_relevance_score(chunk_text: str, keywords: List[str]) -> float:
    """
    Score a chunk's relevance to the query based on keyword presence.

    Returns a score in [0, 1] based on how many keywords appear in the text.
    This is a lightweight proxy for human relevance judgments.
    """
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords) if keywords else 0.0


def label_relevant_chunks(
    retrieved: List[Dict],
    keywords: List[str],
    threshold: float = 0.15
) -> Set[str]:
    """
    Label which retrieved chunks are relevant based on keyword overlap.

    Returns
    -------
    Set of chunk_ids that meet the relevance threshold.
    """
    relevant = set()
    for chunk in retrieved:
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
    """
    MRR = 1 / (rank of first relevant chunk)
    Returns 0 if no relevant chunk found.
    """
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    NDCG@k (Normalized Discounted Cumulative Gain)
    Binary relevance: relevant=1, non-relevant=0
    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = DCG of perfect ranking (all relevant first)
    NDCG = DCG / IDCG
    """
    if len(relevant_ids) == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 if cid in relevant_ids else 0.0) / math.log2(i + 1)
        for i, cid in enumerate(top_k, start=1)
    )

    # Ideal DCG: all relevant chunks at the top
    ideal_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(graph, test_cases: List[Dict], output_path: Path) -> None:
    """Run all test cases and compute aggregate metrics."""

    cfg = get_settings()
    results = []
    aggregate = defaultdict(list)

    print("\n" + "═" * 80)
    print("  RETRIEVAL QUALITY EVALUATION")
    print("═" * 80)
    print(f"  Test cases    : {len(test_cases)}")
    print(f"  Retrieval     : Dense (Pinecone) + Sparse (BM25) + RRF")
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

        print(f"[{i}/{len(test_cases)}] Testing Q{q_id} ({difficulty.upper()}): {question[:60]}...")

        # Run retrieval
        result = graph.invoke(
            {"question": question},
            config={"configurable": {"thread_id": f"eval-{q_id}"}},
        )

        fused = result.get("fused_chunks", [])
        retrieved_ids = [c["chunk_id"] for c in fused]

        # Label relevance based on keywords
        # We use all dense+sparse results to find ALL potentially relevant chunks,
        # then check what made it into top-k
        all_candidates = result.get("dense_results", []) + result.get("sparse_results", [])
        relevant_ids = label_relevant_chunks(all_candidates, keywords, threshold=0.15)

        # Compute metrics @ k=5 (final_top_k)
        k = 5
        p_at_k = precision_at_k(retrieved_ids, relevant_ids, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k)
        f1_k = f1_at_k(retrieved_ids, relevant_ids, k)
        mrr = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        ndcg_k = ndcg_at_k(retrieved_ids, relevant_ids, k)

        # Store
        result_entry = {
            "id": q_id,
            "difficulty": difficulty,
            "question": question,
            "retrieved_count": len(fused),
            "relevant_count": len(relevant_ids),
            "precision@5": p_at_k,
            "recall@5": r_at_k,
            "f1@5": f1_k,
            "mrr": mrr,
            "ndcg@5": ndcg_k,
            "top_rrf_score": fused[0]["rrf_score"] if fused else 0.0,
        }
        results.append(result_entry)

        # Aggregate by difficulty
        aggregate[difficulty].append(result_entry)
        aggregate["all"].append(result_entry)

        print(f"      P@5={p_at_k:.3f} R@5={r_at_k:.3f} F1@5={f1_k:.3f} MRR={mrr:.3f} NDCG@5={ndcg_k:.3f}")

    print("\n" + "─" * 80 + "\n")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    def avg_metrics(entries: List[Dict]) -> Dict[str, float]:
        if not entries:
            return {}
        return {
            "precision@5": sum(e["precision@5"] for e in entries) / len(entries),
            "recall@5": sum(e["recall@5"] for e in entries) / len(entries),
            "f1@5": sum(e["f1@5"] for e in entries) / len(entries),
            "mrr": sum(e["mrr"] for e in entries) / len(entries),
            "ndcg@5": sum(e["ndcg@5"] for e in entries) / len(entries),
        }

    agg_all = avg_metrics(aggregate["all"])
    agg_easy = avg_metrics(aggregate.get("easy", []))
    agg_medium = avg_metrics(aggregate.get("medium", []))
    agg_hard = avg_metrics(aggregate.get("hard", []))

    # ── Write report ──────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RETRIEVAL QUALITY EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test cases   : {len(test_cases)} questions\n")
        f.write(f"Retrieval    : Hybrid (Dense Pinecone + Sparse BM25 + RRF)\n")
        f.write(f"Dense top-k  : {cfg.dense_top_k}\n")
        f.write(f"BM25 top-k   : {cfg.bm25_top_k}\n")
        f.write(f"RRF k        : {cfg.rrf_k}\n")
        f.write(f"Final top-k  : {cfg.final_top_k}\n")
        f.write(f"Eval @ k     : 5\n\n")

        f.write("=" * 80 + "\n")
        f.write("  AGGREGATE METRICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"ALL ({len(aggregate['all'])} questions)\n")
        f.write(f"  Precision@5  : {agg_all['precision@5']:.4f}\n")
        f.write(f"  Recall@5     : {agg_all['recall@5']:.4f}\n")
        f.write(f"  F1@5         : {agg_all['f1@5']:.4f}\n")
        f.write(f"  MRR          : {agg_all['mrr']:.4f}\n")
        f.write(f"  NDCG@5       : {agg_all['ndcg@5']:.4f}\n\n")

        f.write(f"EASY ({len(aggregate.get('easy', []))} questions)\n")
        if agg_easy:
            f.write(f"  Precision@5  : {agg_easy['precision@5']:.4f}\n")
            f.write(f"  Recall@5     : {agg_easy['recall@5']:.4f}\n")
            f.write(f"  F1@5         : {agg_easy['f1@5']:.4f}\n")
            f.write(f"  MRR          : {agg_easy['mrr']:.4f}\n")
            f.write(f"  NDCG@5       : {agg_easy['ndcg@5']:.4f}\n\n")

        f.write(f"MEDIUM ({len(aggregate.get('medium', []))} questions)\n")
        if agg_medium:
            f.write(f"  Precision@5  : {agg_medium['precision@5']:.4f}\n")
            f.write(f"  Recall@5     : {agg_medium['recall@5']:.4f}\n")
            f.write(f"  F1@5         : {agg_medium['f1@5']:.4f}\n")
            f.write(f"  MRR          : {agg_medium['mrr']:.4f}\n")
            f.write(f"  NDCG@5       : {agg_medium['ndcg@5']:.4f}\n\n")

        f.write(f"HARD ({len(aggregate.get('hard', []))} questions)\n")
        if agg_hard:
            f.write(f"  Precision@5  : {agg_hard['precision@5']:.4f}\n")
            f.write(f"  Recall@5     : {agg_hard['recall@5']:.4f}\n")
            f.write(f"  F1@5         : {agg_hard['f1@5']:.4f}\n")
            f.write(f"  MRR          : {agg_hard['mrr']:.4f}\n")
            f.write(f"  NDCG@5       : {agg_hard['ndcg@5']:.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("  PER-QUESTION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Q{r['id']} ({r['difficulty'].upper()})\n")
            f.write(f"  Question     : {r['question']}\n")
            f.write(f"  Retrieved    : {r['retrieved_count']} chunks\n")
            f.write(f"  Relevant     : {r['relevant_count']} chunks (keyword-based)\n")
            f.write(f"  Precision@5  : {r['precision@5']:.4f}\n")
            f.write(f"  Recall@5     : {r['recall@5']:.4f}\n")
            f.write(f"  F1@5         : {r['f1@5']:.4f}\n")
            f.write(f"  MRR          : {r['mrr']:.4f}\n")
            f.write(f"  NDCG@5       : {r['ndcg@5']:.4f}\n")
            f.write(f"  Top RRF score: {r['top_rrf_score']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("  METRIC DEFINITIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Precision@5  : What fraction of top-5 retrieved chunks are relevant?\n")
        f.write("Recall@5     : What fraction of all relevant chunks are in top-5?\n")
        f.write("F1@5         : Harmonic mean of Precision@5 and Recall@5\n")
        f.write("MRR          : Mean Reciprocal Rank — 1/(rank of first relevant chunk)\n")
        f.write("NDCG@5       : Normalized Discounted Cumulative Gain — rewards relevant chunks at top\n\n")
        f.write("Relevance labeling: keyword-based (15%+ keyword overlap = relevant)\n")
        f.write("Higher is better for all metrics (range: 0.0 to 1.0)\n\n")

    print(f"✓ Report written to: {output_path}\n")
    print("SUMMARY")
    print("─" * 80)
    print(f"  Overall Precision@5 : {agg_all['precision@5']:.4f}")
    print(f"  Overall Recall@5    : {agg_all['recall@5']:.4f}")
    print(f"  Overall F1@5        : {agg_all['f1@5']:.4f}")
    print(f"  Overall MRR         : {agg_all['mrr']:.4f}")
    print(f"  Overall NDCG@5      : {agg_all['ndcg@5']:.4f}")
    print("─" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\nBuilding graph (loading BM25 corpus)...")
    graph = build_graph()
    print("✓ Graph ready\n")

    # Load test cases from external file
    test_data_path = Path(__file__).parent / "test_data.txt"
    if not test_data_path.exists():
        print(f"  [ERROR] Test data file not found: {test_data_path}")
        print("  Create tests/test_data.txt with test questions.")
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

    output_path = Path("d:/RAG_Project/results/retriever_quality.txt")
    run_evaluation(graph, test_cases, output_path)


if __name__ == "__main__":
    main()
