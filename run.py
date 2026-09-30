"""
run.py

Terminal-based RAG pipeline tester with detailed output.

Shows:
  - Dense retrieval results (Pinecone + scores)
  - Sparse retrieval results (BM25 + scores)
  - RRF fused results (merged + RRF scores)
  - Final LLM answer
  - Pipeline status

Run from project root:
    python run.py
"""

import json
import logging
import sys
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from src.graph.graph_build import build_graph
from src.config import get_settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_header(title: str, char: str = "═") -> None:
    """Print a section header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")


def print_chunk(idx: int, chunk: dict, score_label: str = "score") -> None:
    """Print one retrieved chunk with metadata."""
    print(f"  [{idx}] {chunk['chunk_id']}")
    print(f"      {score_label:>12} : {chunk.get('score', chunk.get('rrf_score', 0)):.4f}")
    print(f"      {'page_num':>12} : {chunk['page_num']}")
    print(f"      {'label':>12} : {chunk['label']}")
    print(f"      {'text':>12} : {chunk['text'][:150]}...")
    print()


def main() -> None:
    print_header("RAG Pipeline Test Runner", char="═")

    cfg = get_settings()
    print(f"  LLM Model       : {cfg.llm_model}")
    print(f"  Embedding Model : {cfg.embedding_model}")
    print(f"  Pinecone Index  : {cfg.pinecone_index_name}")
    print(f"  Dense top-k     : {cfg.dense_top_k}")
    print(f"  BM25 top-k      : {cfg.bm25_top_k}")
    print(f"  RRF k           : {cfg.rrf_k}")
    print(f"  Final top-k     : {cfg.final_top_k}")
    print(f"  Relevance thresh: {cfg.relevance_threshold}")

    # Build graph (loads BM25 corpus)
    print_header("Building Graph", char="─")
    graph = build_graph()
    print("  ✓ Graph built and compiled\n")

    # Get question from user
    print_header("Enter Your Question", char="─")
    question = input("  Question: ").strip()

    if not question:
        print("\n  [ERROR] No question provided, exiting.\n")
        return

    # Run the graph
    print_header("Running RAG Pipeline", char="═")
    thread_id = "test-session"
    result = graph.invoke(
        {"question": question},
        config={"configurable": {"thread_id": thread_id}},
    )

    # ── Show dense results ─────────────────────────────────────────────────────
    print_header("Dense Retrieval (Pinecone Vector Search)", char="─")
    dense = result.get("dense_results", [])
    print(f"  Retrieved: {len(dense)} chunks\n")
    for i, chunk in enumerate(dense[:5], start=1):  # show top 5
        print_chunk(i, chunk, score_label="cosine_sim")

    # ── Show sparse results ────────────────────────────────────────────────────
    print_header("Sparse Retrieval (BM25 Keyword Search)", char="─")
    sparse = result.get("sparse_results", [])
    print(f"  Retrieved: {len(sparse)} chunks\n")
    for i, chunk in enumerate(sparse[:5], start=1):  # show top 5
        print_chunk(i, chunk, score_label="bm25_score")

    # ── Show fused results (RRF) ───────────────────────────────────────────────
    print_header("Reciprocal Rank Fusion (RRF)", char="─")
    fused = result.get("fused_chunks", [])
    print(f"  Final top-k: {len(fused)} chunks")
    print(f"  RRF constant k: {cfg.rrf_k}\n")
    for i, chunk in enumerate(fused, start=1):
        print_chunk(i, chunk, score_label="rrf_score")

    # ── Show final answer ──────────────────────────────────────────────────────
    print_header("Generated Answer", char="═")
    answer = result.get("answer", "")
    status = result.get("status", "unknown")
    attempt = result.get("retrieval_attempt", 0)

    print(f"  Status          : {status}")
    print(f"  Retrieval attempt: {attempt}")
    print(f"\n  Answer:\n")
    print(f"  {answer}\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print_header("Summary", char="─")
    print(f"  Question        : {question}")
    print(f"  Dense results   : {len(dense)}")
    print(f"  Sparse results  : {len(sparse)}")
    print(f"  Fused (RRF)     : {len(fused)}")
    print(f"  Top RRF score   : {fused[0]['rrf_score']:.4f}" if fused else "  Top RRF score   : N/A")
    print(f"  Status          : {status}")
    print(f"  Answer length   : {len(answer)} chars")
    print(header := "═" * 80)
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  [INTERRUPTED] Exiting...\n")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\n  [ERROR] {e}\n")
        sys.exit(1)
