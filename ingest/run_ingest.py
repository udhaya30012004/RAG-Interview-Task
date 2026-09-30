"""
ingest/run_ingest.py

Orchestrator: runs the full ingestion pipeline in two explicit stages.

  Stage A — Process & Save metadata (no API calls):
      extract → clean → chunk → writes chunks.jsonl + bm25_corpus.jsonl

  Stage B — Embed & Upsert to Pinecone (API calls):
      reads chunks.jsonl → Gemini embed → Pinecone upsert → verify count

Run from the project root:
    python -m ingest.run_ingest            # runs both Stage A + Stage B
    python -m ingest.run_ingest --metadata-only   # Stage A only (no Pinecone)
    python -m ingest.run_ingest --embed-only      # Stage B only (already have JSONL)
"""

import argparse
import logging
import sys
from pathlib import Path

from ingest.extract import extract_pages
from ingest.clean import clean_pages
from ingest.chunk import chunk_pages
from ingest.embed_store import embed_and_store

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH     = PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"
CHUNKS_PATH  = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
BM25_PATH    = PROJECT_ROOT / "data" / "processed" / "bm25_corpus.jsonl"


def stage_a_metadata() -> int:
    """
    Stage A — extract, clean, chunk, write both JSONL files to data/processed/.
    Returns the number of chunks produced.
    """
    logger.info("─" * 60)
    logger.info("Stage A — Process & save metadata")
    logger.info("─" * 60)

    logger.info("[1/3] Extracting pages from PDF …")
    pages = extract_pages(PDF_PATH)
    logger.info("      %d raw pages extracted", len(pages))

    logger.info("[2/3] Cleaning pages …")
    clean = clean_pages(pages)
    logger.info("      %d pages kept, %d dropped", len(clean), len(pages) - len(clean))

    logger.info("[3/3] Chunking and writing metadata …")
    chunks = chunk_pages(clean, output_path=CHUNKS_PATH)
    logger.info("      %d chunks written", len(chunks))
    logger.info("      chunks.jsonl    → %s", CHUNKS_PATH)
    logger.info("      bm25_corpus.jsonl → %s", BM25_PATH)

    return len(chunks)


def stage_b_embed() -> int:
    """
    Stage B — read chunks.jsonl, embed with Gemini, upsert to Pinecone.
    Returns the number of vectors upserted.
    """
    logger.info("─" * 60)
    logger.info("Stage B — Embed & upsert to Pinecone")
    logger.info("─" * 60)

    if not CHUNKS_PATH.exists():
        logger.error("chunks.jsonl not found at %s", CHUNKS_PATH)
        logger.error("Run Stage A first:  python -m ingest.run_ingest --metadata-only")
        sys.exit(1)

    upserted = embed_and_store(chunks_path=CHUNKS_PATH)
    logger.info("      %d vectors upserted to Pinecone", upserted)
    return upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--metadata-only",
        action="store_true",
        help="Run Stage A only — extract/clean/chunk, write JSONL files, no Pinecone",
    )
    group.add_argument(
        "--embed-only",
        action="store_true",
        help="Run Stage B only — embed existing chunks.jsonl and upsert to Pinecone",
    )
    args = parser.parse_args()

    logger.info("═" * 60)
    logger.info("RAG Ingestion Pipeline")
    logger.info("═" * 60)

    if args.metadata_only:
        n = stage_a_metadata()
        logger.info("═" * 60)
        logger.info("Stage A complete — %d chunks saved to data/processed/", n)
        logger.info("Verify with:  python -m ingest.verify_chunks")
        logger.info("Then embed:   python -m ingest.run_ingest --embed-only")
        logger.info("═" * 60)

    elif args.embed_only:
        n = stage_b_embed()
        logger.info("═" * 60)
        logger.info("Stage B complete — %d vectors in Pinecone", n)
        logger.info("═" * 60)

    else:
        # Default: run both
        n_chunks   = stage_a_metadata()
        n_upserted = stage_b_embed()
        logger.info("═" * 60)
        logger.info("Pipeline complete")
        logger.info("  Chunks written  : %d", n_chunks)
        logger.info("  Vectors in Pinecone : %d", n_upserted)
        logger.info("═" * 60)


if __name__ == "__main__":
    main()
