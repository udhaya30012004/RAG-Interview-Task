"""
ingest/chunk.py

Split cleaned pages into overlapping chunks, tag each with a
structural label, and write results to data/processed/chunks.jsonl.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ── Splitter config ────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Order matters: first match wins.
_LABEL_PATTERNS: list[tuple[str, str]] = [
    ("introduction", r"\b(introduction|overview|preface|foreword|about this)\b"),
    ("case_study",   r"\b(case study|real.world|example|use case|scenario)\b"),
    ("conclusion",   r"\b(conclusion|summary|key takeaway|looking ahead|next step)\b"),
    ("core_concept", r"."),   # catch-all — matches everything
]


def _label_page(text: str) -> str:
    """Return a structural label for the page based on keyword heuristics."""
    lowered = text.lower()
    for label, pattern in _LABEL_PATTERNS:
        if re.search(pattern, lowered):
            return label
    return "core_concept"


def chunk_pages(
    pages: list[dict[str, Any]],
    output_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split cleaned pages into chunks and write to *output_path* as JSONL.

    Each chunk dict contains:
        chunk_id   : str   e.g. "Ebook-Agentic-AI_p3_c1"
        text       : str   the chunk content
        source     : str   PDF stem
        page_num   : int   source page (1-indexed)
        label      : str   structural tag

    Parameters
    ----------
    pages       : output of clean.clean_pages()
    output_path : path to write chunks.jsonl
    chunk_size  : max characters per chunk
    chunk_overlap: overlap between consecutive chunks

    Returns
    -------
    list[dict]  all chunks (also written to disk)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[dict[str, Any]] = []

    for page in pages:
        label = _label_page(page["text"])
        raw_chunks = splitter.split_text(page["text"])

        for i, text in enumerate(raw_chunks, start=1):
            chunk_id = f"{page['source']}_p{page['page_num']}_c{i}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "label": label,
                }
            )

    # ── Write chunks.jsonl — full chunk records ────────────────────────────────
    with output_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    
    bm25_path = output_path.parent / "bm25_corpus.jsonl"
    with bm25_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            record = {
                "chunk_id" : chunk["chunk_id"],
                "tokens"   : chunk["text"].lower().split(),   # pre-tokenised
                "text"     : chunk["text"],                   # needed for RRF merge
                "source"   : chunk["source"],
                "page_num" : chunk["page_num"],
                "label"    : chunk["label"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "chunk_pages: %d chunks from %d pages → %s",
        len(all_chunks),
        len(pages),
        output_path,
    )
    logger.info("BM25 corpus written → %s", bm25_path)
    return all_chunks
