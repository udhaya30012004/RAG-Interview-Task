"""
ingest/embed_store.py

Read chunks.jsonl → embed each chunk with Gemini → upsert to Pinecone.
Also verifies the final vector count matches the number of chunks written.

Uses:
  - google-genai  (google.genai)        — new Gemini SDK
  - pinecone      (pinecone >= 5.x)     — new Pinecone package
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger(__name__)


EMBEDDING_MODEL = "gemini-embedding-001"   
EMBEDDING_DIM   = 768
EMBEDDING_TASK  = "RETRIEVAL_DOCUMENT"     
BATCH_SIZE      = 100                     
EMBED_DELAY     = 0.1                     


def _get_pinecone_index(api_key: str, region: str, index_name: str):
    """Return a Pinecone Index object, creating the index first if needed."""
    pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d, metric=cosine)",
            index_name, EMBEDDING_DIM,
        )
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),
        )
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("Waiting for Pinecone index to be ready …")
            time.sleep(2)

    return pc.Index(index_name)


def _embed_text(client: genai.Client, text: str) -> list[float]:
    """Embed a single text string using gemini-embedding-001."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            task_type=EMBEDDING_TASK,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def embed_and_store(
    chunks_path: str | Path,
    index_name: str | None = None,
) -> int:
    """
    Read chunks.jsonl, embed each chunk, upsert to Pinecone in batches.

    Parameters
    ----------
    chunks_path : path to chunks.jsonl produced by chunk.py
    index_name  : Pinecone index name (defaults to env var PINECONE_INDEX_NAME)

    Returns
    -------
    int  number of vectors upserted
    """
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

    # ── Credentials ───────────────────────────────────────────────────────────
    api_key_google   = os.environ["GOOGLE_API_KEY"]
    api_key_pinecone = os.environ["PINECONE_API_KEY"]
    region           = os.environ["PINECONE_REGION"]
    index_name       = index_name or os.environ["PINECONE_INDEX_NAME"]

    # ── Clients ───────────────────────────────────────────────────────────────
    gemini_client = genai.Client(api_key=api_key_google)
    index         = _get_pinecone_index(api_key_pinecone, region, index_name)

    # ── Read chunks ───────────────────────────────────────────────────────────
    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path.name)

   
    vectors: list[dict] = []
    total_upserted = 0

    for i, chunk in enumerate(chunks):
        embedding = _embed_text(gemini_client, chunk["text"])
        time.sleep(EMBED_DELAY)

        vectors.append(
            {
                "id":     chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text":     chunk["text"],
                    "source":   chunk["source"],
                    "page_num": chunk["page_num"],
                    "label":    chunk["label"],
                },
            }
        )

        if len(vectors) == BATCH_SIZE or i == len(chunks) - 1:
            index.upsert(vectors=vectors)
            total_upserted += len(vectors)
            logger.info(
                "Upserted batch — %d vectors (total: %d / %d)",
                len(vectors), total_upserted, len(chunks),
            )
            vectors = []

    
    time.sleep(3)  # give Pinecone a moment to update stats
    stats        = index.describe_index_stats()
    vector_count = stats["total_vector_count"]

    if vector_count != len(chunks):
        logger.warning(
            "Count mismatch — JSONL: %d chunks, Pinecone: %d vectors",
            len(chunks), vector_count,
        )
    else:
        logger.info(
            "✓ Verification passed — %d vectors in Pinecone match %d chunks",
            vector_count, len(chunks),
        )

    return total_upserted
