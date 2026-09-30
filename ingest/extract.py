'''
this files extract and read the pdf page by page and return the text in a list of strings

'''
import logging
from pathlib import Path
from typing import Any

import pymupdf4llm

logger = logging.getLogger(__name__)


def extract_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract all pages from *pdf_path* as markdown.

    Returns
    -------
    list[dict]
        Each dict has:
          - page_num   : int   (1-indexed)
          - text       : str   (raw markdown from pymupdf4llm)
          - source     : str   (stem of the PDF filename)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting pages from: %s", pdf_path.name)

    # page_chunks=True → one chunk per page instead of one giant string
    raw_pages: list[dict] = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
    )

    pages = []
    for i, page in enumerate(raw_pages, start=1):
        text = page.get("text", "").strip()
        pages.append(
            {
                "page_num": i,
                "text": text,
                "source": pdf_path.stem,   # e.g. "Ebook-Agentic-AI"
            }
        )

    logger.info("Extracted %d pages", len(pages))
    return pages


