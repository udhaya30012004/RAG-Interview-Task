
"""
ingest/clean.py

Clean raw extracted pages before chunking.
Drops near-empty pages and fixes common PDF text artifacts.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pages with fewer than this many non-whitespace characters
# are considered near-empty / junk.
MIN_CHARS = 100


def _fix_artifacts(text: str) -> str:
    """
    Fix common PDF-to-markdown artifacts:

    - Unicode ligatures (ﬁ → fi, ﬂ → fl)
    - Non-breaking spaces → regular spaces
    - Hyphenated line breaks → rejoined words
    - Excessive blank lines → normalized spacing
    """

    # Unicode ligatures
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")

    # Non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Rejoin words split across a line break:
    # "computa-\ntion" → "computation"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_pages(pages: list[dict],min_chars: int = MIN_CHARS,) -> list[dict]:

    cleaned = []
    dropped = 0

    for page in pages:
        text = _fix_artifacts(page["text"])

        # Count meaningful characters rather than whitespace.
        non_whitespace_chars = len(re.sub(r"\s+", "", text))

        if non_whitespace_chars < min_chars:
            logger.debug(
                "Dropping page %d (%d non-whitespace chars) — below threshold",
                page["page_num"],
                non_whitespace_chars,
            )
            dropped += 1
            continue

        cleaned.append({
            **page,
            "text": text,
        })

    logger.info(
        "clean_pages: kept %d pages, dropped %d near-empty pages",
        len(cleaned),
        dropped,
    )

    return cleaned
