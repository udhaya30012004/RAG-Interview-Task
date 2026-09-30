"""
src/llm/model.py

LLM initialization and singleton management.

The ChatGroq instance is created once on first use and reused across
all graph invocations to avoid re-initialization overhead.
"""

from langchain_groq import ChatGroq

from src.config import get_settings

# ── LLM singleton ─────────────────────────────────────────────────────────────
_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    
    global _llm
    if _llm is None:
        cfg = get_settings()
        _llm = ChatGroq(
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            api_key=cfg.groq_api_key,
        )
    return _llm
