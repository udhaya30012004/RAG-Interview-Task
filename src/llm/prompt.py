"""
src/llm/prompt.py

System and human prompt templates for the RAG answer generation node.
Kept separate from nodes.py so prompt wording can be tuned independently.
"""

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions strictly based "
    "on the provided context excerpts from the Agentic AI ebook. "
    "If the context does not contain enough information to answer the question, "
    "say so clearly — do not fabricate facts. "
    "Keep answers concise, accurate, and grounded in the excerpts."
)

# ── Human prompt template ──────────────────────────────────────────────────────
# {context} and {question} are filled in at runtime.
HUMAN_PROMPT_TEMPLATE = (
    "Context excerpts:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer based only on the context above:"
)


def build_context_block(chunks: list[dict]) -> str:
    """
    Format the fused chunks into a numbered context block for the prompt.

    Each entry shows the chunk's label and page number so the LLM can
    reference the source in its reasoning.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] (page {chunk['page_num']}, {chunk['label']})"
        parts.append(f"{header}\n{chunk['text'].strip()}")
    return "\n\n".join(parts)


def build_human_prompt(question: str, chunks: list[dict]) -> str:
    """Return the filled-in human prompt string."""
    context = build_context_block(chunks)
    return HUMAN_PROMPT_TEMPLATE.format(context=context, question=question)
