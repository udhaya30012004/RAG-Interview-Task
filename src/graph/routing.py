"""
src/graph/routing.py

Conditional edge functions for LangGraph.

Each function reads state and returns a string node name, which LangGraph
uses to pick the next node.
"""

from src.graph.state import RAGState


def route_after_validate(state: RAGState) -> str:
    """
    After validate_query:
      - invalid_query → "invalid" (mapped to END)
      - valid         → "valid" (mapped to parallel retrieve_dense + retrieve_sparse)
    """
    status = state.get("status")
    if status == "invalid_query":
        return "invalid"
    return "valid"


def route_after_check_relevance(state: RAGState) -> str:
    """
    After check_relevance:
      - answered                           → generate_answer
      - low_relevance + first attempt (0)  → retry_retrieve
      - low_relevance + retry done (1)     → fallback_answer
    """
    status  = state.get("status")
    attempt = state.get("retrieval_attempt", 0)

    if status == "answered":
        return "generate_answer"

    # low_relevance
    if attempt == 0:
        return "retry_retrieve"
    else:
        return "fallback_answer"
