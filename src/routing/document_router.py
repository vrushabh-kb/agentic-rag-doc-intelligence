"""
Decides WHICH document(s) a query should be searched against.
Deterministic: matches known doc titles/aliases against the query text via
substring/fuzzy matching - no LLM call. Falls back to "search everything"
when nothing is detected, which is always a safe default.
"""
import re
from dataclasses import dataclass
from src.routing.query_classifier import QueryType


@dataclass
class RoutingDecision:
    target_doc_ids: list[str] | None  # None = search all docs
    reason: str  # human-readable, shown in UI for demo transparency


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _detect_mentioned_docs(query: str, doc_registry: dict[str, str]) -> list[str]:
    """doc_registry: {doc_id: doc_title}. We check if any doc's title words
    (or a short alias) appear in the query. Simple substring match on
    normalized text - good enough since paper titles are usually distinctive."""
    norm_query = _normalize(query)
    hits = []
    for doc_id, title in doc_registry.items():
        norm_title = _normalize(title)
        # match on full title, or on doc_id itself (users often say "paper 2" etc)
        if norm_title and norm_title in norm_query:
            hits.append(doc_id)
        elif doc_id.lower() in norm_query:
            hits.append(doc_id)
    return hits


def route_document(
    query: str,
    query_type: QueryType,
    doc_registry: dict[str, str],
    last_used_doc_ids: list[str] | None,
) -> RoutingDecision:
    mentioned = _detect_mentioned_docs(query, doc_registry)

    if query_type == QueryType.FOLLOWUP:
        if last_used_doc_ids:
            return RoutingDecision(last_used_doc_ids, "reused_from_session_memory")
        # no memory yet despite followup pattern - fall through to normal routing
        query_type = QueryType.FACTUAL

    if query_type == QueryType.COMPARISON:
        if len(mentioned) >= 2:
            return RoutingDecision(mentioned, "comparison_docs_detected")
        # comparison intent but couldn't identify both docs - search everything
        # and let retrieval_strategy widen top_k to compensate
        return RoutingDecision(None, "comparison_fallback_search_all")

    if mentioned:
        return RoutingDecision(mentioned, "doc_name_detected_in_query")

    return RoutingDecision(None, "no_doc_detected_search_all")
