"""
Given a query type + routing decision, decide HOW to retrieve:
top_k, and whether to bias toward certain page sections (abstract/conclusion
for summaries, body for factual lookups). Deterministic lookup table -
no model call.
"""
from dataclasses import dataclass
from src.routing.query_classifier import QueryType
from src.routing.document_router import RoutingDecision


@dataclass
class RetrievalParams:
    top_k: int
    section_bias: list[str] | None  # None = no bias, search all sections
    where_filter: dict | None       # Chroma metadata filter


def build_retrieval_params(
    query_type: QueryType, routing: RoutingDecision
) -> RetrievalParams:
    where: dict = {}

    if routing.target_doc_ids:
        if len(routing.target_doc_ids) == 1:
            where["doc_id"] = routing.target_doc_ids[0]
        else:
            where["doc_id"] = {"$in": routing.target_doc_ids}

    if query_type == QueryType.SUMMARY:
        # Bias toward abstract/conclusion chunks, wider top_k since we're
        # synthesizing across more of the document.
        return RetrievalParams(top_k=8, section_bias=["abstract", "conclusion"], where_filter=where or None)

    if query_type == QueryType.COMPARISON:
        # Wider net since we're pulling from 2+ docs and need enough from each.
        return RetrievalParams(top_k=10, section_bias=None, where_filter=where or None)

    if query_type == QueryType.DEFINITION:
        return RetrievalParams(top_k=3, section_bias=None, where_filter=where or None)

    if query_type == QueryType.FOLLOWUP:
        return RetrievalParams(top_k=4, section_bias=None, where_filter=where or None)

    # FACTUAL default
    return RetrievalParams(top_k=4, section_bias=None, where_filter=where or None)
