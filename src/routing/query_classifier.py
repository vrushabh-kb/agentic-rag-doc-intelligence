"""
DETERMINISTIC query classifier - no LLM call, no ambiguity, fully unit-testable.
This is the layer you built yourself: a rule-based tagger that decides what
KIND of question the user is asking, before any retrieval happens.

Why deterministic instead of "let the LLM decide"? Three reasons worth saying
out loud in an interview:
  1. Reproducible - same query always routes the same way, easy to debug.
  2. Zero extra API cost/latency - classification is regex, not a model call.
  3. No hallucinated routing - an LLM router can occasionally invent a doc
     name or misfire; rules can't.

Trade-off (say this too, don't oversell it): rules don't generalize to
phrasing they weren't written for. That's a real limitation, and the
natural next iteration is a hybrid - rules first, LLM fallback only when
no rule matches.
"""
import re
from dataclasses import dataclass
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "factual"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    FOLLOWUP = "followup"


_SUMMARY_PATTERNS = [
    r"\bsummar(y|ize|ise)\b", r"\boverview\b", r"\btl;?dr\b", r"\bmain (points|findings)\b",
    r"\bwhat (does|is) (this|the) paper (about|say)\b",
]
_COMPARISON_PATTERNS = [
    r"\bcompar(e|ison)\b", r"\bvs\.?\b", r"\bversus\b", r"\bdifference between\b",
    r"\bwhich (one|paper|approach) is better\b",
]
_DEFINITION_PATTERNS = [
    r"\bwhat is\b", r"\bwhat are\b", r"\bdefine\b", r"\bmeaning of\b", r"\bexplain the term\b",
]
# Weak pronoun/reference signals that suggest this query depends on prior turns.
_FOLLOWUP_PATTERNS = [
    r"^\s*(and|also|what about|how about)\b", r"\bthat (paper|method|approach|result)\b",
    r"\b(it|this|that|those|these)\b.{0,15}\?$",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


@dataclass
class ClassifiedQuery:
    query_type: QueryType
    matched_rule: str  # for debugging/demo transparency - show this in the UI


def classify_query(query: str, has_prior_context: bool) -> ClassifiedQuery:
    """Order matters: check FOLLOWUP first only if there IS prior context
    (a followup pattern with no memory yet is meaningless), then the more
    specific intents (comparison/summary/definition), falling back to FACTUAL."""
    if has_prior_context and _matches_any(query, _FOLLOWUP_PATTERNS):
        return ClassifiedQuery(QueryType.FOLLOWUP, "followup_pattern")

    if _matches_any(query, _COMPARISON_PATTERNS):
        return ClassifiedQuery(QueryType.COMPARISON, "comparison_pattern")

    if _matches_any(query, _SUMMARY_PATTERNS):
        return ClassifiedQuery(QueryType.SUMMARY, "summary_pattern")

    if _matches_any(query, _DEFINITION_PATTERNS):
        return ClassifiedQuery(QueryType.DEFINITION, "definition_pattern")

    return ClassifiedQuery(QueryType.FACTUAL, "default_fallback")
