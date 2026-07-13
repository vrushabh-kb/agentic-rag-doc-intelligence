"""
Unit tests for the deterministic routing layer. Run with: pytest tests/
The whole point of building routing as rules instead of an LLM call is that
it's testable like this - show this file as evidence the routing is reliable.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.routing.query_classifier import classify_query, QueryType
from src.routing.document_router import route_document


def test_summary_classification():
    result = classify_query("Can you summarize this paper?", has_prior_context=False)
    assert result.query_type == QueryType.SUMMARY


def test_comparison_classification():
    result = classify_query("Compare the LSTM approach vs the transformer approach", has_prior_context=False)
    assert result.query_type == QueryType.COMPARISON


def test_definition_classification():
    result = classify_query("What is a Sharpe ratio?", has_prior_context=False)
    assert result.query_type == QueryType.DEFINITION


def test_followup_requires_context():
    # Followup pattern present but NO prior context -> should NOT be tagged followup
    result = classify_query("What about that result?", has_prior_context=False)
    assert result.query_type != QueryType.FOLLOWUP


def test_followup_with_context():
    result = classify_query("What about that result?", has_prior_context=True)
    assert result.query_type == QueryType.FOLLOWUP


def test_default_factual_fallback():
    result = classify_query("What was the accuracy reported in the experiments?", has_prior_context=False)
    assert result.query_type == QueryType.FACTUAL


def test_document_router_detects_named_doc():
    registry = {"paper_a": "Deep Learning For Stock Prediction", "paper_b": "Factor Models In Finance"}
    decision = route_document(
        query="What does the deep learning for stock prediction paper say about accuracy?",
        query_type=QueryType.FACTUAL,
        doc_registry=registry,
        last_used_doc_ids=None,
    )
    assert decision.target_doc_ids == ["paper_a"]


def test_document_router_falls_back_to_all():
    registry = {"paper_a": "Deep Learning For Stock Prediction"}
    decision = route_document(
        query="What is a moving average?",
        query_type=QueryType.DEFINITION,
        doc_registry=registry,
        last_used_doc_ids=None,
    )
    assert decision.target_doc_ids is None


def test_followup_reuses_last_docs():
    registry = {"paper_a": "Deep Learning For Stock Prediction"}
    decision = route_document(
        query="What about that result?",
        query_type=QueryType.FOLLOWUP,
        doc_registry=registry,
        last_used_doc_ids=["paper_a"],
    )
    assert decision.target_doc_ids == ["paper_a"]
