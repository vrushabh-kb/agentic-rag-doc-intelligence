"""
The orchestrator IS the "agent" - not an LLM-driven ReAct loop, but a fixed
pipeline where every step is one you built: classify -> route -> retrieve
params -> vector search -> prompt -> generate -> update memory.

This is the file to walk an interviewer through end-to-end.
"""
from dataclasses import dataclass

from src.routing.query_classifier import classify_query
from src.routing.document_router import route_document
from src.routing.retrieval_strategy import build_retrieval_params
from src.memory.session_memory import memory_store
from src.generation.gemini_client import embed_text, generate_answer
from src.generation.prompt_templates import build_answer_prompt
from src.vectorstore.chroma_client import VectorStore


@dataclass
class AgentResponse:
    answer: str
    query_type: str
    routing_reason: str
    docs_searched: list[str] | None
    sources: list[dict]


class DocumentIntelligenceAgent:
    def __init__(self, vector_store: VectorStore, doc_registry: dict[str, str]):
        self.vector_store = vector_store
        self.doc_registry = doc_registry  # {doc_id: doc_title}, used by router

    def handle_query(self, session_id: str, query: str) -> AgentResponse:
        has_context = memory_store.has_context(session_id)
        session = memory_store.get(session_id)

        # 1. Classify
        classified = classify_query(query, has_prior_context=has_context)

        # 2. Route to document(s)
        routing = route_document(
            query=query,
            query_type=classified.query_type,
            doc_registry=self.doc_registry,
            last_used_doc_ids=session.last_used_doc_ids,
        )

        # 3. Decide retrieval params
        params = build_retrieval_params(classified.query_type, routing)

        # 4. Retrieve
        query_embedding = embed_text(query, task_type="retrieval_query")
        sources = self._retrieve(query_embedding, params, routing)

        # 5. Build prompt + generate
        history_text = memory_store.history_as_text(session_id)
        prompt = build_answer_prompt(query, sources, history_text)
        answer = generate_answer(prompt)

        # 6. Update memory
        memory_store.add_turn(session_id, "user", query)
        memory_store.add_turn(session_id, "assistant", answer)
        used_doc_ids = routing.target_doc_ids or sorted({s["doc_id"] for s in sources})
        memory_store.set_last_used_docs(session_id, used_doc_ids or None)

        return AgentResponse(
            answer=answer,
            query_type=classified.query_type.value,
            routing_reason=routing.reason,
            docs_searched=routing.target_doc_ids,
            sources=sources,
        )

    def _retrieve(self, query_embedding: list[float], params, routing) -> list[dict]:
        """Single-doc or no-doc-filter queries use one combined similarity
        search as before. But when 2+ docs are targeted (comparison queries,
        or a followup reusing 2+ docs from memory), a single combined query
        can starve one doc entirely if its chunks score slightly lower on
        similarity - the model then has nothing to say about that doc and
        has to admit it found nothing, even though relevant chunks exist.
        Fix: query each targeted doc separately with a floor on how many
        chunks it's guaranteed, then merge. Every targeted doc is
        guaranteed representation."""
        if routing.target_doc_ids and len(routing.target_doc_ids) > 1:
            per_doc_k = max(2, params.top_k // len(routing.target_doc_ids))
            merged: list[dict] = []
            for doc_id in routing.target_doc_ids:
                results = self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=per_doc_k,
                    where={"doc_id": doc_id},
                )
                merged.extend(self._format_sources(results))
            return merged

        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=params.top_k,
            where=params.where_filter,
        )
        return self._format_sources(results)

    @staticmethod
    def _format_sources(chroma_results) -> list[dict]:
        if not chroma_results.get("documents") or not chroma_results["documents"][0]:
            return []
        docs = chroma_results["documents"][0]
        metas = chroma_results["metadatas"][0]
        return [
            {
                "text": doc,
                "doc_id": meta["doc_id"],
                "doc_title": meta["doc_title"],
                "page_number": meta["page_number"],
            }
            for doc, meta in zip(docs, metas)
        ]