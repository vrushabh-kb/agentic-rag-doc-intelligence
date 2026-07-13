"""
FastAPI backend. Deploy target: Render (free tier).
This is what proves you built a real API, not just a Streamlit script -
show this file when someone asks "how is this different from a notebook".
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from src import config
from src.orchestrator import DocumentIntelligenceAgent
from src.vectorstore.chroma_client import VectorStore

app = FastAPI(title="Agentic RAG Document Intelligence API")

_vector_store = VectorStore()
_registry_path = Path(config.BASE_DIR) / "data" / "processed" / "doc_registry.json"
_doc_registry = json.loads(_registry_path.read_text()) if _registry_path.exists() else {}
_agent = DocumentIntelligenceAgent(vector_store=_vector_store, doc_registry=_doc_registry)


class ChatRequest(BaseModel):
    session_id: str | None = None
    query: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    query_type: str
    routing_reason: str
    docs_searched: list[str] | None
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "docs_indexed": len(_doc_registry), "chunks_indexed": _vector_store.count()}


@app.get("/documents")
def list_documents():
    return {"documents": _doc_registry}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    result = _agent.handle_query(session_id=session_id, query=req.query)
    return ChatResponse(
        session_id=session_id,
        answer=result.answer,
        query_type=result.query_type,
        routing_reason=result.routing_reason,
        docs_searched=result.docs_searched,
        sources=result.sources,
    )
