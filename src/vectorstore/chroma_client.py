"""
Thin wrapper around ChromaDB. Library handles: vector index, similarity
search, persistence to disk. We add: a stable interface the rest of the
codebase depends on, so we could swap Chroma for another vector DB later
without touching routing/orchestrator code.
"""
import chromadb
from src import config


class VectorStore:
    def __init__(self, persist_dir: str = config.CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = config.DEFAULT_TOP_K,
        where: dict | None = None,
    ):
        """where: Chroma metadata filter, e.g. {"doc_id": "paper1"} or
        {"doc_id": {"$in": ["paper1", "paper2"]}} for multi-doc routing."""
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

    def list_doc_ids(self) -> list[str]:
        """Used by document_router to know what's available to route to."""
        data = self.collection.get(include=["metadatas"])
        doc_ids = {m["doc_id"] for m in data["metadatas"]} if data["metadatas"] else set()
        return sorted(doc_ids)

    def count(self) -> int:
        return self.collection.count()
