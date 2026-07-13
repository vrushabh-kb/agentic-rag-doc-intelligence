"""
Run this once (or whenever you add new PDFs) to populate ChromaDB.

Usage:
    python scripts/run_ingestion.py

Drop your PDFs into data/raw_pdfs/ first.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.ingestion.pdf_loader import load_all_pdfs
from src.ingestion.chunker import chunk_pages
from src.generation.gemini_client import embed_batch
from src.vectorstore.chroma_client import VectorStore


def main():
    print(f"Loading PDFs from {config.RAW_PDF_DIR} ...")
    pages = load_all_pdfs(config.RAW_PDF_DIR)
    if not pages:
        print("No PDFs found. Add files to data/raw_pdfs/ and re-run.")
        return
    print(f"Loaded {len(pages)} pages.")

    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding chunks (this calls the Gemini API, may take a bit)...")
    embeddings = embed_batch([c.text for c in chunks], task_type="retrieval_document")

    store = VectorStore()
    store.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "page_number": c.page_number,
                "section_guess": c.section_guess,
            }
            for c in chunks
        ],
    )
    print(f"Stored {store.count()} total chunks in Chroma at {config.CHROMA_PERSIST_DIR}")

    # Write a doc registry file the router/API can load without re-scanning PDFs.
    doc_registry = {}
    for c in chunks:
        doc_registry[c.doc_id] = c.doc_title
    registry_path = Path(config.BASE_DIR) / "data" / "processed" / "doc_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(doc_registry, indent=2))
    print(f"Wrote doc registry ({len(doc_registry)} docs) to {registry_path}")


if __name__ == "__main__":
    main()
