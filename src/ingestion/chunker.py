"""
Splits page text into overlapping chunks and attaches metadata.
The splitting itself is a simple sliding window (we don't need a library
for this - it's ~15 lines and gives us full control over chunk boundaries,
which matters because our routing/retrieval logic filters on this metadata).
"""
from dataclasses import dataclass
from src.ingestion.pdf_loader import PageChunkSource
from src import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page_number: int
    section_guess: str
    text: str


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def chunk_pages(
    pages: list[PageChunkSource],
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        pieces = _split_text(page.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_id}_p{page.page_number}_c{i}",
                    doc_id=page.doc_id,
                    doc_title=page.doc_title,
                    page_number=page.page_number,
                    section_guess=page.section_guess,
                    text=piece,
                )
            )
    return chunks
