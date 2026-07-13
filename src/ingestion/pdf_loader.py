"""
PDF -> raw text extraction.
Library does the heavy lifting (PyMuPDF parses the PDF binary format,
handles fonts/encoding/layout) - what WE add is the per-page metadata
tagging and section-guessing that the routing layer depends on later.
"""
from dataclasses import dataclass, field
from pathlib import Path
import fitz  # PyMuPDF


@dataclass
class PageChunkSource:
    doc_id: str
    doc_title: str
    page_number: int
    text: str
    section_guess: str = "body"  # abstract / conclusion / references / body


# Cheap heuristic to tag likely section of a page. This is deliberately
# simple (keyword match on first ~200 chars) - it's a signal for the
# retrieval_strategy layer, not meant to be perfect.
SECTION_KEYWORDS = {
    "abstract": ["abstract"],
    "conclusion": ["conclusion", "conclusions", "concluding remarks"],
    "references": ["references", "bibliography"],
}


def _guess_section(page_text: str) -> str:
    head = page_text.strip().lower()[:200]
    for section, keywords in SECTION_KEYWORDS.items():
        if any(head.startswith(kw) or f"\n{kw}" in head for kw in keywords):
            return section
    return "body"


def load_pdf(pdf_path: Path, doc_id: str, doc_title: str) -> list[PageChunkSource]:
    """Extract text page-by-page from a single PDF."""
    pages: list[PageChunkSource] = []
    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            pages.append(
                PageChunkSource(
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page_number=page_number,
                    text=text,
                    section_guess=_guess_section(text),
                )
            )
    return pages


def load_all_pdfs(raw_pdf_dir: Path) -> list[PageChunkSource]:
    """Load every PDF in a directory. doc_id = filename stem, doc_title = filename
    with underscores/dashes replaced by spaces (good enough default; you can
    maintain a proper title registry later in doc_registry.json)."""
    all_pages: list[PageChunkSource] = []
    for pdf_file in sorted(raw_pdf_dir.glob("*.pdf")):
        doc_id = pdf_file.stem
        doc_title = doc_id.replace("_", " ").replace("-", " ")
        all_pages.extend(load_pdf(pdf_file, doc_id, doc_title))
    return all_pages
