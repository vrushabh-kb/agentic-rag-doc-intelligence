"""
Central config. Loads from .env so nothing is hardcoded.
Everything else in the codebase imports from here instead of
reading os.environ directly - keeps config in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")

CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "processed" / "chroma_db")
)
RAW_PDF_DIR = BASE_DIR / "data" / "raw_pdfs"

# Chunking
CHUNK_SIZE = 900          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# Retrieval defaults (overridden per query-type by retrieval_strategy.py)
DEFAULT_TOP_K = 4

COLLECTION_NAME = "documents"
