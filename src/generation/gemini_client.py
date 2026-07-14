"""
Wraps Gemini API calls (embeddings + generation). Library (google-generativeai)
handles the HTTP/auth/protocol; we add batching for embeddings and a single
choke point so prompt construction stays in generation/prompt_templates.py.
"""
import time
import google.generativeai as genai
from src import config

genai.configure(api_key=config.GEMINI_API_KEY)

MAX_BATCH_SIZE = 100  # Gemini's embed_content batch limit per call


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    """Batches requests (up to 100 texts per API call) instead of one call
    per chunk. With 500+ chunks, one-call-per-chunk blows through the free
    tier's ~100 requests/minute limit almost immediately."""
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        batch_num = i // MAX_BATCH_SIZE + 1
        print(f"  Embedding batch {batch_num} ({len(batch)} chunks)...")
        embeddings = _embed_batch_with_retry(batch, task_type)
        all_embeddings.extend(embeddings)
    return all_embeddings


def _embed_batch_with_retry(
    batch: list[str], task_type: str, max_retries: int = 5, base_delay: float = 15.0
) -> list[list[float]]:
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=config.EMBEDDING_MODEL,
                content=batch,
                task_type=task_type,
            )
            return result["embedding"]
        except Exception as e:
            is_rate_limit = "429" in str(e) or "ResourceExhausted" in type(e).__name__
            if is_rate_limit and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                print(f"  Rate limited, waiting {delay:.0f}s before retry...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("Failed to embed batch after all retries")


def generate_answer(prompt: str) -> str:
    model = genai.GenerativeModel(config.GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text