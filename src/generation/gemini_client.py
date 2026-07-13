"""
Wraps Gemini API calls (embeddings + generation). Library (google-generativeai)
handles the HTTP/auth/protocol; we add batching for embeddings and a single
choke point so prompt construction stays in generation/prompt_templates.py.
"""
import google.generativeai as genai
from src import config

genai.configure(api_key=config.GEMINI_API_KEY)


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    """task_type differs for indexing ('retrieval_document') vs querying
    ('retrieval_query') - Gemini's embedding model uses this to optimize
    the vector space for asymmetric search."""
    result = genai.embed_content(
        model=config.EMBEDDING_MODEL,
        content=text,
        task_type=task_type,
    )
    return result["embedding"]


def embed_batch(texts: list[str], task_type: str = "retrieval_document") -> list[list[float]]:
    return [embed_text(t, task_type) for t in texts]


def generate_answer(prompt: str) -> str:
    model = genai.GenerativeModel(config.GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text
