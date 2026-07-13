"""
Prompt construction, kept separate from the API-calling code so prompts
can be iterated on without touching gemini_client.py.
"""

ANSWER_PROMPT = """You are a document intelligence assistant. Answer the user's \
question using ONLY the context passages below. If the context doesn't contain \
the answer, say so explicitly instead of guessing.

Conversation so far:
{history}

Retrieved context passages:
{context}

Question: {question}

Answer clearly and concisely. When you use a specific fact, mention which \
document/page it came from (shown in the context passage headers).
"""


def build_answer_prompt(question: str, context_chunks: list[dict], history: str) -> str:
    context_str = "\n\n".join(
        f"[Source: {c['doc_title']}, page {c['page_number']}]\n{c['text']}"
        for c in context_chunks
    )
    if not context_str:
        context_str = "(no relevant passages found)"
    return ANSWER_PROMPT.format(
        history=history or "(none yet)",
        context=context_str,
        question=question,
    )
