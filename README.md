# Agentic RAG Document Intelligence System

Multi-document Q&A over research papers with a **deterministic routing layer**
(no LLM decides the route — pure rule-based classification) and **session
memory** for follow-up questions.

## Why this isn't "just a RAG chatbot"

Most student RAG projects: PDF → chunk → embed → similarity search → LLM answers.
That's the retrieval part, and libraries (LangChain/Chroma) do all of it for you.

What's actually engineered here, by hand, in `src/routing/`:
- **Query classifier** — regex/keyword rules tag each query as factual /
  summary / comparison / definition / followup. Zero LLM calls, fully unit
  tested (`tests/test_routing.py`), fully reproducible.
- **Document router** — decides which paper(s) to search based on what's
  named in the query, or reuses the last-used paper(s) for a follow-up.
- **Retrieval strategy** — different query types get different `top_k` and
  section bias (summaries pull from abstract/conclusion; factual lookups
  stay tight).
- **Session memory** — plain in-memory conversation state so "what about
  that result?" resolves correctly within a session.

Libraries handle: PDF parsing (PyMuPDF), embeddings + generation (Gemini),
vector storage/search (ChromaDB), API framework (FastAPI), UI (Streamlit).

## Setup

```bash
git clone <your-repo>
cd agentic-rag-doc-intelligence
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then add your free Gemini API key
```

Get a free Gemini API key: https://aistudio.google.com/apikey

## Add documents

Drop 6-8 PDFs (papers) into `data/raw_pdfs/`, then:

```bash
python scripts/run_ingestion.py
```

This chunks, embeds, and stores everything in a local ChromaDB (persisted
to `data/processed/chroma_db/`), and writes `data/processed/doc_registry.json`
which the router uses to detect document names in queries.

## Run locally

Backend:
```bash
uvicorn api.main:app --reload --port 8000
```

Frontend (separate terminal):
```bash
streamlit run app/streamlit_app.py
```

## Run tests

```bash
pytest tests/ -v
```

## Deploy free

- **Backend → Render**: new Web Service, build command
  `pip install -r requirements.txt`, start command
  `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Add `GEMINI_API_KEY`
  as an environment variable. Note: Render's free tier has an ephemeral
  filesystem — for a persistent demo, either commit the pre-built
  `chroma_db` folder to the repo (small dataset, this is fine) or re-run
  ingestion on startup.
- **Frontend → Streamlit Community Cloud**: point at `app/streamlit_app.py`,
  add `API_URL` (your Render backend URL) in Streamlit's secrets manager.

## Folder structure

```
agentic-rag-doc-intelligence/
├── data/
│   ├── raw_pdfs/            # source PDFs you curate
│   └── processed/           # chroma_db/, doc_registry.json (generated)
├── src/
│   ├── config.py
│   ├── ingestion/
│   │   ├── pdf_loader.py    # PDF -> text + section-guess metadata
│   │   └── chunker.py       # sliding-window chunking
│   ├── routing/              # <- the hand-built differentiator
│   │   ├── query_classifier.py
│   │   ├── document_router.py
│   │   └── retrieval_strategy.py
│   ├── memory/
│   │   └── session_memory.py
│   ├── generation/
│   │   ├── gemini_client.py
│   │   └── prompt_templates.py
│   ├── vectorstore/
│   │   └── chroma_client.py
│   └── orchestrator.py       # ties everything together
├── api/main.py                # FastAPI backend
├── app/streamlit_app.py       # Streamlit frontend
├── scripts/run_ingestion.py
└── tests/test_routing.py
```

