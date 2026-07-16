# Agentic RAG Document Intelligence System

🔗 **Live demo:** https://agentic-rag-doc-intelligence-8.streamlit.app


> Note: both are hosted on free tiers and spin down after ~15 min of
> inactivity. First request after idling can take 30-60s to wake up -
> that's expected, not a bug.

Multi-document Q&A over research papers with a **deterministic routing
layer** (no LLM decides the route — pure rule-based classification) and
**session memory** for follow-up questions.

## Why this isn't "just a RAG chatbot"

Most student RAG projects: PDF → chunk → embed → similarity search → LLM
answers. That's the retrieval part, and libraries (LangChain/Chroma) do
all of it for you.

What's actually engineered here, by hand, in `src/routing/`:
- **Query classifier** — regex/keyword rules tag each query as factual /
  summary / comparison / definition / followup. Zero LLM calls, fully
  unit tested (`tests/test_routing.py`), fully reproducible.
- **Document router** — decides which paper(s) to search based on what's
  named in the query, or reuses the last-used paper(s) for a follow-up.
- **Retrieval strategy** — different query types get different `top_k`
  and section bias (summaries pull from abstract/conclusion; factual
  lookups stay tight).
- **Balanced multi-doc retrieval** — when 2+ docs are targeted (a
  comparison, or a followup reusing 2+ docs), each doc is queried
  separately and merged, so one document can't get shut out of the
  answer by scoring slightly lower in a single combined similarity
  search. (This was a real bug found through testing — see "Known
  limitations" below for how it was diagnosed and fixed.)
- **Session memory** — plain in-memory conversation state so "what about
  that result?" resolves correctly within a session.

Libraries handle: PDF parsing (PyMuPDF), embeddings + generation
(Gemini), vector storage/search (ChromaDB), API framework (FastAPI), UI
(Streamlit).

The embedding model and paper content are off-the-shelf — this project's
value is the hand-built routing/memory layer around them, not a custom
trained model. (Project 2, the Plant Disease Detector, is the
from-scratch-model counterpart to this one.)

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

**Python version:** this project is pinned to Python 3.13 (see
`.python-version`). Newer versions (3.14+) don't yet have prebuilt wheels
for some dependencies (`pydantic-core`, `pillow`), which forces a
from-source compile that fails in most cloud build environments. If
you're setting this up locally and hit compiler errors, check your
Python version first.

## Add documents

Drop PDFs into `data/raw_pdfs/`, then:

```bash
python scripts/run_ingestion.py
```

This chunks, embeds, and stores everything in a local ChromaDB
(persisted to `data/processed/chroma_db/`), and writes
`data/processed/doc_registry.json` which the router uses to detect
document names in queries.

**Document naming matters for routing.** The router matches query text
against each document's `doc_id` (the PDF filename) and its title. If
your files are named generically (e.g. `rag1.pdf`), users can refer to
them as "rag1" and routing will work — but it won't recognize the paper's
*actual* title unless you either (a) name your PDF files after their
real titles, or (b) add a `data/raw_pdfs/titles.json` mapping filenames
to real titles, e.g.:
```json
{ "rag1": "Liquidity Premium and Investment Horizons" }
```

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

### Backend → Render
- New Web Service, connect this repo
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `bash start.sh`
  (this script builds the vector DB from the committed PDFs on first
  boot if it's missing — Render's free tier wipes the filesystem on
  every redeploy, and the DB is too large to commit directly to git, so
  it rebuilds automatically instead. First cold start after any deploy
  will be slower than normal because of this.)
- **Environment variables:**
  - `GEMINI_API_KEY`
  - `EMBEDDING_MODEL=models/gemini-embedding-001`
  - `GENERATION_MODEL=gemini-flash-latest`
  - `CHROMA_PERSIST_DIR=./data/processed/chroma_db`
  - `PYTHON_VERSION=3.13.5`

### Frontend → Streamlit Community Cloud
- New app, point at `app/streamlit_app.py`
- In **Advanced settings → Python version**, select `3.13`
- In **App settings → Secrets**, add:
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
│   │   ├── gemini_client.py  # batched embedding calls w/ retry/backoff
│   │   └── prompt_templates.py
│   ├── vectorstore/
│   │   └── chroma_client.py
│   └── orchestrator.py       # ties everything together
├── api/main.py                # FastAPI backend
├── app/streamlit_app.py       # Streamlit frontend
├── scripts/run_ingestion.py
├── start.sh                   # Render startup: build DB if missing, then serve
└── tests/test_routing.py

