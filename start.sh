#!/bin/bash
set -e

if [ ! -d "data/processed/chroma_db" ] || [ -z "$(ls -A data/processed/chroma_db 2>/dev/null)" ]; then
    echo "No existing vector DB found - running ingestion..."
    python scripts/run_ingestion.py
else
    echo "Vector DB already present - skipping ingestion."
fi

uvicorn api.main:app --host 0.0.0.0 --port "$PORT"