#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting Vibe Video Studio FastAPI Backend on http://0.0.0.0:8000..."
exec .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
