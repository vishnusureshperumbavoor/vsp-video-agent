#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
echo "Starting VSP Video Agent FastAPI Backend on http://0.0.0.0:8000..."
exec .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir server
