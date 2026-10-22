#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting Vibe Video Studio React Frontend on http://localhost:5174..."
exec npm run dev
