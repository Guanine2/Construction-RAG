#!/bin/bash
set -e

# Start Ollama service in background
ollama serve &

echo "Waiting for Ollama daemon to initialize..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 1
done

echo "Ollama is ready. Starting FastAPI application..."

exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}