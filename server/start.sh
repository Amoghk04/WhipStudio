#!/bin/bash
set -e

export ENABLE_WEB_INTERFACE=false

# Single process: FastAPI serves both API and mounted Gradio at /ui
uvicorn server.app:app --host 0.0.0.0 --port "${PORT:-7860}"