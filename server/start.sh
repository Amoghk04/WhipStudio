#!/bin/bash
set -e
export ENABLE_WEB_INTERFACE=true
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1 &
sleep 5 && curl -s http://localhost:7860/health >/dev/null || true
wait
