# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Whipstudio Environment.

This module creates an HTTP server that exposes the WhipstudioEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
import sys

# Ensure project root is on sys.path so package imports work when
# running uvicorn from the `server/` directory directly.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import WhipstudioAction, WhipstudioObservation
    from .WhipStudio_environment import WhipstudioEnvironment
except ImportError:
    try:
        from WhipStudio.models import WhipstudioAction, WhipstudioObservation
        from WhipStudio.server.WhipStudio_environment import WhipstudioEnvironment
    except ImportError:
        from models import WhipstudioAction, WhipstudioObservation
        from server.WhipStudio_environment import WhipstudioEnvironment

# Create the app with web interface and README integration
app = create_app(
    WhipstudioEnvironment,
    WhipstudioAction,
    WhipstudioObservation,
    env_name="WhipStudio",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

@app.get("/")
def ready():
    return {"message": "WhipStudio Environment Server is running!"}

# Health check endpoint
@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m WhipStudio.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn WhipStudio.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
