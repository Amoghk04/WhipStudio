import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.responses import HTMLResponse

from openenv.core.env_server.http_server import create_app

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from ..models import MLDebugAction, MLDebugObservation
    from .environment import MLDebugEnvironment
    from .tasks.graders import RunResult, score_task
except ImportError:
    from models import MLDebugAction, MLDebugObservation
    from server.environment import MLDebugEnvironment
    from server.tasks.graders import RunResult, score_task

# Disable OpenEnv's default web UI so /web can mirror the custom Gradio UI.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

app: FastAPI = create_app(
    MLDebugEnvironment,
    MLDebugAction,
    MLDebugObservation,
    env_name="whipstudio",
    max_concurrent_envs=4,
)


@app.get("/__build", include_in_schema=False)
def build_info():
    """Build/runtime fingerprint to confirm what code is deployed."""
    import platform

    return {
        "env_name": "whipstudio",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "port": os.environ.get("PORT"),
        "enable_web_interface": os.environ.get("ENABLE_WEB_INTERFACE"),
    }


def _has_route(path: str, method: str) -> bool:
    method = method.upper()
    for route in app.router.routes:
        if getattr(route, "path", None) != path:
            continue
        methods = getattr(route, "methods", None)
        if methods and method in methods:
            return True
    return False


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui/", status_code=307)


if not _has_route("/health", "GET"):

    @app.get("/health", include_in_schema=False)
    def health_get():
        return {"status": "ok"}


if not _has_route("/health", "POST"):

    @app.post("/health", include_in_schema=False)
    def health_post():
        return {"status": "ok"}


@app.get("/reset")
def reset_liveness():
    return {"status": "ok", "message": "use POST /reset to start an episode"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": "task1", "name": "Broken training loop", "difficulty": "easy"},
            {"id": "task2", "name": "Silent NaN loss", "difficulty": "medium"},
            {"id": "task3", "name": "OOM and data leakage", "difficulty": "hard"},
            {"id": "task4", "name": "Wrong loss function", "difficulty": "medium"},
            {"id": "task5", "name": "Frozen backbone", "difficulty": "medium"},
        ],
        "action_schema": {
            "fixed_code": "string (required) — complete runnable Python script",
            "explanation": "string (optional) — description of bugs found",
            "attempt_number": "int 1-3 (optional) — which attempt this is",
        },
    }


@app.post("/grader")
def run_grader(payload: dict):
    task_id = payload.get("task_id", "task1")
    result = RunResult(
        exit_code=payload.get("exit_code", -1),
        stdout=payload.get("stdout", ""),
        stderr=payload.get("stderr", ""),
        elapsed_seconds=payload.get("elapsed", 0.0),
        timed_out=payload.get("timed_out", False),
        fixed_code=payload.get("fixed_code", ""),
    )
    score, breakdown = score_task(task_id, result)
    return {"task_id": task_id, "score": score, "breakdown": breakdown}


@app.get("/baseline")
async def run_baseline(request: Request):
    try:
        from ..baseline_agent import SUPPORTED_MODEL_IDS, run_single_task
    except ImportError:
        from baseline_agent import SUPPORTED_MODEL_IDS, run_single_task

    env_url = str(request.base_url).rstrip("/")
    model_id = request.query_params.get("model_id", "Qwen/Qwen2.5-Coder-32B-Instruct")
    if model_id not in SUPPORTED_MODEL_IDS:
        return {
            "error": f"Unsupported model_id '{model_id}'",
            "supported_model_ids": SUPPORTED_MODEL_IDS,
        }

    results = {}
    task_scores = {}
    for task_id in ["task1", "task2", "task3", "task4", "task5"]:
        try:
            score = await asyncio.wait_for(
                run_single_task(task_id, env_url, model_id=model_id),
                timeout=120.0,
            )
            results[task_id] = round(score, 4)
            task_scores[task_id] = round(score, 4)
        except TimeoutError:
            results[task_id] = 0.0
            task_scores[task_id] = 0.0
            results[f"{task_id}_error"] = "timeout: task took longer than 120s"
        except httpx.HTTPError as exc:
            results[task_id] = 0.0
            task_scores[task_id] = 0.0
            results[f"{task_id}_error"] = f"http_error: {exc.__class__.__name__}: {exc}"
        except Exception as exc:
            results[task_id] = 0.0
            task_scores[task_id] = 0.0
            results[f"{task_id}_error"] = f"internal_error: {exc.__class__.__name__}: {exc}"
    avg = round(sum(task_scores.values()) / max(1, len(task_scores)), 4)
    return {
        "baseline_scores": results,
        "average": avg,
        "env_url": env_url,
        "model_id": model_id,
    }


@app.get("/baseline/task/{task_id}")
async def run_baseline_single(task_id: str, request: Request):
    """Run the baseline agent on a single task. Returns score + details."""
    try:
        from ..baseline_agent import SUPPORTED_MODEL_IDS, run_single_task_detailed
    except ImportError:
        from baseline_agent import SUPPORTED_MODEL_IDS, run_single_task_detailed

    env_url = str(request.base_url).rstrip("/")
    model_id = request.query_params.get("model_id", "Qwen/Qwen2.5-Coder-32B-Instruct")
    if model_id not in SUPPORTED_MODEL_IDS:
        return {
            "task_id": task_id,
            "score": 0.0,
            "status": "error",
            "error": f"Unsupported model_id '{model_id}'",
            "supported_model_ids": SUPPORTED_MODEL_IDS,
        }

    try:
        result = await asyncio.wait_for(
            run_single_task_detailed(task_id, env_url, model_id=model_id),
            timeout=120.0,
        )
        return {
            "task_id": task_id,
            "score": round(result["score"], 4),
            "status": "ok",
            "model_id": model_id,
            "fixed_code": result.get("fixed_code", ""),
            "output": result.get("output", ""),
            "attempts": result.get("attempts", []),
        }
    except TimeoutError:
        return {"task_id": task_id, "score": 0.0, "status": "timeout", "error": "Task took longer than 120s"}
    except Exception as exc:
        return {"task_id": task_id, "score": 0.0, "status": "error", "error": f"{exc.__class__.__name__}: {exc}"}


@app.get("/baseline/health")
def baseline_health():
    hf_token_present = bool(os.environ.get("HF_TOKEN"))
    model_ready = False
    model_error = None

    try:
        try:
            from ..baseline_agent import get_model
        except ImportError:
            from baseline_agent import get_model

        get_model()
        model_ready = True
    except Exception as exc:
        model_error = f"{exc.__class__.__name__}: {exc}"

    status = "ok" if hf_token_present and model_ready else "degraded"
    return {
        "status": status,
        "hf_token_present": hf_token_present,
        "model_ready": model_ready,
        "model_error": model_error,
    }


_ui_mounted = False


@app.get("/ui", include_in_schema=False)
def ui_trailing_slash_redirect():
    # Gradio's HTML references assets as `./assets/...`.
    # Without the trailing slash, browsers resolve those to `/assets/...` (breaking the UI).
    return RedirectResponse(url="/ui/", status_code=307)


try:
    import gradio as gr
    try:
        from ..gradio_app import build_ui
    except ImportError:
        from gradio_app import build_ui

    gradio_ui = build_ui()
    app = gr.mount_gradio_app(app, gradio_ui, path="/ui")
    _ui_mounted = True
except Exception as e:
    # Don't fail silently in Spaces: return a helpful error page at /ui.
    import traceback

    print(f"Failed to mount Gradio UI: {e}")
    traceback.print_exc()


if not _ui_mounted:
    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/", include_in_schema=False)
    def ui_mount_failed():
        return HTMLResponse(
            "<h2>WhipStudio UI failed to start</h2>"
            "<p>The API server is running, but the Gradio UI could not be mounted.</p>"
            "<p>Check container logs for <code>Failed to mount Gradio UI</code>.</p>",
            status_code=500,
        )


@app.api_route("/web", methods=["GET", "POST"], include_in_schema=False)
def web_redirect_root():
    return RedirectResponse(url="/ui/", status_code=307)


@app.api_route("/web/{path:path}", methods=["GET", "POST"], include_in_schema=False)
def web_redirect_path(path: str):
    if path:
        return RedirectResponse(url=f"/ui/{path}", status_code=307)
    return RedirectResponse(url="/ui/", status_code=307)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
