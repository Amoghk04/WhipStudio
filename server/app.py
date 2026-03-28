import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv(override=True)

import httpx
from fastapi import FastAPI, Request

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

os.environ["ENABLE_WEB_INTERFACE"] = "true"

app: FastAPI = create_app(
    MLDebugEnvironment,
    MLDebugAction,
    MLDebugObservation,
    env_name="whipstudio",
    max_concurrent_envs=4,
)


@app.get("/")
def ready():
    return {"status": "ok", "message": "whipstudio server is running"}


@app.get("/health")
def health():
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
        from ..baseline_agent import run_single_task
    except ImportError:
        from baseline_agent import run_single_task

    env_url = str(request.base_url).rstrip("/")
    results = {}
    task_scores = {}
    for task_id in ["task1", "task2", "task3", "task4", "task5"]:
        try:
            score = await asyncio.wait_for(run_single_task(task_id, env_url), timeout=120.0)
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
    avg = round(sum(task_scores.values()) / 3, 4)
    return {"baseline_scores": results, "average": avg, "env_url": env_url}


@app.get("/baseline/task/{task_id}")
async def run_baseline_single(task_id: str, request: Request):
    """Run the baseline agent on a single task. Returns score + details."""
    try:
        from ..baseline_agent import run_single_task_detailed
    except ImportError:
        from baseline_agent import run_single_task_detailed

    env_url = str(request.base_url).rstrip("/")
    try:
        result = await asyncio.wait_for(run_single_task_detailed(task_id, env_url), timeout=120.0)
        return {
            "task_id": task_id,
            "score": round(result["score"], 4),
            "status": "ok",
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


try:
    import gradio as gr
    try:
        from ..gradio_app import build_ui
    except ImportError:
        from gradio_app import build_ui

    gradio_ui = build_ui()
    app = gr.mount_gradio_app(app, gradio_ui, path="/ui")
except ImportError as e:
    print(f"Skipping Gradio UI mount: {e}")


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
