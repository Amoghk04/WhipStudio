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
    max_concurrent_envs=64,
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
        try:
            from .logging_config import get_metrics
        except ImportError:
            from server.logging_config import get_metrics
        
        metrics = get_metrics()
        return {
            "status": "ok",
            "version": "1.1.0",
            "tasks_available": 6,
            "tools_available": 6,
            "uptime_seconds": metrics.get_metrics().get("uptime_seconds", 0),
            "ready": True,
        }


if not _has_route("/health", "POST"):

    @app.post("/health", include_in_schema=False)
    def health_post():
        return {"status": "ok", "version": "1.1.0"}


@app.get("/metrics")
def get_runtime_metrics():
    """Return runtime metrics for monitoring."""
    try:
        from .logging_config import get_metrics
    except ImportError:
        from server.logging_config import get_metrics
    
    return get_metrics().get_metrics()


@app.get("/session/state")
def get_session_state(episode_id: str = ""):
    """Get state for a specific session by episode_id."""
    try:
        from .environment import _get_session, MAX_TURNS_PER_EPISODE
    except ImportError:
        from server.environment import _get_session, MAX_TURNS_PER_EPISODE
    
    if not episode_id:
        return {
            "error": "episode_id query parameter required",
            "usage": "/session/state?episode_id=<your-episode-id>"
        }
    
    session = _get_session(episode_id)
    if session is None:
        return {
            "error": f"No session found for episode_id '{episode_id}'",
            "hint": "Call POST /reset first to start an episode"
        }
    
    return {
        "episode_id": session.episode_id,
        "task_id": session.task_id,
        "step": session.step_count,
        "turn": session.step_count,  # Alias for compatibility
        "submitted": session.submitted,
        "done": session.submitted,  # Alias for compatibility
        "best_reward": session.best_reward,
        "max_turns": MAX_TURNS_PER_EPISODE,
        "turns_remaining": max(0, MAX_TURNS_PER_EPISODE - session.step_count),
    }


@app.get("/reset")
def reset_liveness():
    return {"status": "ok", "message": "use POST /reset to start an episode"}


@app.get("/tasks")
def list_tasks():
    try:
        from .environment import TOOL_DEFINITIONS
    except ImportError:
        from server.environment import TOOL_DEFINITIONS
    
    return {
        "tasks": [
            {"id": "task1", "name": "Broken training loop", "difficulty": "easy"},
            {"id": "task2", "name": "Silent NaN loss", "difficulty": "medium"},
            {"id": "task3", "name": "OOM and data leakage", "difficulty": "hard"},
            {"id": "task4", "name": "Wrong loss function", "difficulty": "medium"},
            {"id": "task5", "name": "Frozen backbone", "difficulty": "medium"},
            {"id": "task6", "name": "Input-Output mismatch", "difficulty": "hard"},
        ],
        "action_schema": {
            "action_type": "string — one of: submit_fix, execute_snippet, inspect_tensor, run_training_probe, get_variable_state, inspect_diff",
            "fixed_code": "string — for submit_fix: complete runnable Python script",
            "code": "string — for execute_snippet/run_training_probe: Python code to run",
            "setup_code": "string — for inspect_tensor/get_variable_state: setup code",
            "target_expression": "string — for inspect_tensor: expression to inspect",
            "expressions": "list[string] — for get_variable_state: expressions to evaluate",
            "proposed_code": "string — for inspect_diff: proposed fix to diff",
            "steps": "int 1-10 — for run_training_probe: number of steps",
        },
        "tools": [t["name"] for t in TOOL_DEFINITIONS],
        "max_turns_per_episode": 10,
    }


@app.get("/tools")
def list_tools():
    """Return available tools and their schemas for agent system prompts."""
    try:
        from .environment import TOOL_DEFINITIONS, MAX_TURNS_PER_EPISODE
    except ImportError:
        from server.environment import TOOL_DEFINITIONS, MAX_TURNS_PER_EPISODE
    
    return {
        "tools": TOOL_DEFINITIONS,
        "max_turns_per_episode": MAX_TURNS_PER_EPISODE,
        "usage": {
            "description": "Call step() with action_type set to the tool name. Tools return observations with episode_done=False. submit_fix is the terminal action.",
            "example": {
                "action": {
                    "action_type": "execute_snippet",
                    "code": "print('hello world')"
                }
            }
        }
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
        from ..baseline_agent import SUPPORTED_MODEL_IDS, run_single_task, TASK_CONFIG
    except ImportError:
        from baseline_agent import SUPPORTED_MODEL_IDS, run_single_task, TASK_CONFIG

    env_url = str(request.base_url).rstrip("/")
    model_id = request.query_params.get("model_id", "Qwen/Qwen2.5-Coder-32B-Instruct")
    use_tools = request.query_params.get("use_tools", "true").lower() == "true"
    
    if model_id not in SUPPORTED_MODEL_IDS:
        return {
            "error": f"Unsupported model_id '{model_id}'",
            "supported_model_ids": SUPPORTED_MODEL_IDS,
        }

    results = {}
    task_scores = {}
    for task_id in ["task1", "task2", "task3", "task4", "task5", "task6"]:
        task_cfg = TASK_CONFIG.get(task_id, {})
        # Increase timeout for tool-using agent (more turns = more time)
        timeout_secs = 180.0 if use_tools else 120.0
        try:
            score = await asyncio.wait_for(
                run_single_task(task_id, env_url, model_id=model_id),
                timeout=timeout_secs,
            )
            results[task_id] = round(score, 4)
            task_scores[task_id] = round(score, 4)
        except TimeoutError:
            results[task_id] = 0.0
            task_scores[task_id] = 0.0
            results[f"{task_id}_error"] = f"timeout: task took longer than {timeout_secs}s"
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
        "use_tools": use_tools,
        "task_config": TASK_CONFIG,
    }


@app.get("/baseline/task/{task_id}")
async def run_baseline_single(task_id: str, request: Request):
    """Run the baseline agent on a single task. Returns score + details."""
    try:
        from ..baseline_agent import SUPPORTED_MODEL_IDS, run_single_task_detailed, TASK_CONFIG
    except ImportError:
        from baseline_agent import SUPPORTED_MODEL_IDS, run_single_task_detailed, TASK_CONFIG

    env_url = str(request.base_url).rstrip("/")
    model_id = request.query_params.get("model_id", "Qwen/Qwen2.5-Coder-32B-Instruct")
    use_tools = request.query_params.get("use_tools", "true").lower() == "true"
    
    if model_id not in SUPPORTED_MODEL_IDS:
        return {
            "task_id": task_id,
            "score": 0.0,
            "status": "error",
            "error": f"Unsupported model_id '{model_id}'",
            "supported_model_ids": SUPPORTED_MODEL_IDS,
        }

    task_cfg = TASK_CONFIG.get(task_id, {"max_turns": 10})
    # Longer timeout for tool-using agent
    timeout_secs = 180.0 if use_tools else 120.0
    
    try:
        result = await asyncio.wait_for(
            run_single_task_detailed(task_id, env_url, model_id=model_id, use_tools=use_tools),
            timeout=timeout_secs,
        )
        return {
            "task_id": task_id,
            "score": round(result["score"], 4),
            "status": "ok",
            "model_id": model_id,
            "use_tools": use_tools,
            "max_turns": task_cfg.get("max_turns", 10),
            "fixed_code": result.get("fixed_code", ""),
            "output": result.get("output", ""),
            "attempts": result.get("attempts", []),
            "tool_history": result.get("tool_history", []),
        }
    except TimeoutError:
        return {"task_id": task_id, "score": 0.0, "status": "timeout", "error": f"Task took longer than {timeout_secs}s"}
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
