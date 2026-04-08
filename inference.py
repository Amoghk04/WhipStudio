#!/usr/bin/env python3
"""
Hackathon-compliant inference script for WhipStudio ML Debug Environment.

This script follows the Scaler Meta PyTorch Hackathon requirements:
- Uses OpenAI-compatible client with API_BASE_URL and MODEL_NAME
- Emits structured stdout logs: [START], [STEP], [END]
- Respects runtime limit (<20 min) and resource constraints

Environment Variables:
    API_BASE_URL: The API endpoint for the LLM (e.g., https://api.openai.com/v1)
    MODEL_NAME: The model identifier (e.g., gpt-4, Qwen/Qwen2.5-Coder-32B-Instruct)
    HF_TOKEN: Your API key / HuggingFace token

Usage:
    # With environment at localhost
    python inference.py --env-url http://localhost:7860

    # With HF Space
    python inference.py --env-url https://your-space.hf.space
"""

import argparse
import json
import os
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs in it.

Rules:
- Return ONLY the complete corrected Python code, nothing else.
- No markdown, no backticks, no explanation text.
- The script must print losses in format: LOSSES:[v1, v2, ...]
- For tasks requiring validation metrics, also print: VAL_ACC:X.XX or VAL_ACCS:[v1,...] and FINAL_LOSS:X.XX
- Keep all torch.manual_seed() calls intact.
- Wrap all metrics in ##METRICS_START## and ##METRICS_END## markers.""".strip()

TASK_IDS = ["task1", "task2", "task3", "task4", "task5", "task6"]

MAX_ATTEMPTS_PER_TASK = 3
REQUEST_TIMEOUT = 180.0  # 3 minutes per LLM call
STEP_TIMEOUT = 120.0     # 2 minutes per step (code execution)


# ── Logging Helpers ───────────────────────────────────────────────────────────

def log_start(task_id: str) -> None:
    """Emit [START] log for a task."""
    print(f"[START] task_id={task_id}", flush=True)


def log_step(task_id: str, step: int, action_summary: str, reward: float, done: bool) -> None:
    """Emit [STEP] log for a step within a task."""
    print(
        f"[STEP] task_id={task_id} step={step} action={action_summary} reward={reward:.4f} done={str(done).lower()}",
        flush=True
    )


def log_end(task_id: str, final_score: float) -> None:
    """Emit [END] log for a task."""
    print(f"[END] task_id={task_id} final_score={final_score:.4f}", flush=True)


# ── LLM Client ────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    """Initialize OpenAI-compatible client from environment variables."""
    api_base = os.environ.get("API_BASE_URL")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError(
            "HF_TOKEN or OPENAI_API_KEY must be set in environment"
        )
    
    # Default to OpenAI API if no base URL specified
    if not api_base:
        api_base = "https://api.openai.com/v1"
    
    return OpenAI(
        base_url=api_base,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT,
    )


def get_model_name() -> str:
    """Get model name from environment or use default."""
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


def generate_fix(client: OpenAI, model: str, prompt: str) -> str:
    """Generate a code fix using the LLM."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        
        content = response.choices[0].message.content or ""
        
        # Strip markdown fences if present
        if "```python" in content:
            content = content.split("```python", 1)[1].split("```", 1)[0].strip()
        elif "```" in content:
            content = content.split("```", 1)[1].split("```", 1)[0].strip()
        
        return content.strip()
    
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        return ""


# ── Environment Client ────────────────────────────────────────────────────────

class WhipStudioClient:
    """HTTP client for the WhipStudio environment."""
    
    def __init__(self, env_url: str):
        self.env_url = env_url.rstrip("/")
        self.timeout = httpx.Timeout(STEP_TIMEOUT, connect=10.0)
        self.episode_id = ""  # Track episode_id for session persistence
    
    def health_check(self) -> bool:
        """Check if the environment is reachable."""
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                resp = client.get(f"{self.env_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
    
    def reset(self, task_id: str) -> dict:
        """Reset environment to a specific task."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.env_url}/reset",
                json={"task_id": task_id}
            )
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observation", data)
            self.episode_id = obs.get("episode_id", "")  # Store episode_id
            return obs
    
    def step(self, fixed_code: str, attempt_number: int = 1) -> dict:
        """Submit a fix and get the result."""
        payload = {
            "action": {
                "action_type": "submit_fix",
                "fixed_code": fixed_code,
                "attempt_number": attempt_number,
                "episode_id": self.episode_id,  # Include for session tracking
            }
        }
        
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.env_url}/step", json=payload)
            resp.raise_for_status()
            return resp.json()
    
    def get_tasks(self) -> list[str]:
        """Get list of available tasks."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.env_url}/tasks")
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        tasks = data.get("tasks", [])
                        return [t.get("id") if isinstance(t, dict) else str(t) for t in tasks]
        except Exception as e:
            print(f"[WARNING] Could not fetch tasks: {e}", file=sys.stderr)
        return TASK_IDS


# ── Main Inference Loop ───────────────────────────────────────────────────────

def build_prompt(obs: dict) -> str:
    """Build the user prompt from observation."""
    task_desc = obs.get("task_description", "Fix the buggy code.")
    buggy_code = obs.get("buggy_code", "")
    error_log = obs.get("error_log", "None")
    last_reward = obs.get("last_reward", 0.0)
    
    return f"""Task: {task_desc}

Buggy code:
{buggy_code}

Previous execution output (if any):
{error_log}

Previous score: {last_reward}""".strip()


def run_task(env: WhipStudioClient, llm_client: OpenAI, model: str, task_id: str) -> float:
    """Run inference on a single task. Returns the best score achieved."""
    
    if isinstance(task_id, dict):
        task_id = task_id.get("id", str(task_id))
    
    log_start(task_id)
    
    try:
        obs = env.reset(task_id)
    except Exception as e:
        print(f"[ERROR] Failed to reset {task_id}: {e}", file=sys.stderr)
        log_end(task_id, 0.0)
        return 0.0
    
    best_score = 0.0
    
    for attempt in range(1, MAX_ATTEMPTS_PER_TASK + 1):
        # Reset for each attempt (except first)
        if attempt > 1:
            try:
                obs = env.reset(task_id)
            except Exception as e:
                print(f"[ERROR] Reset failed for attempt {attempt}: {e}", file=sys.stderr)
                continue
        
        prompt = build_prompt(obs)
        fixed_code = generate_fix(llm_client, model, prompt)
        
        if not fixed_code.strip():
            log_step(task_id, attempt, "empty_response", 0.0, False)
            continue
        
        try:
            result = env.step(fixed_code, attempt_number=attempt)
            
            reward = float(result.get("reward", 0.0) or 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)
            
            if reward > best_score:
                best_score = reward
            
            code_len = len(fixed_code)
            log_step(task_id, attempt, f"submit_fix({code_len}chars)", reward, done)
            
            if reward >= 0.95:
                break
                
        except Exception as e:
            print(f"[ERROR] Step failed: {e}", file=sys.stderr)
            log_step(task_id, attempt, "step_error", 0.0, False)
    
    log_end(task_id, best_score)
    return best_score


def main():
    parser = argparse.ArgumentParser(
        description="WhipStudio inference script for OpenEnv Hackathon"
    )
    parser.add_argument(
        "--env-url",
        default=os.environ.get("ENV_URL", "http://localhost:7860"),
        help="URL of the WhipStudio environment"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all tasks)"
    )
    args = parser.parse_args()
    
    # Initialize clients
    print(f"[INFO] Connecting to environment at {args.env_url}", flush=True)
    env = WhipStudioClient(args.env_url)
    
    if not env.health_check():
        print(f"[ERROR] Cannot reach environment at {args.env_url}", file=sys.stderr)
        sys.exit(1)
    
    print("[INFO] Environment is reachable", flush=True)
    
    llm_client = get_openai_client()
    model = get_model_name()
    print(f"[INFO] Using model: {model}", flush=True)
    
    # Determine tasks
    task_ids = args.tasks if args.tasks else env.get_tasks()
    print(f"[INFO] Running tasks: {task_ids}", flush=True)
    
    # Run inference
    start_time = time.time()
    scores = {}
    
    for task_id in task_ids:
        task_start = time.time()
        score = run_task(env, llm_client, model, task_id)
        scores[task_id] = score
        elapsed = time.time() - task_start
        print(f"[INFO] {task_id} completed in {elapsed:.1f}s with score {score:.4f}", flush=True)
    
    # Summary
    total_elapsed = time.time() - start_time
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    
    print("\n" + "=" * 50, flush=True)
    print("[SUMMARY]", flush=True)
    print(f"  Tasks completed: {len(scores)}", flush=True)
    print(f"  Total time: {total_elapsed:.1f}s", flush=True)
    print(f"  Average score: {avg_score:.4f}", flush=True)
    print("  Per-task scores:", flush=True)
    for tid, score in scores.items():
        status = "✓" if score >= 0.7 else "○"
        print(f"    {status} {tid}: {score:.4f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
