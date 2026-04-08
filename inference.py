#!/usr/bin/env python3
"""
Hackathon-compliant inference script for WhipStudio ML Debug Environment.

This script follows the Scaler Meta PyTorch Hackathon requirements:
- Uses OpenAI-compatible client with API_BASE_URL and MODEL_NAME
- Emits structured stdout logs: [START], [STEP], [END]
- Respects runtime limit (<20 min) and resource constraints

Supports two modes:
- Tool-calling agent (default): Multi-turn debugging with tools before submission
- Simple agent (--no-tools): Direct submit_fix only (legacy behavior)

Environment Variables:
    API_BASE_URL: The API endpoint for the LLM (e.g., https://api.openai.com/v1)
    MODEL_NAME: The model identifier (e.g., gpt-4, Qwen/Qwen2.5-Coder-32B-Instruct)
    HF_TOKEN: Your API key / HuggingFace token

Usage:
    # Tool-calling agent (default)
    python inference.py --env-url http://localhost:7860

    # Simple submit-only mode (legacy)
    python inference.py --env-url http://localhost:7860 --no-tools
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Optional

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

TASK_IDS = ["task1", "task2", "task3", "task4", "task5", "task6"]

TASK_CONFIG = {
    "task1": {"max_turns": 8, "difficulty": "easy"},
    "task2": {"max_turns": 10, "difficulty": "medium"},
    "task3": {"max_turns": 10, "difficulty": "hard"},
    "task4": {"max_turns": 10, "difficulty": "medium"},
    "task5": {"max_turns": 10, "difficulty": "medium"},
    "task6": {"max_turns": 10, "difficulty": "hard"},
}

MAX_ATTEMPTS_PER_TASK = 1  # Single attempt per task (no retries)
DEFAULT_MAX_TURNS = 8     # Tool turns per attempt
REQUEST_TIMEOUT = 180.0   # 3 minutes per LLM call
STEP_TIMEOUT = 120.0      # 2 minutes per step (code execution)
MAX_CODE_LENGTH = 8000    # Safety limit for code/setup_code
MAX_EXPRESSIONS = 10      # Safety limit for get_variable_state
MIN_REWARD = 0.1          # Minimum reward for any submission
MAX_REWARD = 0.9999       # Maximum reward (avoid exact 1.0)

VALID_ACTION_TYPES = {
    "execute_snippet", "inspect_tensor", "get_variable_state",
    "run_training_probe", "inspect_diff", "submit_fix"
}


def clamp_reward(reward: float) -> float:
    """Clamp reward to (0, 1) exclusive - avoid exact 0.0 or 1.0."""
    if reward <= 0.0:
        return MIN_REWARD
    if reward >= 1.0:
        return MAX_REWARD
    return reward


# ── System Prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TOOLS = """You are an expert PyTorch debugging agent with access to debugging tools.
You receive a broken training script and must systematically debug and fix ALL bugs.

AVAILABLE TOOLS:
1. execute_snippet - Run a quick Python snippet to test hypotheses
2. inspect_tensor - Check tensor shape, dtype, gradients, NaN/Inf
3. get_variable_state - Inspect multiple variable values
4. run_training_probe - Run a few training steps to see loss curve
5. inspect_diff - Review your proposed changes before submitting
6. submit_fix - Submit your final fix (TERMINAL ACTION - ends episode)

RESPONSE FORMAT - You MUST respond with ONLY a valid JSON object:
{
  "reasoning": "Brief explanation of your analysis/decision",
  "action_type": "one of: execute_snippet|inspect_tensor|get_variable_state|run_training_probe|inspect_diff|submit_fix",
  "action_params": { <parameters for the chosen action> }
}

ACTION PARAMETERS:
- execute_snippet: {"code": "<python code>"}
- inspect_tensor: {"setup_code": "<python setup>", "target_expression": "<expr>"}
- get_variable_state: {"setup_code": "<python setup>", "expressions": ["<expr1>", "<expr2>"]}
- run_training_probe: {"code": "<full training script>", "steps": <1-10>}
- inspect_diff: {"proposed_code": "<your proposed fix>"}
- submit_fix: {"fixed_code": "<complete fixed Python script>"}

CRITICAL RULES:
1. Respond ONLY with valid JSON - no markdown, no explanation outside JSON
2. For submit_fix: fixed_code must be actual Python code (use \\n for newlines), NOT JSON
3. Fixed code must print: LOSSES:[v1, v2, ...] 
4. For task3: also print VAL_ACCS:[...] and FINAL_LOSS:X.XX
5. Keep torch.manual_seed() calls intact
6. ALWAYS submit_fix before running out of turns - never waste all turns on tools
7. If 2 or fewer turns remain, IMMEDIATELY call submit_fix with your best fix""".strip()

SYSTEM_PROMPT_SIMPLE = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs in it.

Rules:
- Return ONLY the complete corrected Python code, nothing else.
- No markdown, no backticks, no explanation text.
- The script must print losses in format: LOSSES:[v1, v2, ...]
- For task3, also print: VAL_ACCS:[v1,...] and FINAL_LOSS:X.XX
- Keep all torch.manual_seed() calls intact.""".strip()


# ── Logging Helpers ───────────────────────────────────────────────────────────

def log_start(task_id: str, env_name: str, model: str) -> None:
    """Emit [START] log for a task."""
    print(f"[START] task={task_id} env={env_name} model={model}", flush=True)


def log_step(step: int, action_summary: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Emit [STEP] log for a step."""
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_summary} reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    """Emit [END] log for a task."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── LLM Client ────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    """Initialize OpenAI-compatible client from environment variables."""
    api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY must be set in environment")
    
    return OpenAI(base_url=api_base, api_key=api_key, timeout=REQUEST_TIMEOUT)


def get_model_name() -> str:
    """Get model name from environment or use default."""
    return os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")


# ── Response Parsing ──────────────────────────────────────────────────────────

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from text."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```python"):
        text = text[9:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_agent_response(response: str) -> dict:
    """
    Parse the agent's JSON response robustly.
    Returns dict with reasoning, action_type, action_params.
    Falls back to treating response as code for submit_fix if parsing fails.
    """
    response = strip_markdown_fences(response)
    
    # Try direct JSON parse
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON object from response
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Fallback: treat entire response as code for submit_fix
    return {
        "reasoning": "Fallback: could not parse JSON, treating as code",
        "action_type": "submit_fix",
        "action_params": {"fixed_code": response}
    }


def clean_code_field(code: str) -> str:
    """Clean code field - remove markdown fences and validate it's not JSON."""
    code = strip_markdown_fences(code)
    # Detect if model returned nested JSON instead of Python
    if code.strip().startswith("{") and ('"action_type"' in code or '"reasoning"' in code):
        raise ValueError("Model returned nested JSON instead of Python code")
    return code


# ── Tool Result Formatting ────────────────────────────────────────────────────

def format_tool_result(obs: dict, action_type: str) -> str:
    """Format tool observation into a compact summary for the rolling history."""
    turn = obs.get("turn", 0)
    error = obs.get("error")
    
    if error:
        return f"[Turn {turn}] {action_type}: ERROR - {error[:200]}"
    
    if action_type == "execute_snippet":
        stdout = obs.get("stdout", "")[:800]
        stderr = obs.get("stderr", "")[:300]
        exit_code = obs.get("exit_code", 0)
        result = f"[Turn {turn}] execute_snippet (exit={exit_code}):\n{stdout}"
        if stderr:
            result += f"\nSTDERR: {stderr}"
        return result
    
    elif action_type == "inspect_tensor":
        parts = [f"[Turn {turn}] inspect_tensor:"]
        for key in ["shape", "dtype", "requires_grad", "grad_is_none"]:
            if obs.get(key) is not None:
                parts.append(f"  {key}: {obs[key]}")
        for key in ["min_val", "max_val", "mean_val"]:
            if obs.get(key) is not None:
                parts.append(f"  {key}: {obs[key]:.6f}")
        if obs.get("is_nan"):
            parts.append("  ⚠️ CONTAINS NaN")
        if obs.get("is_inf"):
            parts.append("  ⚠️ CONTAINS Inf")
        return "\n".join(parts)
    
    elif action_type == "run_training_probe":
        losses = obs.get("losses", [])[:10]
        final_loss = obs.get("final_loss")
        result = f"[Turn {turn}] run_training_probe:\n  losses: {losses}\n  final_loss: {final_loss}"
        if obs.get("loss_is_nan"):
            result += "\n  ⚠️ NaN LOSS"
        if obs.get("loss_is_inf"):
            result += "\n  ⚠️ Inf LOSS"
        return result
    
    elif action_type == "get_variable_state":
        results = obs.get("results", {})
        lines = [f"[Turn {turn}] get_variable_state:"]
        for expr, res in list(results.items())[:8]:
            if res.get("error"):
                lines.append(f"  {expr}: ERROR - {res['error'][:50]}")
            else:
                val = str(res.get("repr", res.get("value", "?")))[:80]
                lines.append(f"  {expr}: {val}")
        return "\n".join(lines)
    
    elif action_type == "inspect_diff":
        lines_changed = obs.get("lines_changed", 0)
        additions = obs.get("additions", 0)
        deletions = obs.get("deletions", 0)
        return f"[Turn {turn}] inspect_diff: {lines_changed} lines changed (+{additions}/-{deletions})"
    
    elif action_type == "submit_fix":
        reward = obs.get("reward", 0.0)
        return f"[Turn {turn}] submit_fix: reward={reward:.4f}"
    
    return f"[Turn {turn}] {action_type}: {json.dumps(obs, default=str)[:300]}"


# ── Environment Client ────────────────────────────────────────────────────────

class WhipStudioClient:
    """HTTP client for the WhipStudio environment."""
    
    def __init__(self, env_url: str):
        self.env_url = env_url.rstrip("/")
        self.timeout = httpx.Timeout(STEP_TIMEOUT, connect=10.0)
        self.episode_id = ""
    
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
            resp = client.post(f"{self.env_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            data = resp.json()
            obs = data.get("observation", data)
            self.episode_id = obs.get("episode_id", "")
            return obs
    
    def step(self, action: dict) -> dict:
        """Execute an action and get the result."""
        action["episode_id"] = self.episode_id
        payload = {"action": action}
        
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


# ── Action Building ───────────────────────────────────────────────────────────

def build_action(action_type: str, action_params: dict, buggy_code: str = "") -> dict:
    """
    Build action payload from parsed response.
    Applies safety limits and validation.
    """
    action = {"action_type": action_type}
    
    if action_type == "execute_snippet":
        code = action_params.get("code", "print('test')")
        action["code"] = clean_code_field(code)[:MAX_CODE_LENGTH]
    
    elif action_type == "inspect_tensor":
        setup = action_params.get("setup_code", "")[:MAX_CODE_LENGTH]
        action["setup_code"] = setup
        action["target_expression"] = action_params.get("target_expression", "")
    
    elif action_type == "run_training_probe":
        code = action_params.get("code", buggy_code)
        action["code"] = clean_code_field(code)[:MAX_CODE_LENGTH]
        action["steps"] = min(int(action_params.get("steps", 5)), 10)
    
    elif action_type == "get_variable_state":
        setup = action_params.get("setup_code", "")[:MAX_CODE_LENGTH]
        action["setup_code"] = setup
        expressions = action_params.get("expressions", [])
        action["expressions"] = expressions[:MAX_EXPRESSIONS]
    
    elif action_type == "inspect_diff":
        proposed = action_params.get("proposed_code", "")
        action["proposed_code"] = proposed
    
    elif action_type == "submit_fix":
        fixed_code = action_params.get("fixed_code", "")
        fixed_code = clean_code_field(fixed_code)
        if not fixed_code or len(fixed_code) < 20:
            raise ValueError("submit_fix received empty or too-short code")
        action["fixed_code"] = fixed_code
    
    else:
        raise ValueError(f"Unknown action_type: {action_type}")
    
    return action


# ── Agent Loop ────────────────────────────────────────────────────────────────

def run_tool_agent(
    env: WhipStudioClient,
    llm_client: OpenAI,
    model: str,
    task_id: str,
    obs: dict,
    max_turns: int = 8,
) -> tuple[float, str, list[float]]:
    """
    Run multi-turn tool-calling agent loop within a single episode.
    Returns (best_reward, best_fixed_code, step_rewards).
    """
    buggy_code = obs.get("buggy_code", "")
    task_description = obs.get("task_description", "")
    
    tool_history: list[str] = []  # Rolling context for LLM
    best_reward = 0.0
    best_code = ""
    step_rewards: list[float] = []  # Track all step rewards for [END] line
    
    for turn in range(1, max_turns + 1):
        turns_remaining = max_turns - turn
        
        # Build compact history (last 5 entries to control token usage)
        history_text = "\n\n".join(tool_history[-5:]) if tool_history else "No previous tool calls."
        
        # Urgency message for low turns
        urgency = ""
        if turns_remaining == 0:
            urgency = "\n⚠️ THIS IS YOUR LAST TURN! You MUST call submit_fix NOW."
        elif turns_remaining <= 2:
            urgency = f"\n⚠️ ONLY {turns_remaining} TURN(S) LEFT! Submit your fix soon!"
        
        prompt = f"""Task: {task_description}

Buggy Code:
```python
{buggy_code}
```

Turn {turn}/{max_turns} | Best reward: {best_reward:.2f} | Turns remaining: {turns_remaining}{urgency}

Tool History:
{history_text}

Analyze and decide your next action. Respond with JSON only.""".strip()
        
        # Get LLM response
        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_TOOLS},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            response_text = response.choices[0].message.content or ""
            parsed = parse_agent_response(response_text)
        except Exception as e:
            tool_history.append(f"[Turn {turn}] LLM ERROR: {str(e)[:100]}")
            continue
        
        action_type = parsed.get("action_type", "submit_fix")
        action_params = parsed.get("action_params", {})
        reasoning = parsed.get("reasoning", "")[:150]
        
        # Validate action_type
        if action_type not in VALID_ACTION_TYPES:
            action_type = "submit_fix"
            action_params = {"fixed_code": buggy_code}
        
        # Force submit_fix on last turn
        if turns_remaining == 0 and action_type != "submit_fix":
            action_type = "submit_fix"
            # Try to salvage code from params
            fixed = (
                action_params.get("fixed_code") or 
                action_params.get("code") or 
                action_params.get("proposed_code") or 
                buggy_code
            )
            action_params = {"fixed_code": fixed}
        
        # Build and execute action
        try:
            action = build_action(action_type, action_params, buggy_code)
            result = env.step(action)
        except ValueError as ve:
            tool_history.append(f"[Turn {turn}] BUILD ERROR: {str(ve)[:100]}")
            continue
        except Exception as e:
            tool_history.append(f"[Turn {turn}] API ERROR: {str(e)[:100]}")
            continue
        
        obs_result = result.get("observation", {})
        reward = float(result.get("reward", 0) or 0)
        done = result.get("done", False) or obs_result.get("episode_done", False)
        step_error = obs_result.get("error") or None
        
        # Format and store result
        tool_result = format_tool_result(obs_result, action_type)
        tool_history.append(f"Reasoning: {reasoning}\n{tool_result}")
        
        # Track reward for this step
        step_rewards.append(reward)
        
        # Log the step
        action_str = f"submit_fix(reward={reward:.2f})" if action_type == "submit_fix" else action_type
        log_step(turn, action_str, reward, done, step_error)
        
        # Track best
        if action_type == "submit_fix":
            if reward > best_reward:
                best_reward = reward
                best_code = action.get("fixed_code", "")
            
            if reward >= 0.95 or done:
                break
        
        if done:
            break
    
    return best_reward, best_code, step_rewards


def run_simple_agent(
    env: WhipStudioClient,
    llm_client: OpenAI,
    model: str,
    task_id: str,
    obs: dict,
) -> tuple[float, str, list[float]]:
    """
    Run simple submit-only agent (legacy mode).
    Returns (reward, fixed_code, step_rewards).
    """
    buggy_code = obs.get("buggy_code", "")
    task_description = obs.get("task_description", "")
    error_log = obs.get("error_log", "None")
    
    prompt = f"""Task: {task_description}

Buggy code:
{buggy_code}

Previous execution output (if any):
{error_log}""".strip()
    
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SIMPLE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        fixed_code = response.choices[0].message.content or ""
        fixed_code = strip_markdown_fences(fixed_code)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        return 0.0, "", []
    
    if not fixed_code.strip():
        return 0.0, "", []
    
    try:
        action = {"action_type": "submit_fix", "fixed_code": fixed_code}
        result = env.step(action)
        reward = float(result.get("reward", 0) or 0)
        obs_result = result.get("observation", {})
        done = result.get("done", False) or obs_result.get("episode_done", False)
        step_error = obs_result.get("error") or None
        log_step(1, f"submit_fix(reward={reward:.2f})", reward, done, step_error)
        return reward, fixed_code, [reward]
    except Exception as e:
        print(f"[ERROR] Step failed: {e}", file=sys.stderr)
        return 0.0, "", []


# ── Main Task Runner ──────────────────────────────────────────────────────────

def run_task(
    env: WhipStudioClient,
    llm_client: OpenAI,
    model: str,
    task_id: str,
    use_tools: bool = True,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> float:
    """
    Run inference on a single task with multiple attempts.
    Returns the best score achieved.
    """
    if isinstance(task_id, dict):
        task_id = task_id.get("id", str(task_id))
    
    log_start(task_id, "whipstudio", model)
    
    # Get task-specific config
    config = TASK_CONFIG.get(task_id, {"max_turns": max_turns})
    task_max_turns = min(max_turns, config.get("max_turns", max_turns))
    
    best_score = MIN_REWARD  # Start with minimum, not 0.0
    all_step_rewards: list[float] = []
    
    for attempt in range(1, MAX_ATTEMPTS_PER_TASK + 1):
        try:
            obs = env.reset(task_id)
        except Exception as e:
            continue
        
        if use_tools:
            reward, _, step_rewards = run_tool_agent(env, llm_client, model, task_id, obs, task_max_turns)
        else:
            reward, _, step_rewards = run_simple_agent(env, llm_client, model, task_id, obs)
        
        all_step_rewards.extend(step_rewards)
        
        # Clamp reward to avoid exact 0.0 or 1.0
        reward = clamp_reward(reward)
        
        if reward > best_score:
            best_score = reward
    
    success = best_score >= 0.7
    log_end(success, len(all_step_rewards), best_score, all_step_rewards)
    return best_score


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WhipStudio inference script for OpenEnv Hackathon"
    )
    parser.add_argument(
        "--env-url",
        default=os.environ.get("ENV_URL", "https://amogh-kal1-whipstudio.hf.space"),
        help="URL of the WhipStudio environment"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all tasks)"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool use (simple submit-only mode)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns per attempt (default: {DEFAULT_MAX_TURNS})"
    )
    args = parser.parse_args()
    
    use_tools = not args.no_tools
    
    # Initialize clients (all info to stderr — stdout is reserved for [START]/[STEP]/[END])
    print(f"[INFO] Connecting to environment at {args.env_url}", file=sys.stderr, flush=True)
    print(f"[INFO] Mode: {'tool-calling agent' if use_tools else 'simple submit-only'}", file=sys.stderr, flush=True)
    
    env = WhipStudioClient(args.env_url)
    
    if not env.health_check():
        sys.exit(1)
    
    print("[INFO] Environment is reachable", file=sys.stderr, flush=True)
    
    llm_client = get_openai_client()
    model = get_model_name()
    print(f"[INFO] Using model: {model}", file=sys.stderr, flush=True)
    
    # Determine tasks
    task_ids = args.tasks if args.tasks else env.get_tasks()
    print(f"[INFO] Running tasks: {task_ids}", file=sys.stderr, flush=True)
    
    # Run inference
    start_time = time.time()
    scores = {}
    
    for task_id in task_ids:
        task_start = time.time()
        score = run_task(env, llm_client, model, task_id, use_tools, args.max_turns)
        scores[task_id] = score
        elapsed = time.time() - task_start
        print(f"[INFO] {task_id} completed in {elapsed:.1f}s with score {score:.4f}", file=sys.stderr, flush=True)
    
    # Summary (to stderr)
    total_elapsed = time.time() - start_time
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    
    print(f"[INFO] Tasks completed: {len(scores)}, Total time: {total_elapsed:.1f}s, Average score: {avg_score:.4f}", file=sys.stderr, flush=True)
    for tid, score in scores.items():
        status = "✓" if score >= 0.7 else "○"
        print(f"[INFO]   {status} {tid}: {score:.4f}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
