"""
WhipStudio Environment with Agent Tools

This module implements the ML debugging environment with multiple tool-calling
capabilities for agents to debug code step-by-step before submitting a fix.
"""

import ast
import difflib
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MLDebugAction, MLDebugObservation
    from .sandbox import SAFE_ENV, execute_code, strip_markdown_code
    from .tasks import (
        task1_broken_loop,
        task2_nan_loss,
        task3_oom_leakage,
        task4_wrong_loss,
        task5_frozen_backbone,
        task6_io_mismatch,
    )
    from .tasks.graders import RunResult, parse_losses, parse_val_accs, score_task
except ImportError:
    from models import MLDebugAction, MLDebugObservation
    from server.sandbox import SAFE_ENV, execute_code, strip_markdown_code
    from server.tasks import (
        task1_broken_loop,
        task2_nan_loss,
        task3_oom_leakage,
        task4_wrong_loss,
        task5_frozen_backbone,
        task6_io_mismatch,
    )
    from server.tasks.graders import parse_losses, parse_val_accs, score_task


# ── Task Registry ──────────────────────────────────────────────────────────

TASKS = {
    "task1": task1_broken_loop,
    "task2": task2_nan_loss,
    "task3": task3_oom_leakage,
    "task4": task4_wrong_loss,
    "task5": task5_frozen_backbone,
    "task6": task6_io_mismatch,
}

# ── Configuration ──────────────────────────────────────────────────────────

MAX_TURNS_PER_EPISODE = int(os.environ.get("MAX_TURNS_PER_EPISODE", "10"))
TOOL_TIMEOUT_SECONDS = 30  # Increased for PyTorch initialization time
MAX_OUTPUT_BYTES = 50000  # Increased to show complete outputs

# Allowed imports for sandboxed execution
ALLOWED_PACKAGES = {
    "torch", "numpy", "sklearn", "pandas", "matplotlib", "scipy",
    "math", "random", "os", "sys", "collections", "itertools",
    "functools", "json", "re", "typing", "copy", "dataclasses",
    "torch.nn", "torch.optim", "torch.utils", "torch.utils.data",
    "numpy.random", "sklearn.datasets", "sklearn.model_selection",
}

# Banned imports for security
BANNED_IMPORTS = {"socket", "requests", "httpx", "urllib", "subprocess", "shutil"}


# ── Security Helpers ───────────────────────────────────────────────────────

def check_code_security(code: str) -> tuple[bool, str]:
    """
    Analyze code for security violations using AST.
    Returns (is_safe, error_message).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True, ""  # Let it fail at runtime with proper error
    
    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in BANNED_IMPORTS:
                    return False, f"Import of '{alias.name}' is not allowed (network/system access)"
        
        if isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in BANNED_IMPORTS:
                    return False, f"Import from '{node.module}' is not allowed (network/system access)"
        
        # Check file writes outside /tmp
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                if len(node.args) >= 1:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        path = first_arg.value
                        if not path.startswith("/tmp") and not path.startswith("tmp"):
                            # Check mode argument
                            mode = "r"
                            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                                mode = str(node.args[1].value)
                            for kw in node.keywords:
                                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                    mode = str(kw.value.value)
                            if "w" in mode or "a" in mode or "+" in mode:
                                return False, f"File writes outside /tmp are not allowed: {path}"
    
    return True, ""


def run_sandboxed_code(code: str, timeout: int = TOOL_TIMEOUT_SECONDS) -> dict:
    """
    Run code in a sandboxed subprocess with timeout and security constraints.
    Returns dict with stdout, stderr, exit_code, timed_out.
    """
    # Security check
    is_safe, error = check_code_security(code)
    if not is_safe:
        return {
            "stdout": "",
            "stderr": f"Security violation: {error}",
            "exit_code": -1,
            "timed_out": False,
        }
    
    # Clean up markdown
    cleaned_code = strip_markdown_code(code)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(cleaned_code)
        tmp_path = f.name
    
    start = time.time()
    try:
        # Run with restricted environment
        env = dict(SAFE_ENV)
        env["no_proxy"] = "*"  # Disable network
        
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd="/tmp",
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "timed_out": False,
            "elapsed": round(time.time() - start, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "exit_code": -1,
            "timed_out": True,
            "elapsed": timeout,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Tool Implementations ───────────────────────────────────────────────────

def tool_execute_snippet(code: str, turn: int) -> MLDebugObservation:
    """Execute a Python snippet and return stdout/stderr/exit_code."""
    result = run_sandboxed_code(code, timeout=TOOL_TIMEOUT_SECONDS)
    
    return MLDebugObservation(
        action_type="execute_snippet",
        turn=turn,
        episode_done=False,
        reward=None,
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        timed_out=result["timed_out"],
    )


def tool_inspect_tensor(setup_code: str, target_expression: str, turn: int) -> MLDebugObservation:
    """
    Inspect a tensor or module parameter.
    Returns shape, dtype, requires_grad, grad status, min/max/mean, nan/inf checks.
    """
    # Validate target_expression is a simple expression (no newlines, reasonable length)
    if not target_expression or not target_expression.strip():
        return MLDebugObservation(
            action_type="inspect_tensor",
            turn=turn,
            episode_done=False,
            reward=None,
            error="target_expression is required",
        )
    
    # Clean up target expression - strip whitespace, remove newlines
    target_expression = target_expression.strip().replace('\n', ' ').replace('\r', '')
    
    if len(target_expression) > 500:
        return MLDebugObservation(
            action_type="inspect_tensor",
            turn=turn,
            episode_done=False,
            reward=None,
            error="target_expression too long (max 500 chars)",
        )
    
    # Build inspection script - use exec/eval for safer expression handling
    inspection_code = f'''{setup_code}

import torch
import json
import math

def _inspect_target():
    try:
        # Evaluate target expression in the current scope
        _target = eval({repr(target_expression)})
    except Exception as e:
        return {{"error": "Failed to evaluate expression: " + str(e)}}
    
    result = {{}}
    
    # Check if it's a tensor
    if isinstance(_target, torch.Tensor):
        result["shape"] = list(_target.shape)
        result["dtype"] = str(_target.dtype)
        result["requires_grad"] = _target.requires_grad
        result["grad_is_none"] = _target.grad is None if _target.requires_grad else None
        
        # Convert to float for stats if needed
        try:
            _data = _target.detach().float()
            result["min_val"] = float(_data.min().item())
            result["max_val"] = float(_data.max().item())
            result["mean_val"] = float(_data.mean().item())
            result["is_nan"] = bool(torch.isnan(_data).any().item())
            result["is_inf"] = bool(torch.isinf(_data).any().item())
        except Exception as e:
            result["stats_error"] = str(e)
    elif hasattr(_target, 'weight'):
        # It's likely a module
        result["is_module"] = True
        result["type"] = type(_target).__name__
        if hasattr(_target.weight, 'shape'):
            result["shape"] = list(_target.weight.shape)
            result["dtype"] = str(_target.weight.dtype)
            result["requires_grad"] = _target.weight.requires_grad
            result["grad_is_none"] = _target.weight.grad is None if _target.weight.requires_grad else None
    else:
        result["error"] = "Target is not a tensor or module: " + type(_target).__name__
    
    return result

_result = _inspect_target()
print("##INSPECT_RESULT##")
print(json.dumps(_result))
print("##END_INSPECT##")
'''
    
    result = run_sandboxed_code(inspection_code, timeout=TOOL_TIMEOUT_SECONDS)
    
    obs = MLDebugObservation(
        action_type="inspect_tensor",
        turn=turn,
        episode_done=False,
        reward=None,
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        timed_out=result["timed_out"],
    )
    
    # Parse the result
    if "##INSPECT_RESULT##" in result["stdout"]:
        try:
            match = re.search(r"##INSPECT_RESULT##\s*(\{.*?\})\s*##END_INSPECT##", result["stdout"], re.DOTALL)
            if match:
                import json
                data = json.loads(match.group(1))
                if "error" in data:
                    obs.error = data["error"]
                else:
                    obs.shape = data.get("shape")
                    obs.dtype = data.get("dtype")
                    obs.requires_grad = data.get("requires_grad")
                    obs.grad_is_none = data.get("grad_is_none")
                    obs.min_val = data.get("min_val")
                    obs.max_val = data.get("max_val")
                    obs.mean_val = data.get("mean_val")
                    obs.is_nan = data.get("is_nan")
                    obs.is_inf = data.get("is_inf")
        except Exception as e:
            obs.error = f"Failed to parse inspection result: {e}"
    elif result["stderr"]:
        obs.error = result["stderr"][:500]
    
    return obs


def tool_run_training_probe(code: str, steps: int, turn: int) -> MLDebugObservation:
    """
    Run N steps of training and return loss curve + gradient norms.
    """
    # Cap steps at 10
    steps = min(steps, 10)
    
    # Wrap the training code to capture metrics
    probe_code = f'''
import torch
import torch.nn as nn
import json
import math

# Monkey-patch to capture gradient norms
_grad_norms = {{}}
_losses = []
_step_count = 0
_max_steps = {steps}

_original_backward = torch.Tensor.backward

def _patched_backward(self, *args, **kwargs):
    global _step_count, _losses
    result = _original_backward(self, *args, **kwargs)
    
    if _step_count < _max_steps:
        try:
            loss_val = self.item()
            _losses.append(loss_val)
        except:
            pass
        _step_count += 1
    
    return result

torch.Tensor.backward = _patched_backward

# Run the user's code
try:
    exec("""
{code.replace(chr(34)*3, chr(39)*3)}
""")
except Exception as e:
    print(f"EXECUTION_ERROR: {{e}}")

# Restore backward
torch.Tensor.backward = _original_backward

# Try to find model and optimizer in globals
_model = None
_optimizer = None
for _name, _obj in list(globals().items()):
    if isinstance(_obj, nn.Module) and _model is None:
        _model = _obj
    if hasattr(_obj, 'param_groups') and _optimizer is None:
        _optimizer = _obj

# Capture gradient norms
if _model is not None:
    for name, param in _model.named_parameters():
        if param.requires_grad and param.grad is not None:
            _grad_norms[name] = float(param.grad.norm().item())

# Capture optimizer param count
_opt_count = None
if _optimizer is not None:
    _opt_count = sum(p.numel() for g in _optimizer.param_groups for p in g['params'])

# Output results
print("##PROBE_RESULT##")
print(json.dumps({{
    "losses": _losses[:_max_steps],
    "grad_norms": _grad_norms,
    "optimizer_param_count": _opt_count,
    "final_loss": _losses[-1] if _losses else None,
    "loss_is_nan": any(math.isnan(l) if isinstance(l, float) else False for l in _losses),
    "loss_is_inf": any(math.isinf(l) if isinstance(l, float) else False for l in _losses),
}}))
print("##END_PROBE##")
'''
    
    result = run_sandboxed_code(probe_code, timeout=TOOL_TIMEOUT_SECONDS)
    
    obs = MLDebugObservation(
        action_type="run_training_probe",
        turn=turn,
        episode_done=False,
        reward=None,
        stdout=result["stdout"],
        stderr=result["stderr"],
        timed_out=result["timed_out"],
    )
    
    # Parse the probe result
    if "##PROBE_RESULT##" in result["stdout"]:
        try:
            match = re.search(r"##PROBE_RESULT##\s*(\{.*?\})\s*##END_PROBE##", result["stdout"], re.DOTALL)
            if match:
                import json
                data = json.loads(match.group(1))
                obs.losses = data.get("losses", [])
                obs.grad_norms = data.get("grad_norms", {})
                obs.optimizer_param_count = data.get("optimizer_param_count")
                obs.final_loss = data.get("final_loss")
                obs.loss_is_nan = data.get("loss_is_nan", False)
                obs.loss_is_inf = data.get("loss_is_inf", False)
        except Exception as e:
            obs.error = f"Failed to parse probe result: {e}"
    
    return obs


def tool_get_variable_state(setup_code: str, expressions: list[str], turn: int) -> MLDebugObservation:
    """
    Evaluate multiple expressions and return their repr, type, value, shape.
    """
    # Limit expressions
    expressions = expressions[:10]
    
    expr_list_str = repr(expressions)
    
    eval_code = f'''
{setup_code}

import torch
import json

_expressions = {expr_list_str}
_results = {{}}

for _expr in _expressions:
    try:
        _val = eval(_expr)
        _result = {{
            "repr": repr(_val)[:500],
            "type": type(_val).__name__,
            "value": None,
            "shape": None,
            "error": None,
        }}
        
        # Extract scalar value
        if isinstance(_val, (int, float, bool)):
            _result["value"] = _val
        elif isinstance(_val, str):
            _result["value"] = _val[:200]
        elif hasattr(_val, 'item') and _val.numel() == 1:
            _result["value"] = float(_val.item())
        
        # Extract shape
        if hasattr(_val, 'shape'):
            _result["shape"] = list(_val.shape)
        elif hasattr(_val, '__len__') and not isinstance(_val, str):
            _result["shape"] = [len(_val)]
        
        _results[_expr] = _result
    except Exception as e:
        _results[_expr] = {{
            "repr": "",
            "type": "",
            "value": None,
            "shape": None,
            "error": str(e)[:200],
        }}

print("##VAR_RESULT##")
print(json.dumps(_results))
print("##END_VAR##")
'''
    
    result = run_sandboxed_code(eval_code, timeout=TOOL_TIMEOUT_SECONDS)
    
    obs = MLDebugObservation(
        action_type="get_variable_state",
        turn=turn,
        episode_done=False,
        reward=None,
        stdout=result["stdout"],
        stderr=result["stderr"],
        timed_out=result["timed_out"],
    )
    
    # Parse results
    if "##VAR_RESULT##" in result["stdout"]:
        try:
            match = re.search(r"##VAR_RESULT##\s*(\{.*?\})\s*##END_VAR##", result["stdout"], re.DOTALL)
            if match:
                import json
                obs.results = json.loads(match.group(1))
        except Exception as e:
            obs.error = f"Failed to parse variable results: {e}"
    
    return obs


def tool_inspect_diff(original_code: str, proposed_code: str, turn: int) -> MLDebugObservation:
    """
    Generate a unified diff between original and proposed code.
    """
    original_lines = original_code.strip().splitlines(keepends=True)
    proposed_lines = proposed_code.strip().splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        original_lines,
        proposed_lines,
        fromfile="original.py",
        tofile="proposed.py",
        lineterm="",
    ))
    
    # Count changes
    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    lines_changed = additions + deletions
    
    return MLDebugObservation(
        action_type="inspect_diff",
        turn=turn,
        episode_done=False,
        reward=None,
        diff="\n".join(diff),
        lines_changed=lines_changed,
        additions=additions,
        deletions=deletions,
    )


# ── Global Session Store ───────────────────────────────────────────────────
# OpenEnv creates a new environment instance for each HTTP request, so we need
# a global store to track state (step_count, submitted status) across requests.

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SessionState:
    """State for a single episode session."""
    episode_id: str
    task_id: str
    step_count: int = 0
    submitted: bool = False
    best_reward: float = 0.0
    original_buggy_code: str = ""
    trajectory: list = field(default_factory=list)

# Global session store keyed by episode_id
_SESSION_STORE: Dict[str, SessionState] = {}


def _get_session(episode_id: str) -> Optional[SessionState]:
    """Get session state for an episode, or None if not found."""
    return _SESSION_STORE.get(episode_id)


def _create_session(episode_id: str, task_id: str, original_buggy_code: str) -> SessionState:
    """Create a new session and store it."""
    session = SessionState(
        episode_id=episode_id,
        task_id=task_id,
        original_buggy_code=original_buggy_code,
    )
    _SESSION_STORE[episode_id] = session
    return session


def _clear_session(episode_id: str) -> None:
    """Remove a session from the store."""
    _SESSION_STORE.pop(episode_id, None)


# ── Main Environment ───────────────────────────────────────────────────────

class MLDebugEnvironment(Environment):
    """
    ML Debug Environment with global session store.
    
    OpenEnv creates a new instance for each HTTP request, so state is tracked
    in a global session store keyed by episode_id.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        # Instance state (transient per request)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self, task_id: str = "task1", **kwargs) -> MLDebugObservation:
        if task_id not in TASKS:
            task_id = "task1"

        # Create new episode ID and session
        episode_id = str(uuid4())
        task = TASKS[task_id]
        original_buggy_code = task.BUGGY_CODE.strip()
        
        # Create session in global store
        _create_session(episode_id, task_id, original_buggy_code)
        
        self._state = State(episode_id=episode_id, step_count=0)

        return MLDebugObservation(
            action_type="reset",
            task_id=task_id,
            task_description=task.TASK_DESCRIPTION.strip(),
            buggy_code=original_buggy_code,
            error_log="",
            last_reward=0.0,
            metrics={},
            done=False,
            episode_done=False,
            reward=None,
            turn=0,
            episode_id=episode_id,
        )

    def step(self, action: MLDebugAction) -> MLDebugObservation:
        # Get episode_id from action (passed from client)
        episode_id = getattr(action, 'episode_id', None) or ""
        
        # Look up session in global store
        session = _get_session(episode_id) if episode_id else None
        
        if session is None:
            # No valid session - return error
            return MLDebugObservation(
                action_type=action.action_type,
                error=f"No active session for episode_id '{episode_id}'. Call reset() first.",
                episode_done=True,
                done=True,
                reward=0.001,
                turn=0,
                task_id="task1",
                task_description="",
                buggy_code="",
                episode_id=episode_id or "",
            )
        
        # Check if episode already complete
        if session.submitted:
            return MLDebugObservation(
                action_type=action.action_type,
                error="Episode already complete. Call reset() to start a new episode.",
                episode_done=True,
                done=True,
                reward=0.001,
                turn=session.step_count,
                task_id=session.task_id,
                task_description=TASKS[session.task_id].TASK_DESCRIPTION.strip(),
                buggy_code=session.original_buggy_code,
                episode_id=episode_id,
            )
        
        # Increment turn in session store
        session.step_count += 1
        current_turn = session.step_count
        
        # Check max turns
        if current_turn > MAX_TURNS_PER_EPISODE:
            session.submitted = True
            return MLDebugObservation(
                action_type=action.action_type,
                error=f"Maximum turns ({MAX_TURNS_PER_EPISODE}) exceeded. Episode terminated.",
                episode_done=True,
                done=True,
                reward=0.001,
                turn=current_turn,
                task_id=session.task_id,
                task_description=TASKS[session.task_id].TASK_DESCRIPTION.strip(),
                buggy_code=session.original_buggy_code,
                episode_id=episode_id,
            )
        
        # Dispatch based on action type
        action_type = action.action_type
        task_id = session.task_id
        original_buggy_code = session.original_buggy_code
        
        if action_type == "execute_snippet":
            obs = tool_execute_snippet(action.code, current_turn)
            obs.task_id = task_id
            obs.task_description = TASKS[task_id].TASK_DESCRIPTION.strip()
            obs.buggy_code = original_buggy_code
            obs.episode_id = episode_id
            return obs
        
        elif action_type == "inspect_tensor":
            obs = tool_inspect_tensor(action.setup_code, action.target_expression, current_turn)
            obs.task_id = task_id
            obs.task_description = TASKS[task_id].TASK_DESCRIPTION.strip()
            obs.buggy_code = original_buggy_code
            obs.episode_id = episode_id
            return obs
        
        elif action_type == "run_training_probe":
            obs = tool_run_training_probe(action.code, action.steps, current_turn)
            obs.task_id = task_id
            obs.task_description = TASKS[task_id].TASK_DESCRIPTION.strip()
            obs.buggy_code = original_buggy_code
            obs.episode_id = episode_id
            return obs
        
        elif action_type == "get_variable_state":
            obs = tool_get_variable_state(action.setup_code, action.expressions, current_turn)
            obs.task_id = task_id
            obs.task_description = TASKS[task_id].TASK_DESCRIPTION.strip()
            obs.buggy_code = original_buggy_code
            obs.episode_id = episode_id
            return obs
        
        elif action_type == "inspect_diff":
            obs = tool_inspect_diff(
                original_buggy_code,
                action.proposed_code,
                current_turn,
            )
            obs.task_id = task_id
            obs.task_description = TASKS[task_id].TASK_DESCRIPTION.strip()
            obs.buggy_code = original_buggy_code
            obs.episode_id = episode_id
            return obs
        
        elif action_type == "submit_fix":
            return self._handle_submit_fix(action, session)
        
        else:
            # Unknown action type - treat as submit_fix for backward compat
            if action.fixed_code:
                return self._handle_submit_fix(action, session)
            else:
                return MLDebugObservation(
                    action_type=action_type,
                    error=f"Unknown action type: {action_type}",
                    episode_done=False,
                    reward=None,
                    turn=current_turn,
                    task_id=task_id,
                    task_description=TASKS[task_id].TASK_DESCRIPTION.strip(),
                    buggy_code=original_buggy_code,
                    episode_id=episode_id,
                )
    
    def _handle_submit_fix(self, action: MLDebugAction, session: SessionState) -> MLDebugObservation:
        """Handle the submit_fix action (terminal action)."""
        
        # Mark as submitted in session
        session.submitted = True
        turns_used = session.step_count
        episode_id = session.episode_id
        task_id = session.task_id
        original_buggy_code = session.original_buggy_code
        
        # Get the fixed code
        fixed_code = action.fixed_code or action.code or ""
        
        if not fixed_code or not fixed_code.strip():
            return MLDebugObservation(
                action_type="submit_fix",
                turn=turns_used,
                episode_done=True,
                done=True,
                reward=0.001,
                success=False,
                grader_details={"error": "empty code submitted"},
                turns_used=turns_used,
                task_id=task_id,
                task_description=TASKS[task_id].TASK_DESCRIPTION.strip(),
                buggy_code=original_buggy_code,
                error_log="empty code submitted",
                error="Empty code submitted",
                episode_id=episode_id,
            )
        
        # Execute and grade
        run_result1 = execute_code(fixed_code)
        reward1, breakdown1 = score_task(task_id, run_result1)

        # Consistency check for high scores
        consistency_flag = False
        reward_variance = 0.0
        final_reward = reward1
        final_breakdown = breakdown1
        run_result = run_result1

        if reward1 > 0.5:
            run_result2 = execute_code(fixed_code)
            reward2, breakdown2 = score_task(task_id, run_result2)
            reward_variance = abs(reward1 - reward2)
            if reward_variance > 0.15:
                consistency_flag = True
                final_reward = min(reward1, reward2)
                if reward2 < reward1:
                    final_breakdown = breakdown2
                    run_result = run_result2
            else:
                consistency_flag = False
                final_reward = (reward1 + reward2) / 2.0

        session.best_reward = max(session.best_reward, final_reward)

        # Parse metrics
        losses = parse_losses(run_result.stdout)
        val_accs = parse_val_accs(run_result.stdout)
        final_loss = None
        if losses:
            final_loss = losses[-1]
        else:
            match = re.search(r"FINAL_LOSS:([-\d.]+)", run_result.stdout)
            if match:
                final_loss = float(match.group(1))

        metrics = {
            "exit_code": run_result.exit_code,
            "elapsed_seconds": run_result.elapsed_seconds,
            "timed_out": run_result.timed_out,
            "step": session.step_count,
            "best_reward_so_far": session.best_reward,
            "final_loss": final_loss,
            "nan_count": sum(1 for x in losses if math.isnan(x) or math.isinf(x)) if losses else 0,
            "val_acc": val_accs[-1] if val_accs else None,
            "consistency_flag": consistency_flag,
            "reward_variance": round(reward_variance, 4),
            "reward_breakdown": final_breakdown,
        }

        session.trajectory.append({
            "step": session.step_count,
            "reward": final_reward,
            "best_reward": session.best_reward,
            "metrics": metrics,
            "done": True,
            "timestamp": time.time(),
        })

        task = TASKS[task_id]

        # Return complete outputs without truncation
        return MLDebugObservation(
            action_type="submit_fix",
            turn=turns_used,
            episode_done=True,
            done=True,
            reward=final_reward,
            success=final_reward >= 0.7,
            grader_details=final_breakdown,
            turns_used=turns_used,
            task_id=task_id,
            task_description=task.TASK_DESCRIPTION.strip(),
            buggy_code=original_buggy_code,
            error_log=(run_result.stdout + "\n" + run_result.stderr).strip(),
            last_reward=final_reward,
            metrics=metrics,
            stdout=run_result.stdout,
            stderr=run_result.stderr,
            exit_code=run_result.exit_code,
            timed_out=run_result.timed_out,
            losses=losses if losses else [],
            final_loss=final_loss,
            episode_id=episode_id,
        )

    @property
    def trajectory(self) -> list[dict]:
        return list(self._trajectory)

    @property
    def state(self) -> State:
        return self._state


# ── Tool Definitions for /tools endpoint ───────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "execute_snippet",
        "description": "Execute a short Python code snippet and return stdout, stderr, and exit code. Use for quick probes and checks.",
        "action_fields": {
            "action_type": "execute_snippet",
            "code": "str - The Python snippet to run",
        },
        "observation_fields": ["stdout", "stderr", "exit_code", "timed_out", "turn"],
        "constraints": {
            "timeout": "10 seconds",
            "network": "disabled",
            "file_writes": "/tmp only",
        },
    },
    {
        "name": "inspect_tensor",
        "description": "Inspect a tensor or module parameter. Returns shape, dtype, requires_grad, grad status, min/max/mean, and NaN/Inf checks.",
        "action_fields": {
            "action_type": "inspect_tensor",
            "setup_code": "str - Code that defines the tensor/module (imports + definitions)",
            "target_expression": "str - The expression to inspect (e.g., 'model.weight.grad')",
        },
        "observation_fields": ["shape", "dtype", "requires_grad", "grad_is_none", "min_val", "max_val", "mean_val", "is_nan", "is_inf", "error", "turn"],
    },
    {
        "name": "run_training_probe",
        "description": "Run N steps of training and return loss curve + gradient norms. Use to verify if a fix is working before submission.",
        "action_fields": {
            "action_type": "run_training_probe",
            "code": "str - The full training script to probe",
            "steps": "int - Number of training steps (1-10, default 5)",
        },
        "observation_fields": ["losses", "grad_norms", "optimizer_param_count", "final_loss", "loss_is_nan", "loss_is_inf", "stderr", "timed_out", "turn"],
    },
    {
        "name": "get_variable_state",
        "description": "Evaluate multiple expressions and return their repr, type, value, and shape. Useful for checking model.training, optimizer params, etc.",
        "action_fields": {
            "action_type": "get_variable_state",
            "setup_code": "str - Setup code (imports + definitions)",
            "expressions": "list[str] - List of expressions to evaluate (max 10)",
        },
        "observation_fields": ["results", "turn"],
    },
    {
        "name": "inspect_diff",
        "description": "Generate a unified diff between the original buggy code and your proposed fix. Review your changes before submitting.",
        "action_fields": {
            "action_type": "inspect_diff",
            "proposed_code": "str - Your proposed fix",
        },
        "observation_fields": ["diff", "lines_changed", "additions", "deletions", "turn"],
    },
    {
        "name": "submit_fix",
        "description": "Submit your final fix. This is the terminal action that ends the episode and returns the grader score.",
        "action_fields": {
            "action_type": "submit_fix",
            "fixed_code": "str - The complete fixed Python script",
        },
        "observation_fields": ["reward", "success", "grader_details", "turns_used", "episode_done", "turn"],
        "terminal": True,
    },
]
