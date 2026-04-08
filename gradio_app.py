"""
WhipStudio — ML Debug Arena
A polished Gradio UI for the ML Debugging RL environment.
Provides code editing, loss curve visualization, diff views, tool calling, and episode history.
"""
import difflib
import json
import math
import re
from typing import Any

import gradio as gr
import httpx
import os

DEFAULT_BASE_URL = os.environ.get("BASE_URL", "http://localhost:7860")

# ── Task metadata ──────────────────────────────────────────────────────────

TASK_INFO = {
    "task1": {
        "name": "Broken Training Loop",
        "difficulty": "🟢 Easy",
        "description": "Fix optimizer order + learning rate bugs in a linear classifier.",
        "hints": "Look at optimizer.step() / loss.backward() order and the learning rate.",
    },
    "task2": {
        "name": "Silent NaN Loss",
        "difficulty": "🟡 Medium",
        "description": "Fix numerical instability causing NaN loss from log(0).",
        "hints": "The loss computation uses torch.log() without clamping — pred can be 0.",
    },
    "task3": {
        "name": "OOM + Data Leakage",
        "difficulty": "🔴 Hard",
        "description": "Fix memory leak (graph accumulation) AND train/val data leakage.",
        "hints": "Two bugs: total_loss accumulates graph, and augmentation is applied before split.",
    },
    "task4": {
        "name": "Wrong Loss Function",
        "difficulty": "🟡 Medium",
        "description": "Multi-label classification incorrectly using CrossEntropyLoss. Fix loss and eval.",
        "hints": "Use BCEWithLogitsLoss for multi-label. Ensure predictions are multi-hot.",
    },
    "task5": {
        "name": "Frozen Backbone",
        "difficulty": "🟡 Medium",
        "description": "Backbone frozen but its parameters are passed to the optimizer.",
        "hints": "Unfreeze backend or only pass head parameters to Adam.",
    },
    "task6": {
        "name": "Input-Output Mismatch",
        "difficulty": "🔴 Hard",
        "description": "CNN has 4 bugs: shape mismatch, channel order (HWC/CHW), label encoding, batch dimension.",
        "hints": "Fix image size (32→28), permute HWC→CHW, use class indices not one-hot, add unsqueeze(0).",
    },
}

# ── Tool metadata ──────────────────────────────────────────────────────────

TOOL_INFO = {
    "execute_snippet": {
        "name": "Execute Snippet",
        "icon": "▶️",
        "description": "Run a Python code snippet and see stdout/stderr.",
        "fields": ["code"],
    },
    "inspect_tensor": {
        "name": "Inspect Tensor",
        "icon": "🔬",
        "description": "Inspect tensor shape, dtype, gradients, and stats.",
        "fields": ["setup_code", "target_expression"],
    },
    "run_training_probe": {
        "name": "Training Probe",
        "icon": "📈",
        "description": "Run N training steps and see loss curve + gradient norms.",
        "fields": ["code", "steps"],
    },
    "get_variable_state": {
        "name": "Variable State",
        "icon": "🔎",
        "description": "Evaluate expressions and inspect their values.",
        "fields": ["setup_code", "expressions"],
    },
    "inspect_diff": {
        "name": "Inspect Diff",
        "icon": "📝",
        "description": "Compare your fix against the original buggy code.",
        "fields": ["proposed_code"],
    },
}

# ── Theme ──────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Dark arena theme */
.gradio-container { max-width: 1400px !important; }

/* Score display */
.score-high { color: #22c55e !important; font-weight: 700; }
.score-med  { color: #eab308 !important; font-weight: 700; }
.score-low  { color: #ef4444 !important; font-weight: 700; }

/* Task badges */
.task-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; }
.badge-easy { background: #16a34a22; color: #22c55e; border: 1px solid #22c55e44; }
.badge-medium { background: #ca8a0422; color: #eab308; border: 1px solid #eab30844; }
.badge-hard { background: #dc262622; color: #ef4444; border: 1px solid #ef444444; }

/* Diff highlighting */
.diff-add { color: #22c55e; background: #22c55e11; }
.diff-del { color: #ef4444; background: #ef444411; }

/* Score bar */
.score-bar-container { position: relative; height: 28px; background: #1e293b; border-radius: 6px; overflow: hidden; margin: 8px 0; }
.score-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease-in-out; }
.score-bar-label { position: absolute; right: 8px; top: 4px; font-weight: 600; font-size: 0.9em; }

/* Episode step indicators */
.step-indicator { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 16px; font-size: 0.85em; margin: 2px; }
.step-done { background: #16a34a22; border: 1px solid #22c55e44; color: #22c55e; }
.step-active { background: #3b82f622; border: 1px solid #3b82f644; color: #60a5fa; }
.step-pending { background: #334155; border: 1px solid #47556966; color: #94a3b8; }

/* Header styling */
.arena-header { text-align: center; padding: 12px 0; }
.arena-header h1 { margin: 0; font-size: 1.8em; }
.arena-header p { margin: 4px 0 0 0; color: #94a3b8; }

/* Tool panel styling */
.tool-panel { background: #1e293b; border-radius: 8px; padding: 12px; margin: 8px 0; }
.tool-result { background: #0f172a; border-radius: 6px; padding: 10px; font-family: monospace; font-size: 0.85em; }
.tool-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; background: #3b82f622; color: #60a5fa; border: 1px solid #3b82f644; }
"""


# ── API helpers ────────────────────────────────────────────────────────────

def _api(base_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    """Call the WhipStudio API and return parsed JSON."""
    base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    url = f"{base_url}{path}"
    try:
        with httpx.Client(timeout=120.0) as client:
            if method == "GET":
                resp = client.get(url)
            else:
                resp = client.post(url, json=payload or {})
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _parse_losses_from_log(log: str) -> list[float]:
    """Extract LOSSES:[...] from stdout."""
    match = re.search(r"LOSSES:\[([^\]]+)\]", log)
    if not match:
        return []
    try:
        return [float(x.strip()) for x in match.group(1).split(",")]
    except Exception:
        return []


def _parse_val_accs_from_log(log: str) -> list[float]:
    """Extract VAL_ACCS:[...] from stdout."""
    match = re.search(r"VAL_ACCS:\[([^\]]+)\]", log)
    if not match:
        return []
    try:
        return [float(x.strip()) for x in match.group(1).split(",")]
    except Exception:
        return []


def _score_color(score: float) -> str:
    if score >= 0.7:
        return "#22c55e"
    if score >= 0.4:
        return "#eab308"
    return "#ef4444"


def _score_html(score: float) -> str:
    pct = int(score * 100)
    color = _score_color(score)
    return f"""
<div style="text-align:center; margin: 8px 0;">
    <div style="font-size: 2.4em; font-weight: 700; color: {color};">{score:.2f}</div>
    <div style="position:relative; height:24px; background:#1e293b; border-radius:6px; overflow:hidden; margin:8px auto; max-width:280px;">
        <div style="height:100%; width:{pct}%; background:linear-gradient(90deg, {color}88, {color}); border-radius:6px; transition:width 0.6s ease;"></div>
    </div>
    <div style="color:#94a3b8; font-size:0.85em;">{pct}% complete</div>
</div>"""


def _diff_html(original: str, fixed: str) -> str:
    """Generate an HTML diff view between original and fixed code."""
    orig_lines = original.strip().splitlines()
    fixed_lines = fixed.strip().splitlines()
    diff = difflib.unified_diff(orig_lines, fixed_lines, lineterm="", n=3)
    lines = []
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            lines.append(f'<div style="color:#60a5fa;margin:8px 0 2px 0;font-size:0.85em;">{line}</div>')
        elif line.startswith("+"):
            lines.append(f'<div style="color:#22c55e;background:#22c55e0d;padding:1px 6px;font-family:monospace;font-size:0.85em;">+ {line[1:]}</div>')
        elif line.startswith("-"):
            lines.append(f'<div style="color:#ef4444;background:#ef44440d;padding:1px 6px;font-family:monospace;font-size:0.85em;">- {line[1:]}</div>')
        else:
            lines.append(f'<div style="color:#94a3b8;padding:1px 6px;font-family:monospace;font-size:0.85em;">  {line}</div>')
    if not lines:
        return '<div style="color:#94a3b8;text-align:center;padding:20px;">No changes detected</div>'
    return '<div style="background:#0f172a;border-radius:8px;padding:12px;overflow-x:auto;overflow-y:auto;">' + "\n".join(lines) + "</div>"


def _format_step_history(history: list[dict]) -> str:
    """Format step history into a Markdown log for display."""
    if not history:
        return "*No steps yet. Reset an episode and start debugging.*"
    
    lines = ["### 📜 Step History\n"]
    for entry in history:
        turn = entry.get("turn", "?")
        action = entry.get("action_type", "unknown")
        timestamp = entry.get("timestamp", "")
        
        # Action badge color
        if action == "submit_fix":
            badge = "🎯"
            color = "#22c55e"
        elif action in ["execute_snippet", "run_training_probe"]:
            badge = "▶️"
            color = "#3b82f6"
        elif action in ["inspect_tensor", "get_variable_state"]:
            badge = "🔍"
            color = "#f59e0b"
        elif action == "inspect_diff":
            badge = "🔀"
            color = "#8b5cf6"
        else:
            badge = "📝"
            color = "#94a3b8"
        
        lines.append(f"**Turn {turn}** {badge} `{action}`")
        
        # Show key details based on action type
        if action == "execute_snippet":
            code = entry.get("code", "")
            stdout = entry.get("stdout", "")
            exit_code = entry.get("exit_code", 0)
            lines.append(f"- Exit: `{exit_code}`")
            if stdout.strip():
                lines.append(f"```\n{stdout}\n```")
        elif action == "inspect_tensor":
            target = entry.get("target_expression", "?")
            shape = entry.get("shape")
            dtype = entry.get("dtype")
            lines.append(f"- Target: `{target}`")
            if shape:
                lines.append(f"- Shape: `{shape}`, dtype: `{dtype}`")
            if entry.get("error"):
                lines.append(f"- ⚠️ Error: {entry['error']}")
        elif action == "submit_fix":
            reward = entry.get("reward", 0.0)
            color = "#22c55e" if reward >= 0.7 else "#ef4444"
            lines.append(f"- **Reward: {reward:.2f}**")
        elif action == "run_training_probe":
            losses = entry.get("losses", [])
            if losses:
                lines.append(f"- Losses: `{losses}`")
        elif action == "get_variable_state":
            results = entry.get("results", {})
            for expr, res in results.items():
                lines.append(f"- `{expr}`: `{res.get('repr', '?')}`")
        elif action == "inspect_diff":
            lines_changed = entry.get("lines_changed", 0)
            lines.append(f"- Lines changed: {lines_changed}")
        
        lines.append("")  # Blank line between entries
    
    return "\n".join(lines)


def _step_timeline_html(trajectory: list[dict], current_step: int, max_steps: int = 3) -> str:
    """Render step timeline as HTML."""
    items = []
    for i in range(1, max_steps + 1):
        entry = next((t for t in trajectory if t.get("step") == i), None)
        if entry:
            r = entry["reward"]
            color = _score_color(r)
            items.append(
                f'<span class="step-indicator step-done" style="border-color:{color}44;background:{color}11;color:{color};">'
                f'Step {i} → {r:.2f}</span>'
            )
        elif i == current_step + 1:
            items.append(f'<span class="step-indicator step-active">Step {i} ▶</span>')
        else:
            items.append(f'<span class="step-indicator step-pending">Step {i}</span>')

    return '<div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;padding:8px 0;">' + "".join(items) + "</div>"


def _loss_plot(losses: list[float], title: str = "Loss Curve"):
    """Generate a matplotlib figure for loss curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if losses:
        valid_losses = [(i, l) for i, l in enumerate(losses) if not (math.isnan(l) or math.isinf(l))]
        nan_steps = [i for i, l in enumerate(losses) if math.isnan(l) or math.isinf(l)]

        if valid_losses:
            steps, vals = zip(*valid_losses)
            ax.plot(steps, vals, color="#60a5fa", linewidth=2, marker="o", markersize=3, zorder=3)
            ax.fill_between(steps, vals, alpha=0.15, color="#60a5fa")

        if nan_steps:
            ax.scatter(nan_steps, [max(v for _, v in valid_losses) if valid_losses else 1.0] * len(nan_steps),
                      color="#ef4444", marker="x", s=60, zorder=4, label="NaN/Inf")
            ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")

    ax.set_xlabel("Step", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Loss", color="#94a3b8", fontsize=9)
    ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.15, color="#475569")

    fig.tight_layout()
    return fig


def _val_acc_plot(accs: list[float]):
    """Generate a matplotlib figure for validation accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if accs:
        epochs = list(range(1, len(accs) + 1))
        ax.plot(epochs, accs, color="#a78bfa", linewidth=2, marker="o", markersize=3)
        ax.fill_between(epochs, accs, alpha=0.15, color="#a78bfa")
        ax.axhline(y=0.7, color="#22c55e", linestyle="--", alpha=0.5, label="Target (0.70)")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")

    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Accuracy", color="#94a3b8", fontsize=9)
    ax.set_title("Validation Accuracy", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, alpha=0.15, color="#475569")

    fig.tight_layout()
    return fig


# ── Episode state ──────────────────────────────────────────────────────────

class EpisodeState:
    """Track episode state across Gradio interactions."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.task_id = ""
        self.buggy_code = ""
        self.task_description = ""
        self.episode_id = ""  # Track episode_id for session persistence
        self.step = 0
        self.turn = 0
        self.best_reward = 0.0
        self.last_reward = 0.0
        self.error_log = ""
        self.trajectory: list[dict] = []
        self.tool_history: list[dict] = []
        self.done = False
        self.last_fixed_code = ""


_state = EpisodeState()


# ── Tool handlers ──────────────────────────────────────────────────────────

def _format_tool_result(data: dict, tool_type: str) -> str:
    """Format tool observation as readable HTML."""
    obs = data.get("observation", data)
    turn = obs.get("turn", 0)
    error = obs.get("error")
    
    if error:
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">{tool_type}</span><br>
<span style="color:#ef4444;">❌ Error: {error}</span>
</div>"""
    
    if tool_type == "execute_snippet":
        stdout = obs.get("stdout", "") or "(empty)"
        stderr = obs.get("stderr", "")
        exit_code = obs.get("exit_code", 0)
        timed_out = obs.get("timed_out", False)
        status = "⏱️ Timed out" if timed_out else ("✅" if exit_code == 0 else f"❌ Exit {exit_code}")
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">▶️ Execute Snippet</span> {status}<br>
<pre style="background:#0f172a;padding:8px;border-radius:4px;overflow-x:auto;overflow-y:auto;">{stdout}</pre>
{f'<pre style="color:#ef4444;font-size:0.85em;">{stderr}</pre>' if stderr else ''}
</div>"""
    
    elif tool_type == "inspect_tensor":
        shape = obs.get("shape")
        dtype = obs.get("dtype")
        requires_grad = obs.get("requires_grad")
        grad_is_none = obs.get("grad_is_none")
        min_val = obs.get("min_val")
        max_val = obs.get("max_val")
        mean_val = obs.get("mean_val")
        is_nan = obs.get("is_nan")
        is_inf = obs.get("is_inf")
        
        parts = []
        if shape is not None: parts.append(f"<b>Shape:</b> {shape}")
        if dtype: parts.append(f"<b>Dtype:</b> {dtype}")
        if requires_grad is not None: parts.append(f"<b>requires_grad:</b> {requires_grad}")
        if grad_is_none is not None: parts.append(f"<b>grad is None:</b> {grad_is_none}")
        if min_val is not None: parts.append(f"<b>Min:</b> {min_val:.4f}")
        if max_val is not None: parts.append(f"<b>Max:</b> {max_val:.4f}")
        if mean_val is not None: parts.append(f"<b>Mean:</b> {mean_val:.4f}")
        if is_nan: parts.append("<span style='color:#ef4444;'>⚠️ Contains NaN</span>")
        if is_inf: parts.append("<span style='color:#ef4444;'>⚠️ Contains Inf</span>")
        
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">🔬 Inspect Tensor</span><br>
{' | '.join(parts) if parts else 'No tensor info available'}
</div>"""
    
    elif tool_type == "run_training_probe":
        losses = obs.get("losses", [])
        grad_norms = obs.get("grad_norms", {})
        final_loss = obs.get("final_loss")
        loss_is_nan = obs.get("loss_is_nan", False)
        loss_is_inf = obs.get("loss_is_inf", False)
        timed_out = obs.get("timed_out", False)
        
        loss_str = ", ".join(f"{l:.4f}" for l in losses[:10]) if losses else "N/A"
        grad_str = ", ".join(f"{k}: {v:.4f}" for k, v in list(grad_norms.items())[:5]) if grad_norms else "N/A"
        
        warnings = []
        if loss_is_nan: warnings.append("⚠️ NaN loss detected")
        if loss_is_inf: warnings.append("⚠️ Inf loss detected")
        if timed_out: warnings.append("⏱️ Timed out")
        
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">📈 Training Probe</span><br>
<b>Losses:</b> [{loss_str}]<br>
<b>Final Loss:</b> {f'{final_loss:.4f}' if final_loss is not None else 'N/A'}<br>
<b>Grad Norms:</b> {grad_str}<br>
{' '.join(f'<span style="color:#ef4444;">{w}</span>' for w in warnings)}
</div>"""
    
    elif tool_type == "get_variable_state":
        results = obs.get("results", {})
        items = []
        for expr, res in list(results.items())[:10]:
            if res.get("error"):
                items.append(f"<b>{expr}:</b> <span style='color:#ef4444;'>Error: {res['error']}</span>")
            else:
                val = res.get("repr", str(res.get("value", "?")))
                typ = res.get("type", "?")
                shape = res.get("shape")
                shape_str = f" shape={shape}" if shape else ""
                items.append(f"<b>{expr}:</b> {val} <i>({typ}{shape_str})</i>")
        
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">🔎 Variable State</span><br>
{'<br>'.join(items) if items else 'No results'}
</div>"""
    
    elif tool_type == "inspect_diff":
        diff = obs.get("diff", "")
        lines_changed = obs.get("lines_changed", 0)
        additions = obs.get("additions", 0)
        deletions = obs.get("deletions", 0)
        
        # Format diff with colors
        diff_lines = []
        for line in diff.split("\n")[:50]:
            if line.startswith("+") and not line.startswith("+++"):
                diff_lines.append(f'<span style="color:#22c55e;">{line}</span>')
            elif line.startswith("-") and not line.startswith("---"):
                diff_lines.append(f'<span style="color:#ef4444;">{line}</span>')
            elif line.startswith("@@"):
                diff_lines.append(f'<span style="color:#60a5fa;">{line}</span>')
            else:
                diff_lines.append(line)
        
        return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">📝 Inspect Diff</span><br>
<b>Changes:</b> {lines_changed} lines (+{additions}/-{deletions})<br>
<pre style="background:#0f172a;padding:8px;border-radius:4px;overflow-x:auto;overflow-y:auto;font-size:0.8em;">{'<br>'.join(diff_lines)}</pre>
</div>"""
    
    return f"""
<div class="tool-result">
<strong>Turn {turn}</strong> — <span class="tool-badge">{tool_type}</span><br>
<pre>{json.dumps(obs, indent=2, default=str)}</pre>
</div>"""


def do_tool_call(base_url: str, tool_type: str, code: str, setup_code: str, 
                 target_expr: str, expressions_str: str, proposed_code: str, steps: int):
    """Execute a tool call and return formatted results."""
    if _state.done:
        return "⚠️ Episode is done. Reset to start a new one.", "", _format_step_history([])
    
    # Require a valid episode to be started first
    if not _state.episode_id:
        return "⚠️ No episode started. Click Reset to start an episode first.", "", _format_step_history([])
    
    # Build action payload based on tool type - ALWAYS include episode_id
    action = {
        "action_type": tool_type,
        "episode_id": _state.episode_id,  # Critical for session tracking
    }
    
    if tool_type == "execute_snippet":
        action["code"] = code
    elif tool_type == "inspect_tensor":
        action["setup_code"] = setup_code
        action["target_expression"] = target_expr
    elif tool_type == "run_training_probe":
        action["code"] = code
        action["steps"] = int(steps)
    elif tool_type == "get_variable_state":
        action["setup_code"] = setup_code
        # Parse expressions (one per line or comma-separated)
        exprs = [e.strip() for e in expressions_str.replace(",", "\n").split("\n") if e.strip()]
        action["expressions"] = exprs[:10]
    elif tool_type == "inspect_diff":
        action["proposed_code"] = proposed_code
    
    payload = {"action": action}
    data = _api(base_url, "POST", "/step", payload)
    
    if "error" in data and not data.get("observation"):
        return f"❌ API Error: {data['error']}", "", _format_step_history([])
    
    obs = data.get("observation", data)
    _state.turn = obs.get("turn", _state.turn + 1)
    
    # Build history entry with all relevant fields
    history_entry = {
        "turn": _state.turn,
        "action_type": tool_type,
        "timestamp": "",  # Could add timestamp here
    }
    
    # Copy relevant fields from observation
    if tool_type == "execute_snippet":
        history_entry["code"] = code
        history_entry["stdout"] = obs.get("stdout", "")
        history_entry["stderr"] = obs.get("stderr", "")
        history_entry["exit_code"] = obs.get("exit_code", 0)
    elif tool_type == "inspect_tensor":
        history_entry["target_expression"] = target_expr
        history_entry["shape"] = obs.get("shape")
        history_entry["dtype"] = obs.get("dtype")
        history_entry["error"] = obs.get("error")
    elif tool_type == "run_training_probe":
        history_entry["losses"] = obs.get("losses", [])
        history_entry["final_loss"] = obs.get("final_loss")
    elif tool_type == "get_variable_state":
        history_entry["results"] = obs.get("results", {})
    elif tool_type == "inspect_diff":
        history_entry["lines_changed"] = obs.get("lines_changed", 0)
        history_entry["additions"] = obs.get("additions", 0)
        history_entry["deletions"] = obs.get("deletions", 0)
    
    _state.tool_history.append(history_entry)
    
    result_html = _format_tool_result(data, tool_type)
    status = f"✅ Turn {_state.turn}/10 — {TOOL_INFO.get(tool_type, {}).get('name', tool_type)} complete"
    
    # Format full history
    formatted_history = _format_step_history(_state.tool_history)
    
    return status, result_html, formatted_history


# ── Action handlers ────────────────────────────────────────────────────────

def do_reset(base_url: str, task_id: str):
    """Reset the environment for a given task."""
    _state.reset()
    data = _api(base_url, "POST", "/reset", {"task_id": task_id})
    if "error" in data:
        return (
            "",                                  # code editor
            "",                                  # task desc
            _score_html(0.0),                    # score
            None,                                # loss plot
            None,                                # acc plot
            "",                                  # diff
            "",                                  # error log
            "",                                  # tool output
            _format_step_history([]),            # step history log
        )

    obs = data.get("observation", data)
    _state.task_id = obs.get("task_id", task_id)
    _state.buggy_code = obs.get("buggy_code", "")
    _state.task_description = obs.get("task_description", "")
    _state.episode_id = obs.get("episode_id", "")  # Store episode_id for session tracking
    _state.step = 0
    _state.turn = 0
    _state.tool_history = []
    _state.done = False

    info = TASK_INFO.get(task_id, {})
    task_md = f"""### {info.get('name', task_id)}  {info.get('difficulty', '')}

{_state.task_description.strip()}

**💡 Hint:** {info.get('hints', 'No hints available.')}
"""

    return (
        _state.buggy_code.strip(),
        task_md,
        _score_html(0.0),
        None,  # loss plot (hidden)
        None,
        '<div style="color:#94a3b8;text-align:center;padding:20px;">Submit a fix to see the diff</div>',
        "",
        "",  # tool_output
        _format_step_history([]),  # step history log (cleared)
    )


def do_step(base_url: str, fixed_code: str):
    """Submit a fix attempt."""
    if _state.done:
        return (
            _score_html(_state.best_reward),
            None, None, "",  "",
            _format_step_history(_state.tool_history),
        )

    # Require a valid episode to be started first
    if not _state.episode_id:
        return (
            _score_html(0.0),
            None, None, "", "",
            _format_step_history([]),
        )

    if not fixed_code or not fixed_code.strip():
        return (
            _score_html(_state.last_reward),
            None, None, "", "",
            _format_step_history(_state.tool_history),
        )

    _state.step += 1
    _state.last_fixed_code = fixed_code

    # Use the new action format with action_type and episode_id
    payload = {"action": {
        "action_type": "submit_fix",
        "fixed_code": fixed_code, 
        "attempt_number": _state.step,
        "episode_id": _state.episode_id,  # Include for session tracking
    }}
    data = _api(base_url, "POST", "/step", payload)

    if "error" in data:
        _state.step -= 1
        return (
            _score_html(_state.last_reward),
            None, None, "", "",
            _format_step_history(_state.tool_history),
        )

    reward = float(data.get("reward", 0.0) or 0.0)
    _state.last_reward = reward
    _state.best_reward = max(_state.best_reward, reward)
    _state.done = data.get("done", False)

    obs = data.get("observation", {})
    # Update turn from server response (authoritative)
    _state.turn = obs.get("turn", _state.turn + 1)
    _state.error_log = obs.get("error_log", "")
    metrics = obs.get("metrics", {})

    _state.trajectory.append({
        "step": _state.step,
        "reward": reward,
        "best_reward": _state.best_reward,
        "metrics": metrics,
        "done": _state.done,
    })
    
    # Add submit_fix to tool history using server's turn value
    _state.tool_history.append({
        "turn": _state.turn,
        "action_type": "submit_fix",
        "reward": reward,
    })

    # Parse outputs for visualization (kept for hidden plots)
    losses = _parse_losses_from_log(_state.error_log)
    val_accs = _parse_val_accs_from_log(_state.error_log)

    loss_fig = _loss_plot(losses, f"Loss Curve — Step {_state.step}")
    acc_fig = _val_acc_plot(val_accs) if val_accs else None

    diff = _diff_html(_state.buggy_code, fixed_code)

    if reward >= 0.95:
        emoji = "🎯"
    elif reward >= 0.7:
        emoji = "✅"
    elif len(_state.trajectory) > 1 and reward > _state.trajectory[-2]["reward"]:
        emoji = "📈"
    else:
        emoji = "⚠️"
    done_msg = " — Episode complete!" if _state.done else ""
    error_display = _state.error_log if _state.error_log else "No errors — code ran successfully."

    return (_score_html(_state.best_reward), loss_fig, acc_fig, diff, error_display,
            _format_step_history(_state.tool_history))


def do_run_baseline(base_url: str, task_id: str):
    """Run the baseline agent on a single task."""
    # First reset
    reset_result = do_reset(base_url, task_id)
    yield reset_result + ("🤖 Resetting environment...",)

    # Call baseline endpoint
    data = _api(base_url, "GET", "/baseline")
    if "error" in data:
        yield reset_result + (f"❌ Baseline error: {data['error']}",)
        return

    scores = data.get("baseline_scores", {})
    avg = data.get("average", 0.0)

    results_md = "### 🤖 Baseline Agent Results\n\n"
    results_md += "| Task | Score |\n|---|---|\n"
    for tid in ["task1", "task2", "task3", "task4", "task5", "task6"]:
        s = scores.get(tid, 0.0)
        emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
        results_md += f"| {tid} | {emoji} {s:.4f} |\n"
    results_md += f"\n**Average: {avg:.4f}**"

    yield reset_result + (results_md,)


def load_current_state(base_url: str):
    """Fetch and format current environment state for UI display."""
    # Use the session-based state endpoint if we have an episode_id
    if _state.episode_id:
        data = _api(base_url, "GET", f"/session/state?episode_id={_state.episode_id}")
    else:
        # Fall back to OpenEnv's default state endpoint (will show default state)
        data = _api(base_url, "GET", "/state")
        
    if "error" in data:
        summary = "⚠️ Could not fetch current state. Start or reset an episode first, then try again."
        return summary, {"error": data["error"]}

    state_obj = data.get("state", data)
    if not isinstance(state_obj, dict):
        # If data itself is the state object (our custom endpoint returns flat dict)
        state_obj = data

    done = bool(state_obj.get("done", state_obj.get("submitted", False)))
    step = state_obj.get("step", state_obj.get("turn", 0))
    task_id = state_obj.get("task_id", "-")
    reward = state_obj.get("best_reward", state_obj.get("last_reward", state_obj.get("reward", 0.0)))
    turns_remaining = state_obj.get("turns_remaining", 10 - step)
    summary = (
        f"**Task:** {task_id}  |  **Turn:** {step}/10  |  "
        f"**Done:** {'yes' if done else 'no'}  |  **Best Reward:** {reward:.2f}  |  "
        f"**Remaining:** {turns_remaining}"
    )
    return summary, state_obj


# ── Build the UI ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f172a",
        body_background_fill_dark="#0f172a",
        block_background_fill="#1e293b",
        block_background_fill_dark="#1e293b",
        block_border_color="#334155",
        block_border_color_dark="#334155",
        block_label_text_color="#e2e8f0",
        block_label_text_color_dark="#e2e8f0",
        block_title_text_color="#e2e8f0",
        block_title_text_color_dark="#e2e8f0",
        input_background_fill="#0f172a",
        input_background_fill_dark="#0f172a",
        button_primary_background_fill="#3b82f6",
        button_primary_background_fill_dark="#3b82f6",
        button_primary_text_color="#ffffff",
    )

    with gr.Blocks(title="WhipStudio — ML Debug Arena") as app:

        # ── Header ──
        gr.HTML("""
        <div class="arena-header">
            <h1>🔧 WhipStudio — ML Debug Arena</h1>
            <p>An RL environment where agents debug broken PyTorch training scripts</p>
        </div>
        """)

        base_url = gr.Textbox(label="🌐 API Base URL", value=DEFAULT_BASE_URL, scale=1)
        
        # Step history state
        step_history_state = gr.State(value=[])

        with gr.Row(equal_height=False):

            # ── Left column: Task selector ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📋 Task Selector")
                baseline_model = gr.Dropdown(
                    choices=[
                        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        "Qwen/Qwen2.5-Coder-3B-Instruct",
                        "Qwen/Qwen2.5-Coder-7B-Instruct",
                        "Qwen/Qwen2.5-Coder-14B-Instruct",
                        "Qwen/Qwen2.5-Coder-32B-Instruct",
                        "mistralai/Mistral-7B-Instruct-v0.3",
                    ],
                    value="Qwen/Qwen2.5-Coder-32B-Instruct",
                    label="Auto-Agent Model",
                    info="Choose which LLM to run for baseline auto-agent",
                )
                task_id = gr.Radio(
                    choices=["task1", "task2", "task3", "task4", "task5", "task6"],
                    value="task1",
                    label="Select Task",
                    info="Choose a debugging challenge",
                )

                task_desc = gr.Markdown(
                    value="""### Broken Training Loop  🟢 Easy

Fix optimizer order + learning rate bugs in a linear classifier.

**💡 Hint:** Look at optimizer.step() / loss.backward() order and the learning rate."""
                )

                with gr.Row():
                    btn_reset = gr.Button("🔄 Reset", variant="primary", size="sm")
                    btn_baseline = gr.Button("🤖 Auto-Agent", variant="secondary", size="sm")

            # ── Center column: Code editor + Tools ──
            with gr.Column(scale=2, min_width=500):
                with gr.Tabs():
                    # ── Tab 1: Code Editor ──
                    with gr.Tab("💻 Code Editor"):
                        gr.Markdown("""
⚠️ **Submit Fix ends the episode!** Use the **Debug Tools tab** first to run tools iteratively.
""")
                        code_editor = gr.Code(
                            label="Your Fix (edit the code below)",
                            language="python",
                            lines=20,
                        )

                        with gr.Row():
                            btn_submit = gr.Button("🚀 Submit Fix (Final Submission)", variant="primary", size="lg")

                        error_log = gr.Textbox(
                            label="📋 Execution Output / Error Log",
                            lines=6,
                            interactive=False,
                        )
                    
                    # ── Tab 2: Debug Tools (Prominent) ──
                    with gr.Tab("🛠️ Debug Tools"):
                        gr.Markdown("""
### 🔍 Use debugging tools BEFORE submitting your fix

Each tool call uses one turn (max 10 per episode). Tools help you:
- **Verify hypotheses** about bugs
- **Test partial fixes** before full submission
- **Inspect tensors** for shape/gradient issues
- **Preview changes** with diff view
""")
                        tool_type = gr.Radio(
                            choices=[
                                ("▶️ Execute Snippet", "execute_snippet"),
                                ("🔬 Inspect Tensor", "inspect_tensor"),
                                ("📈 Training Probe", "run_training_probe"),
                                ("🔎 Variable State", "get_variable_state"),
                                ("📝 Inspect Diff", "inspect_diff"),
                            ],
                            value="execute_snippet",
                            label="Select Tool",
                            info="Choose a debugging tool",
                        )
                        
                        # Tool-specific inputs (shown/hidden based on selection)
                        with gr.Group():
                            tool_code = gr.Code(
                                label="Python Code (runs in sandbox)",
                                language="python",
                                lines=10,
                                visible=True,
                                value="# Enter Python code to run...\nimport torch\nprint(torch.__version__)"
                            )
                            tool_setup_code = gr.Code(
                                label="Setup Code (imports, model definition, etc.)",
                                language="python",
                                lines=8,
                                visible=False,
                                value="import torch\nimport torch.nn as nn\n\nmodel = nn.Linear(10, 2)\nx = torch.randn(5, 10)\ny = model(x)"
                            )
                            tool_target_expr = gr.Textbox(
                                label="Target Expression",
                                placeholder="e.g., model.weight.grad, loss, backbone[0].weight",
                                visible=False,
                            )
                            tool_expressions = gr.Textbox(
                                label="Expressions (one per line)",
                                placeholder="model.training\noptimizer.param_groups[0]['lr']\nloss.item()",
                                lines=4,
                                visible=False,
                            )
                            tool_proposed_code = gr.Code(
                                label="Proposed Code (to diff against original)",
                                language="python",
                                lines=10,
                                visible=False,
                            )
                            tool_steps = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Training Steps",
                                visible=False,
                            )
                        
                        btn_run_tool = gr.Button("▶️ Run Tool", variant="primary", size="lg")
                        tool_status = gr.Textbox(label="Status", interactive=False, lines=1)
                        tool_output = gr.HTML(
                            label="Tool Output",
                            value='<div style="background:#1e293b;border-radius:8px;padding:16px;color:#94a3b8;text-align:center;">Select a tool and click Run to see output</div>'
                        )
                        
                        # Show/hide inputs based on tool selection
                        def update_tool_visibility(tool):
                            return {
                                tool_code: gr.update(visible=tool in ["execute_snippet", "run_training_probe"]),
                                tool_setup_code: gr.update(visible=tool in ["inspect_tensor", "get_variable_state"]),
                                tool_target_expr: gr.update(visible=tool == "inspect_tensor"),
                                tool_expressions: gr.update(visible=tool == "get_variable_state"),
                                tool_proposed_code: gr.update(visible=tool == "inspect_diff"),
                                tool_steps: gr.update(visible=tool == "run_training_probe"),
                            }
                        
                        tool_type.change(
                            fn=update_tool_visibility,
                            inputs=[tool_type],
                            outputs=[tool_code, tool_setup_code, tool_target_expr, tool_expressions, tool_proposed_code, tool_steps],
                        )

            # ── Right column: Results ──
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📊 Results")
                score_display = gr.HTML(value=_score_html(0.0))

                with gr.Tabs():
                    with gr.Tab("🔀 Code Diff"):
                        diff_view = gr.HTML(
                            value='<div style="color:#94a3b8;text-align:center;padding:20px;">Submit a fix to see the diff</div>'
                        )
                    with gr.Tab("🧭 State"):
                        state_summary = gr.Markdown(
                            value="Press Reset to start an episode."
                        )
                        btn_refresh_state = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                        state_json = gr.JSON(label="/state", value={})
                    with gr.Tab("📜 Step History"):
                        step_history_log = gr.Markdown(
                            value="*No steps yet. Reset an episode and start debugging.*",
                            elem_id="step-history-log"
                        )
                        btn_clear_history = gr.Button("🗑️ Clear History", variant="secondary", size="sm")
                
                # Hidden plots for val accuracy and loss (still needed for binding)
                acc_plot = gr.Plot(label="Val Acc", visible=False)
                loss_plot = gr.Plot(label="Loss Curve", visible=False)

                baseline_output = gr.Markdown(label="Baseline Results", visible=False)


        # ── Bottom: Raw API tab (for developers) ──
        with gr.Accordion("🔧 Developer Tools (Raw API)", open=False):
            with gr.Row():
                with gr.Column():
                    dev_method = gr.Radio(["GET", "POST"], value="GET", label="Method")
                    dev_path = gr.Textbox(label="Path", value="/health")
                    dev_payload = gr.Code(label="Payload (JSON)", language="json", value="{}")
                    btn_dev = gr.Button("Send Request", variant="secondary")
                with gr.Column():
                    dev_status = gr.Textbox(label="Status")
                    dev_response = gr.Code(label="Response", language="json")

            def dev_call(base, method, path, payload_text):
                base = (base or DEFAULT_BASE_URL).strip().rstrip("/")
                url = f"{base}{path}"
                try:
                    payload = json.loads(payload_text) if payload_text.strip() else {}
                except json.JSONDecodeError as e:
                    return f"JSON Error", str(e)
                try:
                    with httpx.Client(timeout=120.0) as client:
                        if method == "GET":
                            resp = client.get(url)
                        else:
                            resp = client.post(url, json=payload)
                    ct = resp.headers.get("content-type", "")
                    body = json.dumps(resp.json(), indent=2) if "json" in ct else resp.text
                    return f"{resp.status_code} {resp.reason_phrase}", body
                except Exception as exc:
                    return "Error", f"{exc.__class__.__name__}: {exc}"

            btn_dev.click(
                fn=dev_call,
                inputs=[base_url, dev_method, dev_path, dev_payload],
                outputs=[dev_status, dev_response],
            )


        # ── Event bindings ──

        # Task selector updates description
        def update_task_desc(tid):
            info = TASK_INFO.get(tid, {})
            return f"""### {info.get('name', tid)}  {info.get('difficulty', '')}

{info.get('description', '')}

**💡 Hint:** {info.get('hints', 'No hints available.')}"""

        task_id.change(fn=update_task_desc, inputs=[task_id], outputs=[task_desc])

        # Reset
        btn_reset.click(
            fn=do_reset,
            inputs=[base_url, task_id],
            outputs=[code_editor, task_desc, score_display, loss_plot, acc_plot, diff_view, error_log, tool_output, step_history_log],
        ).then(
            fn=load_current_state,
            inputs=[base_url],
            outputs=[state_summary, state_json],
        )
        
        # Tool execution
        btn_run_tool.click(
            fn=do_tool_call,
            inputs=[base_url, tool_type, tool_code, tool_setup_code, tool_target_expr, tool_expressions, tool_proposed_code, tool_steps],
            outputs=[tool_status, tool_output, step_history_log],
        )

        # Submit fix
        btn_submit.click(
            fn=do_step,
            inputs=[base_url, code_editor],
            outputs=[score_display, loss_plot, acc_plot, diff_view, error_log, step_history_log],
        ).then(
            fn=load_current_state,
            inputs=[base_url],
            outputs=[state_summary, state_json],
        )

        btn_refresh_state.click(
            fn=load_current_state,
            inputs=[base_url],
            outputs=[state_summary, state_json],
        )
        
        # Clear history button
        def clear_history():
            _state.tool_history = []
            return _format_step_history([])
        
        btn_clear_history.click(
            fn=clear_history,
            inputs=[],
            outputs=[step_history_log],
        )

        # Baseline (auto-agent) — live streaming per-task
        TASK_NAMES = {
            "task1": "🟢 Broken Training Loop",
            "task2": "🟡 Silent NaN Loss",
            "task3": "🔴 OOM + Data Leakage",
            "task4": "🟡 Wrong Loss Function",
            "task5": "🟡 Frozen Backbone",
            "task6": "🔴 Input-Output Mismatch",
        }

        def run_baseline_live(base_url_val, model_id_val):
            """Generator that yields live progress as each task completes."""
            base = (base_url_val or DEFAULT_BASE_URL).strip().rstrip("/")
            model_id = (model_id_val or "Qwen/Qwen2.5-Coder-32B-Instruct").strip()
            results = {}
            lines_header = [
                "### 🤖 Baseline Agent — Live Progress\n",
                f"**Model:** `{model_id}`\n",
            ]

            # Phase 1: Show "starting" state
            yield "\n".join(lines_header + ["⏳ Starting baseline agent..."])

            for tid in ["task1", "task2", "task3", "task4", "task5", "task6"]:
                tname = TASK_NAMES.get(tid, tid)

                # Show "running this task" update
                progress_lines = list(lines_header)
                # Show completed tasks
                for done_tid, info in results.items():
                    s = info["score"]
                    emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
                    progress_lines.append(f"- {emoji} **{TASK_NAMES.get(done_tid, done_tid)}**: {s:.4f}")
                    if info.get("error"):
                        progress_lines.append(f"  - ⚠️ `{info['error']}`")
                # Show currently running task
                progress_lines.append(f"\n🤖 **Running {tname}** — agent is analyzing the code and generating a fix...")
                progress_lines.append(f"\n*This may take 30-60 seconds per task (LLM inference + sandbox execution × 3 attempts)*")
                yield "\n".join(progress_lines)

                # Actually call the per-task endpoint
                try:
                    with httpx.Client(timeout=180.0) as client:
                        resp = client.get(f"{base}/baseline/task/{tid}", params={"model_id": model_id})
                        resp.raise_for_status()
                        data = resp.json()
                except Exception as exc:
                    data = {"score": 0.0, "error": f"{exc.__class__.__name__}: {exc}"}

                results[tid] = {
                    "score": float(data.get("score", 0.0)),
                    "error": data.get("error", ""),
                    "fixed_code": data.get("fixed_code", ""),
                    "output": data.get("output", ""),
                }

            # Final summary
            final_lines = ["### 🤖 Baseline Agent Results\n", "| Task | Score |", "|---|---|"]
            total = 0.0
            has_errors = False
            for tid in ["task1", "task2", "task3", "task4", "task5", "task6"]:
                info = results.get(tid, {"score": 0.0})
                s = info["score"]
                total += s
                emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
                final_lines.append(f"| {TASK_NAMES.get(tid, tid)} | {emoji} **{s:.4f}** |")
                if info.get("error"):
                    has_errors = True
                    final_lines.append(f"\n> ⚠️ `{info['error']}`\n")

            avg = total / 6
            final_lines.append(f"\n**Average: {avg:.4f}**")
            if avg >= 0.7:
                final_lines.append("\n🎉 **Agent performed well!** The environment is solvable.")
            elif avg >= 0.3:
                final_lines.append("\n📈 **Agent showed partial progress.** Reward shaping is working.")
            elif not has_errors:
                final_lines.append("\n⚠️ **Agent scored low.** Tasks may be too challenging for zero-shot inference.")

            if has_errors:
                final_lines.append("\n---\n> [!WARNING]\n> Some tasks failed. Check if `HF_TOKEN` is valid and the model is accessible.")

            final_lines.append("\n---\n### 🔍 Auto-Agent Generated Code & Execution Logs")
            for tid in ["task1", "task2", "task3", "task4", "task5", "task6"]:
                info = results.get(tid, {})
                fixed_code = str(info.get("fixed_code", ""))
                output = str(info.get("output", ""))
                if fixed_code.strip() or output.strip():
                    final_lines.append(f"\n#### {TASK_NAMES.get(tid, tid)}")
                    if fixed_code.strip():
                        final_lines.append("<details><summary><b>Show Generated Code</b></summary>\n\n```python\n" + fixed_code + "\n```\n</details>")
                    if output.strip():
                        final_lines.append("<details><summary><b>Show Execution Output</b></summary>\n\n```text\n" + output + "\n```\n</details>")

            yield "\n".join(final_lines)

        btn_baseline.click(
            fn=lambda: gr.update(visible=True),
            outputs=[baseline_output],
        ).then(
            fn=run_baseline_live,
            inputs=[base_url, baseline_model],
            outputs=[baseline_output],
        )

        # Footer
        gr.HTML("""
        <div style="text-align:center; padding:16px 0; color:#64748b; font-size:0.85em; border-top:1px solid #1e293b; margin-top:16px;">
            WhipStudio v1.0 — OpenEnv ML Debug Environment
            · <a href="/web" style="color:#60a5fa;">OpenEnv Web UI →</a>
            · <a href="/docs" style="color:#60a5fa;">API Docs →</a>
        </div>
        """)

    return app


def main(host: str = "0.0.0.0", port: int = 7860):
    app = build_ui()
    app.launch(server_name=host, server_port=port, css=CUSTOM_CSS)


if __name__ == "__main__":
    main()
