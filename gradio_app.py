"""
WhipStudio — ML Debug Arena
A polished Gradio UI for the ML Debugging RL environment.
Provides code editing, loss curve visualization, diff views, and episode history.
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
        "name": "Label Inversion",
        "difficulty": "🟡 Medium",
        "description": "Fix a label transformation bug causing ~50% validation accuracy.",
        "hints": "Look at what labels are being passed to the loss function — are they correct?",
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
"""


# ── API helpers ────────────────────────────────────────────────────────────

def _api(base_url: str, method: str, path: str, payload: dict | None = None) -> dict:
    """Call the WhipStudio API and return parsed JSON."""
    base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    url = f"{base_url}{path}"
    try:
        with httpx.Client(timeout=90.0) as client:
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
    return '<div style="background:#0f172a;border-radius:8px;padding:12px;overflow-x:auto;max-height:500px;overflow-y:auto;">' + "\n".join(lines) + "</div>"


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
        self.step = 0
        self.best_reward = 0.0
        self.last_reward = 0.0
        self.error_log = ""
        self.trajectory: list[dict] = []
        self.done = False
        self.last_fixed_code = ""


_state = EpisodeState()


# ── Action handlers ────────────────────────────────────────────────────────

def do_reset(base_url: str, task_id: str):
    """Reset the environment for a given task."""
    _state.reset()
    data = _api(base_url, "POST", "/reset", {"task_id": task_id})
    if "error" in data:
        return (
            f"❌ Error: {data['error']}",      # status
            "",                                  # code editor
            "",                                  # task desc
            _score_html(0.0),                    # score
            None,                                # loss plot
            None,                                # acc plot
            "",                                  # diff
            _step_timeline_html([], 0),          # timeline
            "",                                  # error log
        )

    obs = data.get("observation", data)
    _state.task_id = obs.get("task_id", task_id)
    _state.buggy_code = obs.get("buggy_code", "")
    _state.task_description = obs.get("task_description", "")
    _state.step = 0
    _state.done = False

    info = TASK_INFO.get(task_id, {})
    task_md = f"""### {info.get('name', task_id)}  {info.get('difficulty', '')}

{_state.task_description.strip()}

**💡 Hint:** {info.get('hints', 'No hints available.')}
"""

    return (
        f"✅ Episode started — {info.get('name', task_id)}",
        _state.buggy_code.strip(),
        task_md,
        _score_html(0.0),
        _loss_plot([], "Loss Curve — Submit a fix to see results"),
        None,
        '<div style="color:#94a3b8;text-align:center;padding:20px;">Submit a fix to see the diff</div>',
        _step_timeline_html([], 0),
        "",
    )


def do_step(base_url: str, fixed_code: str):
    """Submit a fix attempt."""
    if _state.done:
        return (
            "⚠️ Episode is done. Reset to start a new one.",
            _score_html(_state.best_reward),
            None, None, "", _step_timeline_html(_state.trajectory, _state.step), "",
        )

    if not fixed_code or not fixed_code.strip():
        return (
            "⚠️ Please enter code before submitting.",
            _score_html(_state.last_reward),
            None, None, "", _step_timeline_html(_state.trajectory, _state.step), "",
        )

    _state.step += 1
    _state.last_fixed_code = fixed_code

    payload = {"action": {"fixed_code": fixed_code, "attempt_number": _state.step}}
    data = _api(base_url, "POST", "/step", payload)

    if "error" in data:
        _state.step -= 1
        return (
            f"❌ Error: {data['error']}",
            _score_html(_state.last_reward),
            None, None, "", _step_timeline_html(_state.trajectory, _state.step), "",
        )

    reward = float(data.get("reward", 0.0) or 0.0)
    _state.last_reward = reward
    _state.best_reward = max(_state.best_reward, reward)
    _state.done = data.get("done", False)

    obs = data.get("observation", {})
    _state.error_log = obs.get("error_log", "")
    metrics = obs.get("metrics", {})

    _state.trajectory.append({
        "step": _state.step,
        "reward": reward,
        "best_reward": _state.best_reward,
        "metrics": metrics,
        "done": _state.done,
    })

    # Parse outputs for visualization
    losses = _parse_losses_from_log(_state.error_log)
    val_accs = _parse_val_accs_from_log(_state.error_log)

    loss_fig = _loss_plot(losses, f"Loss Curve — Step {_state.step}")
    acc_fig = _val_acc_plot(val_accs) if val_accs else None

    diff = _diff_html(_state.buggy_code, fixed_code)
    timeline = _step_timeline_html(_state.trajectory, _state.step)

    if reward >= 0.95:
        emoji = "🎯"
    elif reward >= 0.7:
        emoji = "✅"
    elif len(_state.trajectory) > 1 and reward > _state.trajectory[-2]["reward"]:
        emoji = "📈"
    else:
        emoji = "⚠️"
    done_msg = " — Episode complete!" if _state.done else ""
    status = f"{emoji} Step {_state.step}/3 — Reward: {reward:.2f} (Best: {_state.best_reward:.2f}){done_msg}"

    error_display = _state.error_log if _state.error_log else "No errors — code ran successfully."

    return (status, _score_html(_state.best_reward), loss_fig, acc_fig, diff, timeline, error_display)


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
    for tid in ["task1", "task2", "task3", "task4", "task5"]:
        s = scores.get(tid, 0.0)
        emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
        results_md += f"| {tid} | {emoji} {s:.4f} |\n"
    results_md += f"\n**Average: {avg:.4f}**"

    yield reset_result + (results_md,)


def load_current_state(base_url: str):
    """Fetch and format current environment state for UI display."""
    data = _api(base_url, "GET", "/state")
    if "error" in data:
        summary = "⚠️ Could not fetch current state. Start or reset an episode first, then try again."
        return summary, {"error": data["error"]}

    state_obj = data.get("state", data)
    if not isinstance(state_obj, dict):
        return "⚠️ State endpoint returned an unexpected response format.", {"raw": data}

    done = bool(state_obj.get("done", False))
    step = state_obj.get("step", 0)
    task_id = state_obj.get("task_id", "-")
    reward = state_obj.get("last_reward", state_obj.get("reward", 0.0))
    summary = (
        f"**Task:** {task_id}  |  **Step:** {step}  |  "
        f"**Done:** {'yes' if done else 'no'}  |  **Last Reward:** {reward}"
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

    with gr.Blocks(title="WhipStudio — ML Debug Arena", css=CUSTOM_CSS) as app:

        # ── Header ──
        gr.HTML("""
        <div class="arena-header">
            <h1>🔧 WhipStudio — ML Debug Arena</h1>
            <p>An RL environment where agents debug broken PyTorch training scripts</p>
        </div>
        """)

        base_url = gr.Textbox(label="🌐 API Base URL", value=DEFAULT_BASE_URL, scale=1)

        with gr.Row(equal_height=False):

            # ── Left column: Task selector ──
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 📋 Task Selector")
                task_id = gr.Radio(
                    choices=["task1", "task2", "task3", "task4", "task5"],
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

                status = gr.Textbox(label="Status", interactive=False, lines=1)
                timeline = gr.HTML(label="Episode Timeline", value=_step_timeline_html([], 0))

            # ── Center column: Code editor ──
            with gr.Column(scale=2, min_width=400):
                gr.Markdown("### 💻 Code Editor")
                code_editor = gr.Code(
                    label="Your Fix (edit the code below)",
                    language="python",
                    lines=22,
                )

                with gr.Row():
                    btn_submit = gr.Button("🚀 Submit Fix", variant="primary", size="lg")

                error_log = gr.Textbox(
                    label="📋 Execution Output / Error Log",
                    lines=6,
                    interactive=False,
                )

            # ── Right column: Results ──
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 📊 Results")
                score_display = gr.HTML(value=_score_html(0.0))

                with gr.Tabs():
                    with gr.Tab("📉 Loss Curve"):
                        loss_plot = gr.Plot(label="Loss Curve")
                    with gr.Tab("📈 Val Accuracy"):
                        acc_plot = gr.Plot(label="Validation Accuracy (Task 3)")
                    with gr.Tab("🔀 Code Diff"):
                        diff_view = gr.HTML(
                            value='<div style="color:#94a3b8;text-align:center;padding:20px;">Submit a fix to see the diff</div>'
                        )
                    with gr.Tab("🧭 Current State"):
                        state_summary = gr.Markdown(
                            value="Press Reset to start an episode, then Current State will appear here."
                        )
                        btn_refresh_state = gr.Button("🔄 Refresh State", variant="secondary", size="sm")
                        state_json = gr.JSON(label="/state response", value={})

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
                    with httpx.Client(timeout=90.0) as client:
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
            outputs=[status, code_editor, task_desc, score_display, loss_plot, acc_plot, diff_view, timeline, error_log],
        ).then(
            fn=load_current_state,
            inputs=[base_url],
            outputs=[state_summary, state_json],
        )

        # Submit fix
        btn_submit.click(
            fn=do_step,
            inputs=[base_url, code_editor],
            outputs=[status, score_display, loss_plot, acc_plot, diff_view, timeline, error_log],
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

        # Baseline (auto-agent) — live streaming per-task
        TASK_NAMES = {
            "task1": "🟢 Broken Training Loop",
            "task2": "🟡 Silent NaN Loss",
            "task3": "� Label Inversion",
            "task4": "🟡 Wrong Loss Function",
            "task5": "🟡 Frozen Backbone",
        }

        def run_baseline_live(base_url_val):
            """Generator that yields live progress as each task completes."""
            base = (base_url_val or DEFAULT_BASE_URL).strip().rstrip("/")
            results = {}
            lines_header = ["### 🤖 Baseline Agent — Live Progress\n"]

            # Phase 1: Show "starting" state
            yield "\n".join(lines_header + ["⏳ Starting baseline agent..."])

            for tid in ["task1", "task2", "task3", "task4", "task5"]:
                tname = TASK_NAMES.get(tid, tid)

                # Show "running this task" update
                progress_lines = list(lines_header)
                # Show completed tasks
                for done_tid, info in results.items():
                    s = info["score"]
                    emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
                    progress_lines.append(f"- {emoji} **{TASK_NAMES.get(done_tid, done_tid)}**: {s:.4f}")
                    if info.get("error"):
                        progress_lines.append(f"  - ⚠️ `{info['error'][:150]}`")
                # Show currently running task
                progress_lines.append(f"\n🤖 **Running {tname}** — agent is analyzing the code and generating a fix...")
                progress_lines.append(f"\n*This may take 30-60 seconds per task (LLM inference + sandbox execution × 3 attempts)*")
                yield "\n".join(progress_lines)

                # Actually call the per-task endpoint
                try:
                    with httpx.Client(timeout=180.0) as client:
                        resp = client.get(f"{base}/baseline/task/{tid}")
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
            for tid in ["task1", "task2", "task3", "task4", "task5"]:
                info = results.get(tid, {"score": 0.0})
                s = info["score"]
                total += s
                emoji = "🎯" if s >= 0.9 else ("✅" if s >= 0.7 else ("📈" if s >= 0.4 else "⚠️"))
                final_lines.append(f"| {TASK_NAMES.get(tid, tid)} | {emoji} **{s:.4f}** |")
                if info.get("error"):
                    has_errors = True
                    final_lines.append(f"\n> ⚠️ `{info['error'][:200]}`\n")

            avg = total / 5
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
            for tid in ["task1", "task2", "task3", "task4", "task5"]:
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
            inputs=[base_url],
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
