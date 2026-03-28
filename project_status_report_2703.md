# WhipStudio Project Status Report

**Date:** March 27, 2026  
**Project:** WhipStudio — ML Debugging Arena (OpenEnv AI Hackathon)  
**Status:** 🟢 **On Track for Submission**

---

## Executive Summary
WhipStudio has successfully transitioned from a conceptual prototype to a fully functional, professional-grade ML Debugging Arena. The core objective of the platform—providing an isolated, standardized environment (Gymnasium) to evaluate and train autonomous AI agents on PyTorch debugging tasks—is now fully operational. The backend conforms exactly to the OpenEnv specification, while the frontend provides a best-in-class spectator and verification experience.

---

## System Architecture & Technical Specifications

### 1. The Core Environment Database (Tasks)
The platform currently ships with three distinct ML debugging challenges, graded automatically:
- **Task 1 (Easy): Broken Training Loop.** Fixes a simple linear classifier that has an excessively high learning rate, steps the optimizer out of order, and computes loss incorrectly.
- **Task 2 (Medium): Silent NaN Loss.** Fixes a CNN training script where the output loss silently turns to `NaN`.
- **Task 3 (Hard): OOM + Data Leakage.** Fixes a complex PyTorch training loop that creates memory leaks by accumulating computation graphs incorrectly, and introduces data leakage by augmenting before splitting train/val. Graded on a granular scale (0.15 - 0.5 per bug fixed).

### 2. The Execution Sandbox API (Backend)
All agent submissions are processed securely in an isolated environment.

**Core Endpoints (OpenEnv Core API):**
- `POST /reset`: Initialize the environment for a specific task. Returns the `observation` payload, which includes the `buggy_code` string and `task_description`.
- `POST /step`: Submit a fixed code attempt (the `action`). The system executes the code in the sandbox, parses the `stdout`/`stderr` using regex metrics extractors (e.g., [parse_scalar](file:///home/amogh/Documents/openenv-comp/WhipStudio/server/tasks/graders.py#35-38) for `FINAL_LOSS`), scores the execution, logs the trajectory, and returns the formal RL `reward` (0.0 to 1.0) along with the `done` state.

**Security & Isolation ([server/sandbox.py](file:///home/amogh/Documents/openenv-comp/WhipStudio/server/sandbox.py)):**
- Python execution restricts dangerous operations using `BANNED_PATTERNS` (e.g., `os.system`, `subprocess.`, `open()`, `socket.`, `requests.`).
- Code runs in `/tmp` natively in a subprocess with a strict 30-second timeout (`TIMEOUT_SECONDS = 30`) and output sizing limits (`MAX_OUTPUT_BYTES = 8192`). 
- Environment dependencies (`PYTHONPATH`) are strictly controlled, ensuring library access (like `torch`) without host system compromise.

### 3. AI Agent Integrations
- `GET /baseline/health`: Checks if the default baseline model (Qwen-32B) is accessible and authenticated via the HuggingFace API using the `.env` configuration.
- `GET /baseline`: Instructs the backend to run the internal autonomous zero-shot agent across all three tasks simultaneously.
- `GET /baseline/task/{task_id}`: Streams execution of the baseline agent for a specific task (120s timeout). Used extensively by the Gradio UI for real-time visualization.

### 4. User Interfaces (Frontend)

WhipStudio provides dual interfaces for varying stakeholder needs:

**A. The "ML Debug Arena" Verification Portal (Gradio App)**
A custom-built, interactive dashboard designed for human verification and hackathon judges. 
- **Code Editor:** Monaco-style interactive Python editor pre-filled with the buggy code payload.
- **Live Loss Visualization:** Integrates `matplotlib` to plot `LOSS` curves and `VAL_ACCURACY` step-by-step from the agent's debugged submission.
- **Live Diff Context:** A source-control style diff view showing exactly what code lines the agent inserted or deleted to achieve the fix.
- **Autonomous Streaming Mode:** The "Auto-Agent" trigger connects to the `/baseline/task/{task_id}` streams to provide a live "Chain-of-Thought" experience, displaying progress and rewards directly to the UI panel in real-time.

**B. The OpenEnv Compliance Web UI (`/web`)**
- Directly embedded via the OpenEnv python framework (`os.environ["ENABLE_WEB_INTERFACE"] = "true"`).
- Accessed via `http://[host]:8000/web`, this provides the raw, standardized OpenEnv chat-style UI validating 100% adherence to the hackathon's core platform requirements.

---

## Completed Milestones

1. **Bug Remediation:** Fixed critical parsing logic failures where double-escaped regex patterns (`[\\\\d\\\\.\\\\-]+`) prevented the extraction of metrics like `FINAL_LOSS:0.4523`. Also resolved the `ModuleNotFoundError: torch` in the sandbox.
2. **Environment Token Logic:** Re-wrote authentication management using `python-dotenv` for local caching overrides, ensuring 401 Unauthorized API failures do not disrupt model inference when tokens expire mid-session.
3. **Task Trajectory Generation:** Enabled persistent trajectory state logging inside `environment.py` for advanced analytics and training extraction.

---

## Strategic Next Steps

**Phase 2: Reinforcement Learning Integration**
The zero-shot inference pipeline currently scores dynamically. The immediate next phase leverages the Trajectory Logic:
1. Export the stored trajectory JSON logs containing `buggy_code`, `reward`, and `error_log`.
2. Integrate the **TRL (Transformer Reinforcement Learning) library**.
3. Implement Group Relative Policy Optimization (GRPO) to iteratively train a specific lightweight model (like a 7B parameter local model) to perform significantly better on PyTorch debugging tasks than larger, generalized zero-shot models.
