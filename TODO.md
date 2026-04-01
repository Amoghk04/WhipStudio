# WhipStudio — Changes & Execution Guide

## What Changed

### Bug Fixes

| # | File | What was fixed |
|---|------|---------------|
| 1 | `server/tasks/graders.py` | **Task 3 grader docstring** — was labeled "Memory Leak + Missing zero_grad" but the actual bug is label inversion (`criterion(out, 1 - yb)`). Docstring now matches reality. |
| 2 | `server/tasks/graders.py` | **AST validation** — `is_valid_submission()` silently passed on `SyntaxError`, allowing invalid code to skip the training-loop check. Now returns `False` with a clear message. |
| 3 | `server/app.py` | **Task 3 metadata** — renamed from "OOM and data leakage" (hard) → "Label inversion" (medium) in the `/tasks` endpoint. |
| 4 | `baseline_agent.py` | **CLI runs all 5 tasks** — was only running `["task1", "task2", "task3"]` and dividing by `3`. Now runs all 5 and divides by `len(scores)`. |
| 5 | `gradio_app.py` | **Task 3 UI metadata** — updated name, difficulty badge, description, and hints to match label inversion bug. |
| 6 | `gradio_app.py` | **CSS not applied** — `CUSTOM_CSS` was defined but never passed to `gr.Blocks()`. Now passed as `css=CUSTOM_CSS`. |
| 7 | `openenv.yaml` | **Task 3 entry** — renamed + difficulty changed to medium. |
| 8 | `README.md` | **Task 3 row** — updated table description. Observation space now lists all 5 task IDs. |

### New Scripts

| # | File | Purpose |
|---|------|---------|
| 1 | `train_grpo.py` | GRPO training script — trains Qwen2.5-1.5B-Coder using the WhipStudio HF Space as a live reward oracle. Supports LoRA, multi-task training, and Hub upload. |
| 2 | `evaluate_mnist.py` | Evaluation script — compares base (untrained) vs GRPO-trained model on an **out-of-distribution** MNIST CNN debugging task with 3 planted bugs. Runs multiple evaluation rounds and produces a comparison table. |

---

## How to Run (Step by Step)

### Prerequisites

- Python 3.11+
- A GPU machine (A100/4090 recommended, or Colab)
- WhipStudio deployed on HuggingFace Spaces
- `HF_TOKEN` environment variable set

### Step 0: Install Dependencies

```bash
# Core project dependencies
pip install openenv-core fastapi uvicorn pydantic httpx torch smolagents

# GRPO training dependencies (on your GPU machine)
pip install trl>=0.15.0 transformers>=4.46.0 datasets accelerate peft bitsandbytes
```

### Step 1: Deploy WhipStudio to HF Spaces

Push the repo to your HF Space. The Dockerfile and start.sh are ready.
Verify the deployment:

```bash
curl https://YOUR-SPACE.hf.space/health
# Should return: {"status": "ok"}

curl https://YOUR-SPACE.hf.space/tasks
# Should list all 5 tasks
```

### Step 2: (Optional) Test the baseline agent

```bash
# Run locally if server is up
python baseline_agent.py --env-url https://YOUR-SPACE.hf.space

# Or via the API
curl https://YOUR-SPACE.hf.space/baseline
```

### Step 3: Run GRPO Training

```bash
python train_grpo.py \
    --env_url https://YOUR-SPACE.hf.space \
    --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output_dir ./whipstudio-debugger-1.5b \
    --num_iterations 30 \
    --group_size 4 \
    --samples_per_task 8 \
    --learning_rate 1e-5 \
    --use_peft \
    --lora_r 16
```

**What this does:**
1. Connects to the WhipStudio environment on HF Spaces
2. Generates 40 training prompts (8 per task × 5 tasks)
3. For each prompt, generates 4 completions (group_size)
4. Submits each completion to the environment for grading (reward 0.0–1.0)
5. Uses GRPO to reinforce high-reward completions and suppress low-reward ones
6. Saves checkpoints every 10 steps to `./whipstudio-debugger-1.5b/`

**Expected timeline:** ~2–4 hours on a single A100 (depends on Space response time).

**Flags explained:**
- `--use_peft` — Uses LoRA instead of full fine-tuning (saves VRAM)
- `--group_size 4` — 4 completions per prompt (reduce to 2 if VRAM-limited)
- `--beta 0.1` — KL divergence penalty (keeps model close to base)

### Step 4: Evaluate Base vs Trained on MNIST

```bash
python evaluate_mnist.py \
    --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --trained_model ./whipstudio-debugger-1.5b \
    --num_runs 3 \
    --output_file mnist_eval_results.json
```

**What this does:**
1. Loads a deliberately buggy MNIST CNN pipeline (NOT in the training set)
   - Bug 1: `F.softmax()` before `CrossEntropyLoss` (double softmax)
   - Bug 2: `NLLLoss` instead of `CrossEntropyLoss` (expects log probs)
   - Bug 3: Learning rate = 5.0 (way too high)
2. Asks the **base model** to fix it → executes the fix → scores it
3. Asks the **trained model** to fix it → executes the fix → scores it
4. Repeats 3 times for robustness
5. Prints a comparison table and saves full results to JSON

**Expected output:**
```
| Metric       | Base (avg)  | Trained (avg) |
|--------------|-------------|---------------|
| Score        | 0.15–0.40   | 0.60–0.90     |
| Val Accuracy | 0.10–0.30   | 0.50–0.80     |
```

### Step 5: (Optional) Push Trained Model to Hub

```bash
python train_grpo.py \
    --env_url https://YOUR-SPACE.hf.space \
    --output_dir ./whipstudio-debugger-1.5b \
    --push_to_hub \
    --hub_model_id YOUR-USERNAME/whipstudio-debugger-1.5b
```

---

## File Map

```
WhipStudio/
├── train_grpo.py          ← NEW: GRPO training script
├── evaluate_mnist.py      ← NEW: Base vs trained evaluation
├── baseline_agent.py      ← FIXED: runs all 5 tasks
├── gradio_app.py          ← FIXED: CSS applied, task3 metadata
├── server/
│   ├── app.py             ← FIXED: task3 name in /tasks endpoint
│   ├── tasks/
│   │   └── graders.py     ← FIXED: task3 docstring, AST validation
│   └── ...
├── openenv.yaml           ← FIXED: task3 name + difficulty
├── README.md              ← FIXED: task3 description
└── TODO.md                ← THIS FILE
```
