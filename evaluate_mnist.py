"""
Evaluate untrained vs GRPO-trained Qwen2.5-1.5B-Coder on a real
MNIST handwritten digit recognition debugging task.

This script:
1. Defines a deliberately buggy MNIST training pipeline (NOT in training set)
2. Asks both the base model and the fine-tuned model to fix it
3. Executes both fixes and compares results

Requirements:
    pip install transformers torch

Usage:
    python evaluate_mnist.py \
        --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --trained_model ./whipstudio-debugger-1.5b
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs.
Return ONLY the complete corrected Python code. No markdown, no backticks, no explanation.
Keep all torch.manual_seed() calls intact."""

# ── Buggy MNIST pipeline (out-of-distribution from training tasks) ──────

MNIST_BUGGY_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

# Simulate MNIST-like data (28x28 images, 10 classes)
X_train = torch.randn(1000, 1, 28, 28)
y_train = torch.randint(0, 10, (1000,))
X_val = torch.randn(200, 1, 28, 28)
y_val = torch.randint(0, 10, (200,))

# Make data learnable: label = argmax of mean pixel value in 10 regions
for i in range(len(X_train)):
    region_means = X_train[i, 0].reshape(10, -1).mean(dim=1)
    y_train[i] = region_means.argmax()
for i in range(len(X_val)):
    region_means = X_val[i, 0].reshape(10, -1).mean(dim=1)
    y_val[i] = region_means.argmax()

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # BUG 1: Applying softmax before CrossEntropyLoss (double softmax)
        x = F.softmax(self.fc2(x), dim=1)
        return x

model = SimpleCNN()

# BUG 2: Using NLLLoss without log_softmax (expects log probabilities)
criterion = nn.NLLLoss()

# BUG 3: Learning rate too high for CNN
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)

losses = []
for epoch in range(20):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Validation
model.eval()
with torch.no_grad():
    val_out = model(X_val)
    val_preds = val_out.argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean().item()

print('##METRICS_START##')
print('LOSSES:' + str(losses))
print('VAL_ACC:' + str(round(val_acc, 4)))
print('##METRICS_END##')
'''

MNIST_TASK_DESCRIPTION = """
This is a CNN-based handwritten digit classifier (MNIST-like, 10 classes).
The model has several bugs preventing it from training properly.
Fix ALL bugs so that after 20 epochs:
- Loss converges below 1.5
- Validation accuracy exceeds 0.50
Print losses as: LOSSES:[val1, val2, ...]
Print validation accuracy as: VAL_ACC:X.XX
Wrap metrics in ##METRICS_START## and ##METRICS_END##.
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def generate_fix(model, tokenizer, task_description: str, buggy_code: str, device: str = "cuda") -> str:
    """Generate a fix using the given model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_description}\n\nBuggy code:\n{buggy_code}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (not the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Strip markdown fences if present
    if "```python" in response:
        response = response.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in response:
        response = response.split("```", 1)[1].split("```", 1)[0].strip()

    return response.strip()


def execute_code(code: str, timeout: int = 60) -> dict:
    """Execute code in a subprocess and return results."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[:8192],
            "stderr": proc.stderr[:2048],
            "elapsed": round(elapsed, 2),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Timed out after {timeout}s",
            "elapsed": timeout,
            "timed_out": True,
        }
    finally:
        os.unlink(tmp_path)


def extract_metrics(stdout: str) -> dict:
    """Parse metrics from stdout."""
    metrics: dict = {}

    # Extract metrics block if present
    block_match = re.search(r"##METRICS_START##(.*?)##METRICS_END##", stdout, re.DOTALL)
    text = block_match.group(1) if block_match else stdout

    # Parse losses
    match = re.search(r"LOSSES:\[([^\]]+)\]", text)
    if match:
        try:
            losses = [float(x.strip()) for x in match.group(1).split(",")]
            metrics["losses"] = losses
            metrics["final_loss"] = losses[-1] if losses else None
            metrics["initial_loss"] = losses[0] if losses else None
            metrics["nan_count"] = sum(1 for l in losses if math.isnan(l) or math.isinf(l))
            metrics["num_steps"] = len(losses)
        except Exception:
            pass

    # Parse val_acc
    match = re.search(r"VAL_ACC:([\d.]+)", text)
    if match:
        metrics["val_acc"] = float(match.group(1))

    return metrics


def score_mnist_fix(metrics: dict) -> float:
    """
    Score an MNIST fix on a 0-1 scale.
    Criteria:
    - No NaN/Inf (base requirement)
    - Final loss < 1.5  (30%)
    - Val accuracy > 0.5 (50%)
    - Learning trajectory (20%)
    """
    if not metrics:
        return 0.0

    if metrics.get("nan_count", 0) > 0:
        return 0.05

    score = 0.0

    # Val accuracy
    val_acc = metrics.get("val_acc")
    if val_acc is not None:
        if val_acc >= 0.7:
            score += 0.50
        elif val_acc >= 0.5:
            score += 0.35
        elif val_acc >= 0.3:
            score += 0.15

    # Final loss
    final_loss = metrics.get("final_loss")
    if final_loss is not None:
        if final_loss < 1.0:
            score += 0.30
        elif final_loss < 1.5:
            score += 0.20
        elif final_loss < 2.5:
            score += 0.10

    # Learning trajectory
    losses = metrics.get("losses", [])
    if len(losses) >= 10:
        first_q = sum(losses[:len(losses) // 4]) / max(1, len(losses) // 4)
        last_q = sum(losses[-len(losses) // 4:]) / max(1, len(losses) // 4)
        if last_q < first_q * 0.7:
            score += 0.20
        elif last_q < first_q:
            score += 0.10

    return min(1.0, score)


def evaluate_model(model_path: str, label: str, device: str = "cuda") -> dict:
    """Load a model, generate a fix, execute it, and return results."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"  Model: {model_path}")
    print(f"{'=' * 60}")

    # Load
    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Generate fix
    print("  Generating fix...")
    start = time.time()
    fixed_code = generate_fix(model, tokenizer, MNIST_TASK_DESCRIPTION, MNIST_BUGGY_CODE, device)
    gen_time = time.time() - start
    print(f"  Generation took {gen_time:.1f}s ({len(fixed_code)} chars)")

    # Execute
    print("  Executing fixed code...")
    result = execute_code(fixed_code)
    metrics = extract_metrics(result["stdout"])
    score = score_mnist_fix(metrics) if result["exit_code"] == 0 else 0.0

    # Report
    print(f"\n  Results for {label}:")
    print(f"    Exit code:    {result['exit_code']}")
    print(f"    Timed out:    {result['timed_out']}")
    print(f"    Val accuracy: {metrics.get('val_acc', 'N/A')}")
    print(f"    Final loss:   {metrics.get('final_loss', 'N/A')}")
    print(f"    NaN count:    {metrics.get('nan_count', 'N/A')}")
    print(f"    Score:        {score:.4f}")

    if result["stderr"] and result["exit_code"] != 0:
        print(f"    Stderr: {result['stderr'][:500]}")

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": label,
        "model_path": model_path,
        "fixed_code": fixed_code,
        "execution": result,
        "metrics": metrics,
        "score": score,
        "generation_time": gen_time,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs trained model on MNIST debugging")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--trained_model", type=str, default="./whipstudio-debugger-1.5b")
    parser.add_argument("--output_file", type=str, default="mnist_eval_results.json")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of evaluation runs per model (averaged for robustness)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\nMNIST Debugging Task (out-of-distribution):")
    print(f"  Bugs: softmax before CrossEntropyLoss, NLLLoss without log, LR=5.0")
    print(f"  Runs per model: {args.num_runs}")

    # Run evaluations
    base_results = []
    trained_results = []

    for run in range(1, args.num_runs + 1):
        print(f"\n--- Run {run}/{args.num_runs} ---")
        base_results.append(evaluate_model(args.base_model, f"Base (run {run})", device))
        trained_results.append(evaluate_model(args.trained_model, f"Trained (run {run})", device))

    # Aggregate
    base_scores = [r["score"] for r in base_results]
    trained_scores = [r["score"] for r in trained_results]
    base_accs = [r["metrics"].get("val_acc", 0) or 0 for r in base_results]
    trained_accs = [r["metrics"].get("val_acc", 0) or 0 for r in trained_results]

    avg_base_score = sum(base_scores) / len(base_scores)
    avg_trained_score = sum(trained_scores) / len(trained_scores)
    avg_base_acc = sum(base_accs) / len(base_accs)
    avg_trained_acc = sum(trained_accs) / len(trained_accs)

    # Table
    print(f"\n{'=' * 60}")
    print(f"COMPARISON: Base vs GRPO-Trained ({args.num_runs} runs)")
    print(f"{'=' * 60}")

    headers = ["Metric", "Base (avg)", "Trained (avg)"]
    rows = [
        ["Score", f"{avg_base_score:.4f}", f"{avg_trained_score:.4f}"],
        ["Val Accuracy", f"{avg_base_acc:.4f}", f"{avg_trained_acc:.4f}"],
        ["Best Score", f"{max(base_scores):.4f}", f"{max(trained_scores):.4f}"],
        ["Best Val Acc", f"{max(base_accs):.4f}", f"{max(trained_accs):.4f}"],
    ]

    col_widths = [max(len(r[i]) for r in [headers] + rows) + 2 for i in range(3)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    print(header_line)
    print(sep_line)
    for row in rows:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, col_widths)) + " |")

    # Winner
    if avg_trained_score > avg_base_score:
        delta = avg_trained_score - avg_base_score
        print(f"\nGRPO-trained model wins by +{delta:.4f} score!")
    elif avg_base_score > avg_trained_score:
        print(f"\nBase model performed better (may need more training)")
    else:
        print(f"\nModels tied on score")

    # Save full results
    output = {
        "task": "MNIST digit debugging (out-of-distribution)",
        "num_runs": args.num_runs,
        "summary": {
            "base_avg_score": avg_base_score,
            "trained_avg_score": avg_trained_score,
            "base_avg_acc": avg_base_acc,
            "trained_avg_acc": avg_trained_acc,
            "improvement_score": avg_trained_score - avg_base_score,
            "improvement_acc": avg_trained_acc - avg_base_acc,
        },
        "runs": {
            "base": [
                {k: v for k, v in r.items() if k != "fixed_code"}
                for r in base_results
            ],
            "trained": [
                {k: v for k, v in r.items() if k != "fixed_code"}
                for r in trained_results
            ],
        },
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output_file}")


if __name__ == "__main__":
    main()
