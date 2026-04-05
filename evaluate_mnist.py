#!/usr/bin/env python3
"""
Evaluate untrained vs GRPO-trained Qwen2.5-1.5B-Coder on a real
MNIST handwritten digit recognition debugging task.

This script demonstrates that RL-trained models outperform base models
on out-of-distribution ML debugging tasks.

The MNIST debugging task is intentionally NOT in the WhipStudio training set,
making it a true test of generalization.

Workflow:
1. Define a deliberately buggy MNIST training pipeline
2. Load both base model and GRPO-fine-tuned model
3. Ask each to fix the buggy code
4. Execute both fixes and compare results
5. Generate a comparison report

Requirements:
    pip install transformers torch peft bitsandbytes

Usage:
    # Basic comparison
    python evaluate_mnist.py \
        --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --trained_model ./whipstudio-debugger/best

    # Multiple runs for statistical significance
    python evaluate_mnist.py \
        --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --trained_model ./whipstudio-debugger/best \
        --num_runs 5

    # Use 4-bit quantization for memory efficiency
    python evaluate_mnist.py \
        --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --trained_model ./whipstudio-debugger/best \
        --use_4bit
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
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Optional PEFT import for LoRA models
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# System Prompt (same as training)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert PyTorch debugging agent.
You receive a broken training script and must fix ALL bugs.
Return ONLY the complete corrected Python code. No markdown, no backticks, no explanation.
Keep all torch.manual_seed() calls intact."""


# ══════════════════════════════════════════════════════════════════════════════
# Buggy MNIST Pipeline (Out-of-Distribution Test)
# ══════════════════════════════════════════════════════════════════════════════

# Two versions of the buggy code: synthetic (fast) and real MNIST (realistic)

MNIST_BUGGY_CODE_SYNTHETIC = '''
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

MNIST_BUGGY_CODE_REAL = '''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

torch.manual_seed(42)

# Load REAL MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Use subset for faster training (5000 train, 1000 val)
train_indices = torch.randperm(len(train_dataset))[:5000]
val_indices = torch.randperm(len(test_dataset))[:1000]

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(test_dataset, val_indices)

train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

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
for epoch in range(10):  # 10 epochs on real MNIST
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Validation on real MNIST test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for xb, yb in val_loader:
        out = model(xb)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

val_acc = correct / total

print('##METRICS_START##')
print('LOSSES:' + str(losses[-100:]))  # Last 100 losses to avoid huge output
print('VAL_ACC:' + str(round(val_acc, 4)))
print('##METRICS_END##')
'''

# Default to synthetic for backward compatibility
MNIST_BUGGY_CODE = MNIST_BUGGY_CODE_SYNTHETIC

MNIST_TASK_DESCRIPTION_SYNTHETIC = """
This is a CNN-based handwritten digit classifier (MNIST-like, 10 classes).
The model has several bugs preventing it from training properly.

Bugs to identify and fix:
1. The forward pass has a problem with activation functions
2. The loss function doesn't match the model output
3. The optimizer has problematic hyperparameters

Fix ALL bugs so that after 20 epochs:
- Loss converges below 1.5
- Validation accuracy exceeds 0.50

Print losses as: LOSSES:[val1, val2, ...]
Print validation accuracy as: VAL_ACC:X.XX
Wrap metrics in ##METRICS_START## and ##METRICS_END##.
"""

MNIST_TASK_DESCRIPTION_REAL = """
This is a CNN-based MNIST handwritten digit classifier using the REAL MNIST dataset.
The model has several bugs preventing it from training properly.

Bugs to identify and fix:
1. The forward pass has a problem with activation functions
2. The loss function doesn't match the model output  
3. The optimizer has problematic hyperparameters

Fix ALL bugs so that after 10 epochs on real MNIST:
- Loss converges and decreases over time
- Validation accuracy exceeds 0.85 (should be achievable on real MNIST)

Print the last 100 losses as: LOSSES:[val1, val2, ...]
Print validation accuracy as: VAL_ACC:X.XX
Wrap metrics in ##METRICS_START## and ##METRICS_END##.
"""

MNIST_TASK_DESCRIPTION = MNIST_TASK_DESCRIPTION_SYNTHETIC


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    model_path: str,
    use_4bit: bool = False,
    is_peft: bool = False,
    base_model_for_peft: Optional[str] = None,
) -> tuple:
    """Load model and tokenizer with optional quantization and PEFT."""
    
    print(f"  Loading model from {model_path}...")
    
    # Quantization config
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Model kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if this is a PEFT/LoRA model
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists() or is_peft:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT model detected but peft is not installed")
        
        # For PEFT models, we need to load base model first
        if base_model_for_peft is None:
            # Try to read from adapter config
            if adapter_config_path.exists():
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                    base_model_for_peft = adapter_config.get("base_model_name_or_path")
        
        if base_model_for_peft is None:
            raise ValueError("PEFT model requires --base_model_for_peft or adapter_config.json with base_model_name_or_path")
        
        print(f"  Loading base model: {base_model_for_peft}")
        base_model = AutoModelForCausalLM.from_pretrained(base_model_for_peft, **model_kwargs)
        
        print(f"  Loading PEFT adapters from: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Regular model
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    return model, tokenizer


def generate_fix(model, tokenizer, task_description: str, buggy_code: str) -> str:
    """Generate a fix using the given model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {task_description}\n\nBuggy code:\n{buggy_code}"},
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    # Strip markdown fences if present
    if "```python" in response:
        response = response.split("```python", 1)[1].split("```", 1)[0].strip()
    elif "```" in response:
        response = response.split("```", 1)[1].split("```", 1)[0].strip()

    return response.strip()


def execute_code(code: str, timeout: int = 120) -> dict:
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

    # Val accuracy (50% of score)
    val_acc = metrics.get("val_acc")
    if val_acc is not None:
        if val_acc >= 0.7:
            score += 0.50
        elif val_acc >= 0.5:
            score += 0.35
        elif val_acc >= 0.3:
            score += 0.15

    # Final loss (30% of score)
    final_loss = metrics.get("final_loss")
    if final_loss is not None:
        if final_loss < 1.0:
            score += 0.30
        elif final_loss < 1.5:
            score += 0.20
        elif final_loss < 2.5:
            score += 0.10

    # Learning trajectory (20% of score)
    losses = metrics.get("losses", [])
    if len(losses) >= 10:
        first_q = sum(losses[:len(losses) // 4]) / max(1, len(losses) // 4)
        last_q = sum(losses[-len(losses) // 4:]) / max(1, len(losses) // 4)
        if last_q < first_q * 0.7:
            score += 0.20
        elif last_q < first_q:
            score += 0.10

    return min(1.0, score)


def evaluate_single_model(
    model_path: str,
    label: str,
    use_4bit: bool = False,
    is_peft: bool = False,
    base_model_for_peft: Optional[str] = None,
    use_real_mnist: bool = False,
) -> dict:
    """Load a model, generate a fix, execute it, and return results."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label}")
    print(f"  Model: {model_path}")
    print(f"  Dataset: {'Real MNIST' if use_real_mnist else 'Synthetic'}")
    print(f"{'=' * 60}")

    # Select appropriate buggy code and task description
    if use_real_mnist:
        buggy_code = MNIST_BUGGY_CODE_REAL
        task_desc = MNIST_TASK_DESCRIPTION_REAL
    else:
        buggy_code = MNIST_BUGGY_CODE_SYNTHETIC
        task_desc = MNIST_TASK_DESCRIPTION_SYNTHETIC

    # Load model
    model, tokenizer = load_model(
        model_path,
        use_4bit=use_4bit,
        is_peft=is_peft,
        base_model_for_peft=base_model_for_peft,
    )

    # Generate fix
    print("  Generating fix...")
    start = time.time()
    fixed_code = generate_fix(model, tokenizer, task_desc, buggy_code)
    gen_time = time.time() - start
    print(f"  Generation took {gen_time:.1f}s ({len(fixed_code)} chars)")

    # Execute (longer timeout for real MNIST due to dataset download)
    timeout = 300 if use_real_mnist else 120
    print(f"  Executing fixed code (timeout={timeout}s)...")
    result = execute_code(fixed_code, timeout=timeout)
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


def print_comparison_table(base_results: list, trained_results: list):
    """Print a nicely formatted comparison table."""
    # Aggregate scores
    base_scores = [r["score"] for r in base_results]
    trained_scores = [r["score"] for r in trained_results]
    base_accs = [r["metrics"].get("val_acc", 0) or 0 for r in base_results]
    trained_accs = [r["metrics"].get("val_acc", 0) or 0 for r in trained_results]

    avg_base_score = sum(base_scores) / len(base_scores)
    avg_trained_score = sum(trained_scores) / len(trained_scores)
    avg_base_acc = sum(base_accs) / len(base_accs)
    avg_trained_acc = sum(trained_accs) / len(trained_accs)

    # Table
    print(f"\n{'=' * 70}")
    print(f"{'COMPARISON: Base vs GRPO-Trained Model':^70}")
    print(f"{'=' * 70}")

    headers = ["Metric", "Base Model", "Trained Model", "Δ (Improvement)"]
    rows = [
        ["Average Score", f"{avg_base_score:.4f}", f"{avg_trained_score:.4f}", 
         f"{avg_trained_score - avg_base_score:+.4f}"],
        ["Average Val Acc", f"{avg_base_acc:.4f}", f"{avg_trained_acc:.4f}",
         f"{avg_trained_acc - avg_base_acc:+.4f}"],
        ["Best Score", f"{max(base_scores):.4f}", f"{max(trained_scores):.4f}",
         f"{max(trained_scores) - max(base_scores):+.4f}"],
        ["Best Val Acc", f"{max(base_accs):.4f}", f"{max(trained_accs):.4f}",
         f"{max(trained_accs) - max(base_accs):+.4f}"],
        ["Success Rate (>0.5)", f"{sum(1 for s in base_scores if s > 0.5)}/{len(base_scores)}",
         f"{sum(1 for s in trained_scores if s > 0.5)}/{len(trained_scores)}", ""],
    ]

    # Calculate column widths
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(4)]
    
    # Print table
    header_line = "│ " + " │ ".join(h.center(w) for h, w in zip(headers, col_widths)) + " │"
    sep_line = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    top_line = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    bottom_line = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
    
    print(top_line)
    print(header_line)
    print(sep_line)
    for row in rows:
        print("│ " + " │ ".join(str(v).center(w) for v, w in zip(row, col_widths)) + " │")
    print(bottom_line)

    # Winner announcement
    print()
    if avg_trained_score > avg_base_score:
        delta = avg_trained_score - avg_base_score
        pct = (delta / max(avg_base_score, 0.001)) * 100
        print(f"🏆 GRPO-trained model wins by +{delta:.4f} score ({pct:.1f}% improvement)!")
    elif avg_base_score > avg_trained_score:
        print(f"⚠️  Base model performed better (may need more training)")
    else:
        print(f"🤝 Models tied on average score")

    return {
        "base_avg_score": avg_base_score,
        "trained_avg_score": avg_trained_score,
        "base_avg_acc": avg_base_acc,
        "trained_avg_acc": avg_trained_acc,
        "improvement_score": avg_trained_score - avg_base_score,
        "improvement_acc": avg_trained_acc - avg_base_acc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare multiple models on MNIST debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare base vs trained model
  python evaluate_mnist.py --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct --trained_model ./trained

  # Use real MNIST dataset
  python evaluate_mnist.py --use_real_mnist --base_model Qwen/Qwen2.5-Coder-1.5B-Instruct

  # Compare multiple models
  python evaluate_mnist.py --models Qwen/Qwen2.5-Coder-1.5B-Instruct ./trained-v1 ./trained-v2

  # Memory-efficient evaluation
  python evaluate_mnist.py --use_4bit --base_model Qwen/Qwen2.5-Coder-7B-Instruct
        """
    )
    
    # Model selection (flexible)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Path or HF name of base model")
    parser.add_argument("--trained_model", type=str, default=None,
                        help="Path to GRPO-trained model (optional if using --models)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="List of models to compare (overrides --base_model and --trained_model)")
    
    # Dataset options
    parser.add_argument("--use_real_mnist", action="store_true",
                        help="Use real MNIST dataset (downloads ~50MB, slower but more realistic)")
    
    # Output
    parser.add_argument("--output_file", type=str, default="mnist_eval_results.json",
                        help="Output file for detailed results")
    parser.add_argument("--num_runs", type=int, default=3,
                        help="Number of evaluation runs per model")
    
    # Memory options
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--trained_is_peft", action="store_true",
                        help="Trained model is a PEFT/LoRA adapter")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_type = "Real MNIST" if args.use_real_mnist else "Synthetic MNIST-like"
    
    print(f"\n{'#' * 70}")
    print(f"{'MNIST DEBUGGING EVALUATION':^70}")
    print(f"{'#' * 70}")
    print(f"\nDevice: {device}")
    print(f"Dataset: {dataset_type}")
    print(f"Runs per model: {args.num_runs}")
    print(f"\nMNIST Debugging Task (out-of-distribution):")
    print(f"  Bugs: softmax before CE, NLLLoss without log, LR=5.0")

    # Determine which models to evaluate
    if args.models:
        # Multi-model comparison mode
        model_list = args.models
        print(f"\nModels to compare ({len(model_list)}):")
        for i, m in enumerate(model_list, 1):
            print(f"  {i}. {m}")
    else:
        # Legacy two-model comparison
        model_list = [args.base_model]
        if args.trained_model:
            model_list.append(args.trained_model)
        print(f"\nBase model: {args.base_model}")
        if args.trained_model:
            print(f"Trained model: {args.trained_model}")

    # Run evaluations for each model
    all_results = {model: [] for model in model_list}
    
    for run in range(1, args.num_runs + 1):
        print(f"\n{'─' * 70}")
        print(f"Run {run}/{args.num_runs}")
        print(f"{'─' * 70}")
        
        for model_path in model_list:
            model_name = Path(model_path).name if "/" not in model_path else model_path.split("/")[-1]
            
            # Determine if this is a PEFT model
            is_peft = args.trained_is_peft and model_path != args.base_model
            base_for_peft = args.base_model if is_peft else None
            
            result = evaluate_single_model(
                model_path,
                f"{model_name} (run {run})",
                use_4bit=args.use_4bit,
                is_peft=is_peft,
                base_model_for_peft=base_for_peft,
                use_real_mnist=args.use_real_mnist,
            )
            all_results[model_path].append(result)

    # Print comparison table for all models
    print(f"\n{'=' * 80}")
    print(f"{'RESULTS SUMMARY':^80}")
    print(f"{'=' * 80}")
    
    # Calculate aggregates for each model
    model_stats = {}
    for model_path, results in all_results.items():
        scores = [r["score"] for r in results]
        accs = [r["metrics"].get("val_acc", 0) or 0 for r in results]
        model_stats[model_path] = {
            "avg_score": sum(scores) / len(scores),
            "avg_acc": sum(accs) / len(accs),
            "best_score": max(scores),
            "best_acc": max(accs),
            "success_rate": sum(1 for s in scores if s > 0.5) / len(scores),
        }
    
    # Print table
    headers = ["Model", "Avg Score", "Avg Acc", "Best Score", "Success Rate"]
    rows = []
    for model_path, stats in model_stats.items():
        model_name = Path(model_path).name if "/" not in model_path else model_path.split("/")[-1]
        rows.append([
            model_name[:25],  # Truncate long names
            f"{stats['avg_score']:.4f}",
            f"{stats['avg_acc']:.4f}",
            f"{stats['best_score']:.4f}",
            f"{stats['success_rate']*100:.0f}%",
        ])
    
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    
    print("┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐")
    print("│ " + " │ ".join(h.center(w) for h, w in zip(headers, col_widths)) + " │")
    print("├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤")
    for row in rows:
        print("│ " + " │ ".join(str(v).center(w) for v, w in zip(row, col_widths)) + " │")
    print("└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘")

    # Find winner
    best_model = max(model_stats.items(), key=lambda x: x[1]["avg_score"])
    print(f"\n🏆 Best model: {best_model[0].split('/')[-1]} (avg score: {best_model[1]['avg_score']:.4f})")

    # Legacy comparison if exactly 2 models
    summary = None
    if len(model_list) == 2:
        base_results = all_results[model_list[0]]
        trained_results = all_results[model_list[1]]
        summary = print_comparison_table(base_results, trained_results)

    # Save detailed results
    output = {
        "task": f"MNIST debugging ({dataset_type})",
        "models": model_list,
        "num_runs": args.num_runs,
        "device": device,
        "use_real_mnist": args.use_real_mnist,
        "model_stats": model_stats,
        "summary": summary,
        "runs": {
            model_path: [
                {k: v for k, v in r.items() if k != "fixed_code"}
                for r in results
            ]
            for model_path, results in all_results.items()
        },
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📄 Full results saved to {args.output_file}")


if __name__ == "__main__":
    main()
