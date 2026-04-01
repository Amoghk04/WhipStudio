import math
import re
import ast
from dataclasses import dataclass


@dataclass
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool
    fixed_code: str = ""


def extract_metrics_block(stdout: str) -> str:
    match = re.search(r"##METRICS_START##(.*?)##METRICS_END##", stdout, re.DOTALL)
    if match:
        return match.group(1)
    return stdout


def parse_losses(stdout: str) -> list[float]:
    stdout = extract_metrics_block(stdout)
    match = re.search(r"LOSSES:\[([^\]]+)\]", stdout)
    if not match:
        return []
    try:
        return [float(x.strip()) for x in match.group(1).split(",")]
    except Exception:
        return []


def parse_val_accs(stdout: str) -> list[float]:
    stdout = extract_metrics_block(stdout)
    match = re.search(r"VAL_ACCS:\[([^\]]+)\]", stdout)
    if not match:
        return []
    try:
        return [float(x.strip()) for x in match.group(1).split(",")]
    except Exception:
        return []


def parse_scalar(stdout: str, key: str) -> float | None:
    stdout = extract_metrics_block(stdout)
    match = re.search(rf"{key}:([-\d.eE+]+)", stdout)
    return float(match.group(1)) if match else None


def is_valid_submission(code: str, stdout: str, exit_code: int, task_id: str = "") -> tuple[bool, str]:
    """Validate submission with task-specific rules."""
    if exit_code == 0:
        if "LOSSES:" not in stdout and "FINAL_LOSS:" not in stdout and "VAL_ACCS:" not in stdout:
            return False, "No valid metrics output detected"
        if "LOSSES:" in stdout:
            losses = parse_losses(stdout)
            if len(losses) < 5:
                return False, "Fewer than 5 loss values parsed"
    
    # Task 5 doesn't require a loop - it's a single forward/backward pass
    if task_id == "task5":
        return True, ""
    
    try:
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)):
            return False, "No ast.For or ast.While node found"
    except SyntaxError:
        return False, "Code has syntax errors and cannot be parsed"
    except Exception:
        pass
    return True, ""


def sigmoid_score(value: float, center: float, steepness: float, higher_is_better: bool = True) -> float:
    """
    Compute sigmoid-based score.
    
    Args:
        value: The metric value to score
        center: The center point of the sigmoid (value at which score = 0.5)
        steepness: How quickly the score transitions around the center
        higher_is_better: If True, reward values > center. If False, reward values < center.
    
    Returns:
        Score between 0.0 and 1.0
    """
    try:
        if higher_is_better:
            x = steepness * (value - center)
        else:
            x = steepness * (center - value)
        return round(1.0 / (1.0 + math.exp(-x)), 4)
    except OverflowError:
        if higher_is_better:
            return 1.0 if value > center else 0.0
        else:
            return 1.0 if value < center else 0.0


# Keep old function for backwards compatibility but mark deprecated
def sigmoid_reward(value: float, center: float, steepness: float, invert: bool = False) -> float:
    """Deprecated: Use sigmoid_score with higher_is_better parameter instead."""
    return sigmoid_score(value, center, steepness, higher_is_better=invert)


def grade_task1(result: RunResult) -> tuple[float, dict]:
    """
    Task 1: Broken Training Loop
    Bugs: 1) lr=10.0 (too high), 2) step() before backward()
    
    Grading criteria:
    - Must have low final loss (<0.3) - indicates proper training
    - Must have high validation accuracy (>0.85) - indicates learning
    - Must show monotonic improvement - indicates proper gradient flow
    - Must NOT have loss spikes - indicates stable training
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task1")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.05, {"reason": "timed_out"}
    if result.exit_code != 0:
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    losses = parse_losses(result.stdout)
    if not losses:
        return 0.1, {"reason": "no_losses_parsed"}
    
    # Check for NaN/Inf - indicates numerical instability
    nan_count = sum(1 for loss in losses if math.isnan(loss) or math.isinf(loss))
    if nan_count > 0:
        return 0.15, {"reason": "nan_inf_found", "nan_count": nan_count}

    val_acc = parse_scalar(result.stdout, "VAL_ACC")
    if val_acc is None:
        return 0.1, {"reason": "no_val_acc"}
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    max_loss = max(losses)
    
    # Check for loss instability (spikes indicate LR too high)
    # Healthy training shouldn't have losses > 5x initial loss
    if max_loss > initial_loss * 5.0 or max_loss > 10.0:
        return 0.2, {
            "reason": "loss_unstable_spikes", 
            "max_loss": max_loss,
            "final_loss": final_loss, 
            "val_acc": val_acc
        }
    
    # Check for loss explosion at end
    if final_loss > 5.0:
        return 0.15, {"reason": "loss_unstable", "final_loss": final_loss, "val_acc": val_acc}
    
    # Primary: Validation accuracy (higher is better, target > 0.85)
    acc_score = sigmoid_score(val_acc, center=0.85, steepness=15.0, higher_is_better=True) * 0.5
    
    # Secondary: Final loss should be low (lower is better, target < 0.3)
    loss_score = sigmoid_score(final_loss, center=0.3, steepness=8.0, higher_is_better=False) * 0.3
    
    # Bonus: Monotonic improvement (loss should decrease over time)
    monotonic_bonus = 0.0
    if len(losses) >= 10:
        first_quarter = sum(losses[:len(losses)//4]) / (len(losses)//4)
        last_quarter = sum(losses[-len(losses)//4:]) / (len(losses)//4)
        if last_quarter < first_quarter * 0.7:  # At least 30% improvement
            monotonic_bonus = 0.2
    
    final_score = min(1.0, acc_score + loss_score + monotonic_bonus)
    breakdown = {
        "acc_score": round(acc_score, 4),
        "loss_score": round(loss_score, 4),
        "monotonic_bonus": monotonic_bonus,
        "val_acc": val_acc,
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "max_loss": max_loss
    }
    return final_score, breakdown


def grade_task2(result: RunResult) -> tuple[float, dict]:
    """
    Task 2: NaN Loss
    Bug: torch.log(pred) when pred can be 0.0 after sigmoid
    
    Grading criteria:
    - Must have NO NaN/Inf losses - this is the primary test
    - Must have good validation accuracy (>0.75)
    - Must show loss convergence (<0.4)
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task2")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.05, {"reason": "timed_out"}
    if result.exit_code != 0:
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    losses = parse_losses(result.stdout)
    if not losses or len(losses) < 30:
        return 0.1, {"reason": "too_few_losses"}

    nan_count = sum(1 for loss in losses if math.isnan(loss) or math.isinf(loss))
    
    # Primary criterion: NO NaN/Inf allowed - this is the core bug being tested
    nan_ratio = nan_count / len(losses)
    if nan_count > 0:
        # Heavily penalize any NaN - this is THE bug we're testing
        return max(0.05, 0.3 * (1.0 - nan_ratio)), {
            "reason": "has_nans", 
            "nan_ratio": nan_ratio,
            "nan_count": nan_count
        }
    
    val_acc = parse_scalar(result.stdout, "VAL_ACC")
    if val_acc is None:
        return 0.2, {"reason": "no_val_acc_but_no_nans"}
    
    finite_losses = [loss for loss in losses if not math.isnan(loss) and not math.isinf(loss)]
    final_loss = finite_losses[-1] if finite_losses else float('inf')
    
    # No NaN = base score of 0.4 (the bug is fixed)
    base_score = 0.4
    
    # Validation accuracy bonus (higher is better, target > 0.75)
    acc_score = sigmoid_score(val_acc, center=0.75, steepness=12.0, higher_is_better=True) * 0.35
    
    # Convergence bonus (lower is better, target < 0.4)
    convergence_score = sigmoid_score(final_loss, center=0.4, steepness=6.0, higher_is_better=False) * 0.25
    
    final_score = min(1.0, base_score + acc_score + convergence_score)
    breakdown = {
        "base_score": base_score,
        "acc_score": round(acc_score, 4),
        "convergence_score": round(convergence_score, 4),
        "nan_count": nan_count,
        "val_acc": val_acc,
        "final_loss": final_loss
    }
    return final_score, breakdown


def grade_task3(result: RunResult) -> tuple[float, dict]:
    """
    Task 3: Label Inversion
    Bug: criterion(out, 1 - yb) inverts the labels — should be criterion(out, yb)
    
    Grading criteria:
    - VAL_ACC should be high (>0.90) after 20 epochs — primary metric
    - FINAL_LOSS should be low (<0.3) — convergence indicator
    - Learning trajectory should improve over epochs
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task3")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        if "out of memory" in result.stderr.lower() or "oom" in result.stderr.lower():
            return 0.1, {"reason": "oom"}
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    val_accs = parse_val_accs(result.stdout)
    final_loss_val = parse_scalar(result.stdout, "FINAL_LOSS")

    # Memory leak check: FINAL_LOSS should be reasonable
    # With .item(), total_loss is sum of scalars (~12-20 for 20 epochs)
    memory_score = 0.0
    if final_loss_val is not None:
        memory_score = sigmoid_score(final_loss_val, center=20.0, steepness=0.2, higher_is_better=False) * 0.35
    else:
        memory_score = 0.0

    # Gradient accumulation check: accuracy should be high if training properly
    # Without zero_grad(), gradients accumulate and training degrades
    acc_score = 0.0
    final_acc = 0.0
    early_acc = 0.0
    trajectory_bonus = 0.0
    
    if val_accs and len(val_accs) >= 2:
        early_acc = sum(val_accs[:3]) / min(3, len(val_accs))
        final_acc = val_accs[-1]
        
        # Final accuracy is the main indicator of correct training
        acc_score = sigmoid_score(final_acc, center=0.8, steepness=15.0, higher_is_better=True) * 0.45
        
        # Learning trajectory: should improve over time
        if len(val_accs) >= 5:
            improvement = final_acc - early_acc
            if improvement > 0.05:
                trajectory_bonus = 0.1
            elif improvement > 0.0:
                trajectory_bonus = 0.05

    final_score = min(1.0, memory_score + acc_score + trajectory_bonus)
    breakdown = {
        "memory_score": round(memory_score, 4),
        "acc_score": round(acc_score, 4),
        "trajectory_bonus": round(trajectory_bonus, 4),
        "early_acc": round(early_acc, 4),
        "final_acc": round(final_acc, 4),
        "final_loss": final_loss_val
    }
    return final_score, breakdown


def grade_task4(result: RunResult) -> tuple[float, dict]:
    """
    Task 4: Wrong Loss (Multi-label Classification)
    Bug: Using CrossEntropyLoss instead of BCEWithLogitsLoss for multi-label
    
    Grading criteria:
    - F1 score should be high (> 0.6) - primary metric
    - avg_labels should be > 1.0 (proper multi-label output)
    - Loss should converge
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task4")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    final_loss = parse_scalar(result.stdout, "FINAL_LOSS")
    avg_labels = parse_scalar(result.stdout, "AVG_LABELS")
    f1 = parse_scalar(result.stdout, "F1_SCORE")

    # F1 score - PRIMARY metric (higher is better, target > 0.6)
    f1_score_val = 0.0
    if f1 is not None:
        f1_score_val = sigmoid_score(f1, center=0.6, steepness=10.0, higher_is_better=True) * 0.5
    
    # Multi-label check: avg_labels should be > 1.0 (proper multi-label predictions)
    # With 30% probability per class and 5 classes, expected avg ~1.5 labels/sample
    labels_score = 0.0
    if avg_labels is not None:
        if avg_labels < 0.5:
            # Way too few labels - likely single-label behavior
            labels_score = 0.0
        elif avg_labels >= 1.0:
            # Good - multiple labels per sample
            labels_score = 0.3
        else:
            # Partial credit
            labels_score = sigmoid_score(avg_labels, center=1.0, steepness=5.0, higher_is_better=True) * 0.3

    # Loss convergence (lower is better, target < 0.5)
    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_score(final_loss, center=0.5, steepness=4.0, higher_is_better=False) * 0.2

    final_score = min(1.0, f1_score_val + labels_score + loss_score)
    breakdown = {
        "f1_score": round(f1_score_val, 4),
        "labels_score": round(labels_score, 4),
        "loss_score": round(loss_score, 4),
        "avg_labels": avg_labels,
        "f1": f1,
        "final_loss": final_loss
    }
    return final_score, breakdown


def grade_task5(result: RunResult) -> tuple[float, dict]:
    """
    Task 5: Frozen Backbone with Optimizer Waste
    Bug: Backbone is frozen but still passed to optimizer (wastes memory)
    
    Valid fixes:
    1. Unfreeze backbone -> grad_norm > 0, same param count
    2. Only pass head params to optimizer -> grad_norm = 0, reduced param count
    
    The buggy code has: grad_norm = 0, param_count = 530442 (full model)
    
    Grading criteria:
    - Either backbone has gradients (unfrozen), OR
    - Optimizer param count is reduced (only head)
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task5")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    final_loss = parse_scalar(result.stdout, "FINAL_LOSS")
    grad_norm = parse_scalar(result.stdout, "BACKBONE_GRAD_NORM")
    param_count = parse_scalar(result.stdout, "OPTIMIZER_PARAM_COUNT")

    # Loss should be reasonable (10-class classification, CE loss)
    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_score(final_loss, center=2.5, steepness=2.0, higher_is_better=False) * 0.3
    
    # The bug: frozen backbone (grad_norm=0) but full params in optimizer (param_count=530442)
    # Fix 1: Unfreeze -> grad_norm > 0 (any amount)
    # Fix 2: Only head -> param_count < 100000 (head has ~5130 params)
    
    fix_score = 0.0
    fix_type = "none"
    
    if grad_norm is not None and grad_norm > 0.1:
        # Backbone is unfrozen and training
        fix_score = 0.7
        fix_type = "unfrozen"
    elif param_count is not None and param_count < 100000:
        # Only head params in optimizer (head has ~5130 params)
        fix_score = 0.7
        fix_type = "head_only"
    elif grad_norm is not None and grad_norm == 0.0 and (param_count is None or param_count > 100000):
        # Buggy state: frozen backbone but full params in optimizer
        fix_score = 0.0
        fix_type = "buggy"

    final_score = min(1.0, loss_score + fix_score)
    breakdown = {
        "loss_score": round(loss_score, 4),
        "fix_score": round(fix_score, 4),
        "fix_type": fix_type,
        "grad_norm": grad_norm,
        "param_count": param_count,
        "final_loss": final_loss
    }
    return final_score, breakdown


def score_task(task_id: str, result: RunResult) -> tuple[float, dict]:
    graders = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
        "task4": grade_task4,
        "task5": grade_task5,
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task_id: {task_id}")

    score, breakdown = graders[task_id](result)
    return round(max(0.0, min(1.0, score)), 4), breakdown
