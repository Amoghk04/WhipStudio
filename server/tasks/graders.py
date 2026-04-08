import ast
import math
import re
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
        Score between 0.0001 and 0.9999 (strictly within (0, 1))
    """
    try:
        if higher_is_better:
            x = steepness * (value - center)
        else:
            x = steepness * (center - value)
        return round(1.0 / (1.0 + math.exp(-x)), 4)
    except OverflowError:
        # Return near-boundary values, not exact 0.0 or 1.0
        if higher_is_better:
            return 0.9999 if value > center else 0.1
        else:
            return 0.9999 if value < center else 0.1


# Keep old function for backwards compatibility but mark deprecated
def sigmoid_reward(value: float, center: float, steepness: float, invert: bool = False) -> float:
    """Deprecated: Use sigmoid_score with higher_is_better parameter instead."""
    return sigmoid_score(value, center, steepness, higher_is_better=invert)


def grade_task1(result: RunResult) -> tuple[float, dict]:
    """
    Task 1: Broken Training Loop
    Bugs: 1) lr=10.0 (too high), 2) step() before backward()
    
    Grading criteria (STRICT thresholds for differentiation):
    - VAL_ACC > 0.90 required for high score (target is >0.85)
    - Final loss < 0.2 required for high score (target is <0.3)
    - Must show monotonic improvement
    - Penalize any instability heavily
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
    
    # Check for NaN/Inf - indicates numerical instability (LR bug not fully fixed)
    nan_count = sum(1 for loss in losses if math.isnan(loss) or math.isinf(loss))
    if nan_count > 0:
        return 0.1, {"reason": "nan_inf_found", "nan_count": nan_count}

    val_acc = parse_scalar(result.stdout, "VAL_ACC")
    if val_acc is None:
        return 0.1, {"reason": "no_val_acc"}
    
    final_loss = losses[-1]
    initial_loss = losses[0]
    max_loss = max(losses)
    
    # STRICT: Check for loss instability (spikes indicate LR still too high)
    if max_loss > initial_loss * 3.0 or max_loss > 5.0:
        return 0.15, {
            "reason": "loss_unstable_spikes", 
            "max_loss": max_loss,
            "final_loss": final_loss, 
            "val_acc": val_acc
        }
    
    # STRICT: Loss must converge well
    if final_loss > 1.0:
        return 0.2, {"reason": "loss_not_converged", "final_loss": final_loss, "val_acc": val_acc}
    
    # STRICT thresholds - center points raised for better differentiation
    # Target: val_acc > 0.90, final_loss < 0.15
    
    # Primary: Validation accuracy (60% weight)
    # Use steeper sigmoid for sharper differentiation
    acc_score = sigmoid_score(val_acc, center=0.90, steepness=25.0, higher_is_better=True) * 0.60
    
    # Secondary: Final loss (30% weight) - must be low
    loss_score = sigmoid_score(final_loss, center=0.15, steepness=15.0, higher_is_better=False) * 0.30
    
    # Bonus: Monotonic improvement - must be significant
    monotonic_bonus = 0.0
    if len(losses) >= 10:
        first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
        last_half = sum(losses[-len(losses)//2:]) / (len(losses)//2)
        improvement_ratio = (first_half - last_half) / first_half if first_half > 0 else 0
        if improvement_ratio > 0.5:  # >50% improvement required
            monotonic_bonus = 0.10
        elif improvement_ratio > 0.3:
            monotonic_bonus = 0.05
    
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
    
    Grading criteria (STRICT - NaN elimination is PRIMARY):
    - ZERO NaN/Inf required (this is the bug!)
    - VAL_ACC > 0.80 required for high score
    - Loss must converge < 0.3
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
    
    # PRIMARY: NO NaN/Inf at ALL - this is THE bug being tested
    nan_ratio = nan_count / len(losses)
    if nan_count > 0:
        # STRICT: Any NaN = major failure (max 0.25 score)
        return max(0.05, 0.25 * (1.0 - nan_ratio)), {
            "reason": "has_nans", 
            "nan_ratio": nan_ratio,
            "nan_count": nan_count
        }
    
    val_acc = parse_scalar(result.stdout, "VAL_ACC")
    if val_acc is None:
        return 0.25, {"reason": "no_val_acc_but_no_nans"}
    
    finite_losses = [loss for loss in losses if not math.isnan(loss) and not math.isinf(loss)]
    final_loss = finite_losses[-1] if finite_losses else float('inf')
    
    # No NaN = base score of 0.35 (bug is fixed but need to verify quality)
    base_score = 0.35
    
    # STRICT: Validation accuracy (40% weight, center at 0.80)
    acc_score = sigmoid_score(val_acc, center=0.80, steepness=20.0, higher_is_better=True) * 0.40
    
    # STRICT: Convergence (25% weight, center at 0.25)
    convergence_score = sigmoid_score(final_loss, center=0.25, steepness=10.0, higher_is_better=False) * 0.25
    
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
    Task 3: Label Inversion Bug
    Bug: criterion(out, 1 - yb) inverts labels — should be criterion(out, yb)
    
    Grading criteria (STRICT - accuracy is PRIMARY):
    - VAL_ACC > 0.90 required (buggy code gives ~0.50)
    - FINAL_LOSS < 0.25 required
    - Must show learning trajectory improvement
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

    # CRITICAL CHECK: Buggy code produces ~0.50 accuracy (random)
    # Fixed code should produce >0.90 accuracy
    
    acc_score = 0.0
    final_acc = 0.0
    early_acc = 0.0
    trajectory_bonus = 0.0
    
    if not val_accs or len(val_accs) < 2:
        return 0.15, {"reason": "no_val_accs_parsed"}
    
    early_acc = sum(val_accs[:3]) / min(3, len(val_accs))
    final_acc = val_accs[-1]
    
    # STRICT: Final accuracy must be high (>0.90 target)
    # The bug makes accuracy ~0.50, so anything <0.70 is likely unfixed
    if final_acc < 0.65:
        return 0.15, {
            "reason": "accuracy_too_low_likely_unfixed",
            "final_acc": final_acc,
            "expected": ">0.90 for fixed code"
        }
    
    # Primary: Final accuracy (60% weight, center at 0.92)
    acc_score = sigmoid_score(final_acc, center=0.92, steepness=30.0, higher_is_better=True) * 0.60
    
    # Secondary: Loss convergence (25% weight)
    loss_score = 0.0
    if final_loss_val is not None:
        loss_score = sigmoid_score(final_loss_val, center=0.20, steepness=12.0, higher_is_better=False) * 0.25
    
    # Bonus: Learning trajectory (15% weight)
    if len(val_accs) >= 5:
        improvement = final_acc - early_acc
        if improvement > 0.15:  # Significant learning
            trajectory_bonus = 0.15
        elif improvement > 0.05:
            trajectory_bonus = 0.08
        elif improvement > 0.0:
            trajectory_bonus = 0.03

    final_score = min(1.0, acc_score + loss_score + trajectory_bonus)
    breakdown = {
        "acc_score": round(acc_score, 4),
        "loss_score": round(loss_score, 4),
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
    
    Grading criteria (STRICT):
    - F1 > 0.70 required (buggy code gives ~0.2-0.3)
    - avg_labels > 1.2 required (proper multi-hot predictions)
    - Loss must converge < 0.4
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

    # CRITICAL: Check for multi-label behavior
    # With CrossEntropyLoss, model predicts only 1 label per sample (avg_labels ≈ 1.0)
    # With BCEWithLogitsLoss, model should predict multiple (avg_labels > 1.0)
    
    if avg_labels is not None and avg_labels < 0.8:
        return 0.15, {
            "reason": "too_few_labels_single_label_behavior",
            "avg_labels": avg_labels,
            "expected": ">1.2 for multi-label"
        }
    
    # STRICT: F1 score - PRIMARY metric (55% weight)
    f1_score_val = 0.0
    if f1 is not None:
        if f1 < 0.40:
            # Very low F1 indicates bug not fixed
            return 0.20, {
                "reason": "f1_too_low_likely_unfixed",
                "f1": f1,
                "expected": ">0.65 for fixed code"
            }
        f1_score_val = sigmoid_score(f1, center=0.70, steepness=15.0, higher_is_better=True) * 0.55
    else:
        return 0.15, {"reason": "no_f1_score_parsed"}
    
    # Multi-label check: avg_labels (25% weight)
    labels_score = 0.0
    if avg_labels is not None:
        if avg_labels >= 1.3:
            labels_score = 0.25  # Full score for proper multi-label
        elif avg_labels >= 1.0:
            labels_score = 0.15  # Partial - borderline multi-label
        else:
            labels_score = sigmoid_score(avg_labels, center=1.0, steepness=8.0, higher_is_better=True) * 0.15

    # Loss convergence (20% weight)
    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_score(final_loss, center=0.35, steepness=8.0, higher_is_better=False) * 0.20

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
    1. Unfreeze backbone -> grad_norm > 0
    2. Only pass head params to optimizer -> param_count < 10000
    
    Buggy state: grad_norm = 0, param_count = 530442
    
    Grading criteria (STRICT - binary fix detection):
    - Must demonstrate ONE of the two valid fixes
    - Loss must be reasonable (<3.0 for CrossEntropy on 10 classes)
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

    # Detect fix type FIRST
    fix_score = 0.0
    fix_type = "none"
    
    # Fix 1: Unfreeze backbone (grad_norm > 0)
    if grad_norm is not None and grad_norm > 0.01:
        fix_score = 0.70
        fix_type = "unfrozen"
    # Fix 2: Only head params (param_count should be ~5130 for Linear(512, 10))
    elif param_count is not None and param_count < 15000:
        fix_score = 0.70
        fix_type = "head_only"
    # Buggy state: frozen (grad_norm=0) but full params (>500000)
    elif grad_norm is not None and grad_norm == 0.0:
        if param_count is not None and param_count > 100000:
            return 0.10, {
                "reason": "buggy_state_unchanged",
                "grad_norm": grad_norm,
                "param_count": param_count,
                "hint": "Either unfreeze backbone or only pass head params to optimizer"
            }
    
    if fix_score == 0.0:
        return 0.15, {
            "reason": "could_not_detect_valid_fix",
            "grad_norm": grad_norm,
            "param_count": param_count
        }

    # Loss should be reasonable (30% weight)
    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_score(final_loss, center=2.5, steepness=3.0, higher_is_better=False) * 0.30

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


def grade_task6(result: RunResult) -> tuple[float, dict]:
    """
    Task 6: Input-Output Mismatch (Multiple Bugs)
    
    Bugs to fix:
    1. Shape mismatch: 32x32 images but model expects 28x28
    2. Channel order: HWC format but model expects CHW
    3. Label encoding: One-hot labels but CrossEntropyLoss expects indices
    4. Batch dimension: Single sample missing batch dim in validation
    
    Anti-gaming measures:
    - Must have actual CNN training (convolutions detected in code)
    - Must show learning trajectory (loss decrease)
    - Must have reasonable epoch count (>= 20)
    - Penalize hardcoded metrics or unrealistic outputs
    - Check for actual tensor operations (permute, reshape, etc.)
    """
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code, "task6")
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.05, {"reason": "timed_out"}

    if result.exit_code != 0:
        # Check for specific error types that indicate partial fixes
        stderr_lower = result.stderr.lower()
        if "shape" in stderr_lower or "size" in stderr_lower:
            return 0.10, {"reason": "shape_error_unfixed", "stderr": result.stderr[:500]}
        if "dimension" in stderr_lower or "dim" in stderr_lower:
            return 0.10, {"reason": "dimension_error_unfixed", "stderr": result.stderr[:500]}
        if "expected" in stderr_lower and "got" in stderr_lower:
            return 0.10, {"reason": "type_mismatch_unfixed", "stderr": result.stderr[:500]}
        return 0.0, {"reason": "crash", "stderr": result.stderr[:500]}

    code = result.fixed_code
    
    # ANTI-GAMING: Check for genuine CNN architecture (not replaced with fake output)
    has_conv = "Conv2d" in code or "conv2d" in code
    has_training_loop = "optimizer.step()" in code or "optimizer.step()" in code
    has_model_forward = "model(" in code
    
    if not has_conv:
        return 0.05, {"reason": "gaming_no_convolution", "hint": "Original CNN architecture must be preserved"}
    if not has_training_loop:
        return 0.05, {"reason": "gaming_no_training", "hint": "Must have actual training loop"}
    if not has_model_forward:
        return 0.05, {"reason": "gaming_no_forward", "hint": "Must use model for inference"}

    # Parse metrics
    losses = parse_losses(result.stdout)
    val_acc = parse_scalar(result.stdout, "VAL_ACC")
    final_loss = parse_scalar(result.stdout, "FINAL_LOSS")

    # ANTI-GAMING: Check for hardcoded/faked metrics
    if "print('VAL_ACC:0.9" in code or "print(\"VAL_ACC:0.9" in code:
        return 0.05, {"reason": "gaming_hardcoded_metrics"}
    
    # ANTI-GAMING: Require reasonable number of loss values (epoch count)
    if len(losses) < 15:
        return 0.15, {"reason": "too_few_epochs", "epoch_count": len(losses), "expected": ">=20"}
    
    # ANTI-GAMING: Loss should show learning (not flat or random)
    if len(losses) >= 10:
        first_quarter = sum(losses[:len(losses)//4]) / (len(losses)//4)
        last_quarter = sum(losses[-len(losses)//4:]) / (len(losses)//4)
        
        if first_quarter <= last_quarter:
            return 0.20, {
                "reason": "no_learning_trajectory",
                "first_quarter_loss": round(first_quarter, 4),
                "last_quarter_loss": round(last_quarter, 4),
                "hint": "Loss should decrease during training"
            }

    # ANTI-GAMING: Check for unrealistic perfect scores with no learning
    # High accuracy is OK if there's a valid learning trajectory
    if val_acc is not None and val_acc > 0.99:
        # Only flag if loss didn't converge properly (suggests hardcoded output)
        if final_loss is not None and final_loss > 0.1:
            return 0.25, {"reason": "unrealistic_accuracy_no_convergence", "val_acc": val_acc, "final_loss": final_loss}

    if val_acc is None:
        return 0.15, {"reason": "no_val_acc_parsed"}

    if final_loss is None:
        return 0.15, {"reason": "no_final_loss_parsed"}

    # Check for NaN/Inf in losses
    nan_count = sum(1 for loss in losses if math.isnan(loss) or math.isinf(loss))
    if nan_count > 0:
        return 0.10, {"reason": "nan_in_losses", "nan_count": nan_count}

    # ====== BUG FIX DETECTION ======
    bug_fixes_detected = 0
    fix_details = {}

    # Bug 1: Shape fix - check for resize or architecture change
    has_resize = any(kw in code for kw in ["resize", "interpolate", "F.adaptive", "8 * 8", "8*8"])
    has_28_in_data = "28, 28" in code or "28,28" in code
    if has_resize or has_28_in_data:
        bug_fixes_detected += 1
        fix_details["shape_fix"] = True
    else:
        fix_details["shape_fix"] = False

    # Bug 2: Channel order fix - check for permute/transpose OR data created in CHW format
    has_permute = "permute" in code or "transpose" in code or "contiguous" in code
    has_channel_reorder = ".permute(0, 3, 1, 2)" in code or "permute(0,3,1,2)" in code
    # Alternative fix: data created directly in CHW format (n_samples, 1, H, W)
    has_chw_data = any(pat in code for pat in ["n_samples, 1, 28", "n_samples, 1, 32", "(n_samples, 1,"])
    if has_permute or has_channel_reorder or has_chw_data:
        bug_fixes_detected += 1
        fix_details["channel_fix"] = True
    else:
        fix_details["channel_fix"] = False

    # Bug 3: Label encoding fix - check for argmax or returning indices
    has_label_fix = any(kw in code for kw in [
        "argmax", "class_indices", "torch.arange", 
        "labels.long()", "y.long()", "remove one_hot"
    ])
    # Also check if one_hot is removed from generate_data
    no_onehot = "one_hot" not in code or ("# " in code and "one_hot" in code)
    if has_label_fix or no_onehot:
        bug_fixes_detected += 1
        fix_details["label_fix"] = True
    else:
        fix_details["label_fix"] = False

    # Bug 4: Batch dimension fix - check for unsqueeze on single sample
    has_batch_fix = any(kw in code for kw in ["unsqueeze(0)", "unsqueeze( 0)", "[None,", "[None ,"])
    if has_batch_fix:
        bug_fixes_detected += 1
        fix_details["batch_fix"] = True
    else:
        fix_details["batch_fix"] = False

    # ====== SCORING ======
    # Base score from bug fixes (40% weight - 10% per bug)
    bug_fix_score = 0.10 * bug_fixes_detected
    
    # Accuracy score (35% weight) - strict threshold
    # With 5 classes, random is 20%, buggy is ~20-30%, fixed should be >80%
    if val_acc < 0.50:
        # Below 50% suggests not all bugs fixed
        acc_score = 0.0
        acc_penalty_reason = "accuracy_too_low"
    else:
        acc_score = sigmoid_score(val_acc, center=0.82, steepness=20.0, higher_is_better=True) * 0.35
        acc_penalty_reason = None

    # Loss convergence score (15% weight)
    loss_score = sigmoid_score(final_loss, center=0.40, steepness=8.0, higher_is_better=False) * 0.15

    # Learning trajectory bonus (10% weight)
    trajectory_bonus = 0.0
    if len(losses) >= 10:
        first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
        last_half = sum(losses[-len(losses)//2:]) / (len(losses)//2)
        improvement_ratio = (first_half - last_half) / first_half if first_half > 0 else 0
        if improvement_ratio > 0.5:
            trajectory_bonus = 0.10
        elif improvement_ratio > 0.3:
            trajectory_bonus = 0.05

    final_score = min(1.0, bug_fix_score + acc_score + loss_score + trajectory_bonus)
    
    breakdown = {
        "bug_fix_score": round(bug_fix_score, 4),
        "bugs_fixed": bug_fixes_detected,
        "fix_details": fix_details,
        "acc_score": round(acc_score, 4),
        "loss_score": round(loss_score, 4),
        "trajectory_bonus": round(trajectory_bonus, 4),
        "val_acc": val_acc,
        "final_loss": final_loss,
        "epoch_count": len(losses),
    }
    
    if acc_penalty_reason:
        breakdown["acc_penalty_reason"] = acc_penalty_reason
    
    return final_score, breakdown


MIN_SCORE = 0.1     # Minimum score for any submission
MAX_SCORE = 0.9999  # Avoid exact 1.0


def clamp_score(score: float) -> float:
    """Clamp score to (0, 1) exclusive - hackathon requires strictly between 0 and 1."""
    if score <= 0.0:
        return MIN_SCORE
    if score >= 1.0:
        return MAX_SCORE
    return score


def score_task(task_id: str, result: RunResult) -> tuple[float, dict]:
    graders = {
        "task1": grade_task1,
        "task2": grade_task2,
        "task3": grade_task3,
        "task4": grade_task4,
        "task5": grade_task5,
        "task6": grade_task6,
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task_id: {task_id}")

    score, breakdown = graders[task_id](result)
    # Clamp to (0, 1) exclusive - hackathon validator requires strict bounds
    return round(clamp_score(score), 4), breakdown
