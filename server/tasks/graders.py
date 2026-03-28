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
    match = re.search(rf"{key}:([-\d.]+)", stdout)
    return float(match.group(1)) if match else None


def is_valid_submission(code: str, stdout: str, exit_code: int) -> tuple[bool, str]:
    if exit_code == 0:
        if "LOSSES:" not in stdout and "FINAL_LOSS:" not in stdout:
            return False, "No valid metrics output detected"
        if "LOSSES:" in stdout:
            losses = parse_losses(stdout)
            if len(losses) < 5:
                return False, "Fewer than 5 loss values parsed"
    try:
        tree = ast.parse(code)
        if not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree)):
            return False, "No ast.For or ast.While node found"
    except Exception:
        pass
    return True, ""


def sigmoid_reward(value: float, center: float, steepness: float, invert: bool = False) -> float:
    try:
        if invert:
            x = steepness * (value - center)
        else:
            x = steepness * (center - value)
        return round(1.0 / (1.0 + math.exp(-x)), 4)
    except OverflowError:
        return 0.0 if (invert and value > center) or (not invert and value < center) else 1.0


def grade_task1(result: RunResult) -> tuple[float, dict]:
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code)
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.05, {"reason": "timed_out"}
    if result.exit_code != 0:
        return 0.0, {"reason": "crash"}

    losses = parse_losses(result.stdout)
    if not losses:
        return 0.1, {"reason": "no_losses_parsed"}
    if any(math.isnan(loss) or math.isinf(loss) for loss in losses):
        return 0.15, {"reason": "nan_inf_found"}

    final = losses[-1]
    base_score = sigmoid_reward(final, center=0.75, steepness=3.0, invert=True)

    bonus = 0.0
    half = len(losses) // 2
    if half > 0:
        first_half = sum(losses[:half]) / half
        second_half = sum(losses[half:]) / len(losses[half:])
        if second_half < 0.85 * first_half:
            bonus = 0.1
            
    final_score = min(1.0, base_score + bonus)
    breakdown = {"base_score": base_score, "monotonicity_bonus": bonus}
    return final_score, breakdown


def grade_task2(result: RunResult) -> tuple[float, dict]:
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code)
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.05, {"reason": "timed_out"}
    if result.exit_code != 0:
        return 0.0, {"reason": "crash"}

    losses = parse_losses(result.stdout)
    if not losses or len(losses) < 30:
        return 0.1, {"reason": "too_few_losses"}

    nan_count = sum(1 for loss in losses if math.isnan(loss) or math.isinf(loss))
    if nan_count == len(losses):
        return 0.0, {"reason": "all_nans"}
        
    nan_ratio = nan_count / len(losses)
    finite_losses = [loss for loss in losses if not math.isnan(loss) and not math.isinf(loss)]
    final_finite_loss = finite_losses[-1] if finite_losses else float('inf')

    convergence_score = sigmoid_reward(final_finite_loss, center=0.5, steepness=4.0, invert=True)
    convergence_score *= (1.0 - nan_ratio)

    stability_bonus = 0.0
    if len(finite_losses) >= 20:
        tail = finite_losses[-20:]
        mean_tail = sum(tail) / len(tail)
        tail_variance = sum((x - mean_tail) ** 2 for x in tail) / len(tail)
        stability_bonus = sigmoid_reward(tail_variance, center=0.01, steepness=200.0, invert=True) * 0.1

    final_score = min(1.0, convergence_score + stability_bonus)
    breakdown = {"convergence_score": convergence_score, "nan_penalty": (1.0 - nan_ratio), "stability_bonus": stability_bonus, "nan_ratio": nan_ratio}
    return final_score, breakdown


def grade_task3(result: RunResult) -> tuple[float, dict]:
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code)
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        if "out of memory" in result.stderr.lower():
            return 0.1, {"reason": "oom"}
        return 0.0, {"reason": "crash"}

    val_accs = parse_val_accs(result.stdout)
    final_loss_val = parse_scalar(result.stdout, "FINAL_LOSS")

    memory_score = 0.0
    if final_loss_val is not None:
        memory_score = sigmoid_reward(final_loss_val, center=50.0, steepness=0.05, invert=True) * 0.5

    leakage_score = 0.0
    early_acc = 0.0
    final_acc = 0.0
    if val_accs and len(val_accs) >= 2:
        early_acc = sum(val_accs[:2]) / 2.0
        final_acc = val_accs[-1]
        
        leak_p1 = sigmoid_reward(early_acc, center=0.75, steepness=20.0, invert=True) * 0.3
        leak_p2 = sigmoid_reward(final_acc, center=0.68, steepness=15.0, invert=False) * 0.7
        leakage_score = (leak_p1 + leak_p2) * 0.5

    final_score = min(1.0, memory_score + leakage_score)
    breakdown = {"memory_score": memory_score, "leakage_score": leakage_score, "early_acc": early_acc, "final_acc": final_acc}
    return final_score, breakdown


def grade_task4(result: RunResult) -> tuple[float, dict]:
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code)
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        return 0.0, {"reason": "crash"}

    final_loss = parse_scalar(result.stdout, "FINAL_LOSS")
    avg_labels = parse_scalar(result.stdout, "AVG_LABELS")
    f1 = parse_scalar(result.stdout, "F1_SCORE")

    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_reward(final_loss, center=0.5, steepness=4.0, invert=True) * 0.3

    labels_score = 0.0
    if avg_labels is not None:
        labels_score = sigmoid_reward(avg_labels, center=1.0, steepness=5.0, invert=False) * 0.3

    f1_s = 0.0
    if f1 is not None:
        f1_s = sigmoid_reward(f1, center=0.6, steepness=10.0, invert=False) * 0.4

    final_score = min(1.0, loss_score + labels_score + f1_s)
    breakdown = {"loss_score": loss_score, "labels_score": labels_score, "f1_score": f1_s}
    return final_score, breakdown


def grade_task5(result: RunResult) -> tuple[float, dict]:
    valid, reason = is_valid_submission(result.fixed_code, result.stdout, result.exit_code)
    if not valid:
        return 0.0, {"reason": reason}

    if result.timed_out:
        return 0.1, {"reason": "timed_out"}

    if result.exit_code != 0:
        return 0.0, {"reason": "crash"}

    final_loss = parse_scalar(result.stdout, "FINAL_LOSS")
    grad_norm = parse_scalar(result.stdout, "BACKBONE_GRAD_NORM")

    loss_score = 0.0
    if final_loss is not None:
        loss_score = sigmoid_reward(final_loss, center=2.2, steepness=3.0, invert=True) * 0.5
        
    grad_score = 0.0
    if grad_norm is not None:
        grad_score = sigmoid_reward(grad_norm, center=0.001, steepness=1000.0, invert=False) * 0.5

    final_score = min(1.0, loss_score + grad_score)
    breakdown = {"loss_score": loss_score, "grad_score": grad_score}
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
