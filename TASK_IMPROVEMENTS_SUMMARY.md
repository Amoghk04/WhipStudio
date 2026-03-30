# Task Improvements Summary

## Overview

This document summarizes the improvements made to the WhipStudio ML debugging environment tasks and graders.

## Key Issues Fixed

### 1. Unstable Datasets (Tasks 1 & 2)
**Problem**: Tasks were generating random data inside training loops, making loss values non-deterministic and graders unreliable.

**Solution**: 
- Fixed datasets with deterministic seeds (`torch.manual_seed()`)
- Clear train/validation splits
- Learnable patterns (e.g., `y = (X[:, 0] > 0).long()`)

### 2. Gameable Graders
**Problem**: High learning rates (e.g., lr=1000) could get full scores by producing low loss values despite unstable training.

**Solution**:
- Added **loss spike detection** in Task 1 grader
- If `max_loss > initial_loss * 5.0` or `max_loss > 10.0`, submission is penalized
- Partial fixes with bad LR get capped at 0.2 score

### 3. Inverted Scoring Logic
**Problem**: The `sigmoid_reward()` function had confusing `invert` parameter that caused inverted scoring (low F1 → high score).

**Solution**:
- Created new `sigmoid_score(value, center, steepness, higher_is_better)` function
- Clear semantics: `higher_is_better=True` rewards values above center

### 4. Task-Specific Validation
**Problem**: Generic validation rejected valid submissions (e.g., Task 5 required loops but single forward pass was valid).

**Solution**:
- `is_valid_submission(code, stdout, exit_code, task_id)` now takes task_id
- Task-specific validation rules

## Task Details

### Task 1: Broken Training Loop
- **Bugs**: `lr=10.0`, `step()` before `backward()`
- **Buggy score**: ~0.003
- **Fixed score**: ~0.74
- **Spike detection**: Penalizes unstable training (score capped at 0.2)

### Task 2: NaN Loss
- **Bug**: `torch.log(pred)` when pred can be 0.0
- **Fix**: Increased buggy LR to 0.5 to actually trigger NaN
- **Buggy score**: ~0.16 (has NaN values)
- **Fixed score**: ~0.83

### Task 3: Label Inversion
- **Bug**: `criterion(out, 1 - yb)` inverts labels
- **Buggy score**: ~0.34 (accuracy ~5%)
- **Fixed score**: ~0.80 (accuracy ~95%)

### Task 4: Wrong Loss (Multi-label)
- **Bug**: Using `CrossEntropyLoss` instead of `BCEWithLogitsLoss`
- **Buggy score**: ~0.74 (F1 ~0.73)
- **Fixed score**: ~0.97 (F1 = 1.0)

### Task 5: Frozen Backbone
- **Bug**: Backbone frozen but still passed to optimizer
- **Two valid fixes**:
  1. Unfreeze backbone (grad_norm > 0)
  2. Only pass head params (param_count < 100k)
- **Added**: `OPTIMIZER_PARAM_COUNT` metric for grading
- **Buggy score**: ~0.18
- **Fixed score**: ~0.88

## Grading Structure

All graders follow a consistent pattern:
```python
# Primary metric (50-60% weight)
primary_score = sigmoid_score(metric, center, steepness, higher_is_better) * weight

# Secondary metrics (30% weight)
secondary_score = ...

# Bonus conditions (10-20%)
bonus = ...

final_score = min(1.0, primary_score + secondary_score + bonus)
```

## Testing Results

| Task | Buggy Score | Fixed Score | Discrimination |
|------|-------------|-------------|----------------|
| 1    | 0.003       | 0.739       | ✅ Excellent   |
| 2    | 0.157       | 0.827       | ✅ Excellent   |
| 3    | 0.344       | 0.804       | ✅ Excellent   |
| 4    | 0.735       | 0.966       | ✅ Good        |
| 5    | 0.179       | 0.879       | ✅ Excellent   |

## Files Modified

- `server/tasks/task1_broken_loop.py` - Fixed dataset, learnable pattern
- `server/tasks/task2_nan_loss.py` - Increased LR to trigger NaN bug
- `server/tasks/task3_oom_leakage.py` - Redesigned with label inversion bug
- `server/tasks/task5_frozen_backbone.py` - Added OPTIMIZER_PARAM_COUNT metric
- `server/tasks/graders.py` - Complete rewrite with proper scoring logic
