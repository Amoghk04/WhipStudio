# Task & Grader Analysis Report

## 🔴 CRITICAL FINDING: Why Scores Were Identical

The score `0.7390` appearing for all tasks suggested the LLM was generating code that:
1. **Ran successfully** (exit_code = 0)
2. **Output valid metrics** (LOSSES, VAL_ACC, etc.)
3. **BUT didn't necessarily fix the actual bugs**

The old graders gave "partial credit" for any running code, leading to similar scores.

## ✅ FIXES APPLIED

### 1. Stricter Sigmoid Centers
- Old centers were too forgiving (e.g., val_acc center at 0.85)
- New centers require better performance (e.g., val_acc center at 0.90-0.92)
- Increased steepness for sharper differentiation (15→25-30)

### 2. Early Rejection for Unfixed Bugs
- Added explicit checks for "likely unfixed" states
- Task 3: Reject if val_acc < 0.65 (buggy code gives ~0.50)
- Task 4: Reject if f1 < 0.40 (buggy code gives ~0.25)
- Task 5: Reject if buggy state unchanged

### 3. Task 3 Mismatch Fixed
- **Old**: Description said "OOM and data leakage" 
- **Actual bug**: Label inversion (`criterion(out, 1 - yb)`)
- **Fixed**: Updated grader to match actual bug

### 4. Reduced Base Scores
- Old task 2 gave 0.40 "free" for avoiding NaN
- New gives 0.35 base, with stricter accuracy requirements

## Updated Grader Summary

| Task | Bug | Key Metric | Threshold | Weight |
|------|-----|------------|-----------|--------|
| task1 | LR + step/backward | VAL_ACC | >0.90 | 60% |
| task2 | NaN loss | No NaN + VAL_ACC | >0.80 | 40% |
| task3 | Label inversion | VAL_ACC | >0.92 | 60% |
| task4 | Wrong loss | F1_SCORE | >0.70 | 55% |
| task5 | Frozen backbone | Fix detection | Binary | 70% |

## Expected Score Distribution After Fix

**Well-fixed code** (correct fix): 0.85-1.00
**Partially fixed** (runs but suboptimal): 0.40-0.70
**Unfixed** (bug still present): 0.10-0.25
**Broken** (crashes): 0.00-0.10

This creates better separation between models of different capability.

## Files Modified

1. `server/tasks/graders.py` - All 5 graders updated
2. `server/tasks/task3_oom_leakage.py` - Description clarified
