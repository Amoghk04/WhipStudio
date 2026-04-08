# WhipStudio Debugging Tools Guide

This guide explains how to use WhipStudio's debugging tools effectively.

## Overview

WhipStudio provides 6 tools for iterative debugging:

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `execute_snippet` | Run quick code tests | Verify imports, check versions, test small fixes |
| `inspect_tensor` | Examine tensor properties | Debug shape mismatches, gradient issues, NaN/Inf |
| `run_training_probe` | Test training loop | Verify loss decreases, check gradient flow |
| `get_variable_state` | Inspect multiple values | Check model state, optimizer config, data properties |
| `inspect_diff` | Preview your changes | Review before submission, catch mistakes |
| `submit_fix` | Submit final solution | When confident in your fix |

## Tool Usage Workflow

### Recommended Debugging Strategy

```
1. Analyze buggy code (read carefully)
       ↓
2. Form hypothesis about bug(s)
       ↓
3. Use tools to verify hypothesis
   ├── execute_snippet: Test specific behavior
   ├── inspect_tensor: Check shapes/gradients
   └── get_variable_state: Check configuration
       ↓
4. Develop fix based on findings
       ↓
5. run_training_probe: Test if fix works
       ↓
6. inspect_diff: Review your changes
       ↓
7. submit_fix: Submit when confident
```

---

## Tool Details

### 1. execute_snippet

Run a short Python code snippet to test specific behaviors.

**Best for:**
- Testing if specific code runs without error
- Checking library versions and availability
- Verifying small code transformations
- Quick experiments

**Example:**
```python
action = {
    "action_type": "execute_snippet",
    "code": """
import torch
import torch.nn as nn

# Test if softmax + log is the issue
pred = torch.tensor([0.0, 1.0])
print("log(0):", torch.log(pred[0]))  # Should be -inf
print("log(1):", torch.log(pred[1]))  # Should be 0

# Test fix: clamp before log
pred_safe = pred.clamp(min=1e-7)
print("log(clamped 0):", torch.log(pred_safe[0]))
"""
}
```

**Returns:**
- `stdout`: Printed output
- `stderr`: Error messages
- `exit_code`: 0 for success, non-zero for errors
- `timed_out`: True if execution exceeded 30 seconds

---

### 2. inspect_tensor

Examine a tensor's properties in detail.

**Best for:**
- Debugging shape mismatches ("Expected [N, 10] got [N, 10, 1]")
- Checking gradient flow (is grad None? is requires_grad set?)
- Finding NaN/Inf values in tensors
- Verifying data types

**Example:**
```python
action = {
    "action_type": "inspect_tensor",
    "setup_code": """
import torch
import torch.nn as nn

# Simulate the training setup
model = nn.Linear(10, 2)
x = torch.randn(32, 10)
y = model(x)
loss = y.sum()
loss.backward()
""",
    "target_expression": "model.weight.grad"
}
```

**Returns:**
- `shape`: List of dimensions, e.g., `[2, 10]`
- `dtype`: Data type, e.g., `"torch.float32"`
- `requires_grad`: Whether gradients are tracked
- `grad_is_none`: True if `.grad` is None (no backward pass)
- `min_val`, `max_val`, `mean_val`: Statistics
- `is_nan`, `is_inf`: True if any NaN/Inf values found

**Pro Tips:**
- Check `grad_is_none: true` → backward() wasn't called or requires_grad=False
- Check `is_nan: true` → numerical instability (log(0), div by 0, etc.)
- Check shape mismatches between layers

---

### 3. run_training_probe

Run a few training steps to observe the loss curve and gradients.

**Best for:**
- Verifying that loss decreases (training works)
- Checking if gradients flow to all layers
- Testing a potential fix before submission
- Detecting exploding/vanishing gradients

**Example:**
```python
action = {
    "action_type": "run_training_probe",
    "code": """
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

losses = []
for epoch in range(10):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(f"LOSSES:{losses}")
""",
    "steps": 5  # Will capture first 5 steps
}
```

**Returns:**
- `losses`: List of loss values per step
- `grad_norms`: Dict of layer name → gradient norm
- `optimizer_param_count`: Number of parameters in optimizer
- `final_loss`: Last loss value
- `loss_is_nan`, `loss_is_inf`: True if loss became NaN/Inf
- `timed_out`: True if exceeded timeout

**Pro Tips:**
- If `losses` are flat or increasing → fix not working
- If `loss_is_nan` → numerical instability remains
- If `grad_norms` has zeros → frozen layers or detached tensors
- Compare grad_norms between layers to find problems

---

### 4. get_variable_state

Evaluate multiple expressions and see their values.

**Best for:**
- Checking model configuration (training mode, layer count)
- Inspecting optimizer settings (learning rate, param groups)
- Verifying data shapes and types
- Debugging complex state

**Example:**
```python
action = {
    "action_type": "get_variable_state",
    "setup_code": """
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
model[0].requires_grad_(False)  # Freeze first layer

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
""",
    "expressions": [
        "model.training",
        "model[0].weight.requires_grad",
        "model[2].weight.requires_grad", 
        "optimizer.param_groups[0]['lr']",
        "len(list(model.parameters()))",
        "sum(p.numel() for p in model.parameters() if p.requires_grad)"
    ]
}
```

**Returns:**
- `results`: Dict mapping expression → result info
  - `repr`: String representation
  - `type`: Python type name
  - `value`: Actual value (for scalars)
  - `shape`: Shape (for tensors/arrays)
  - `error`: Error message if evaluation failed

**Pro Tips:**
- Check `model.training` → should be True during training
- Check `requires_grad` on layers you expect to train
- Verify `lr` is reasonable (not 10.0, not 1e-10)
- Count trainable params vs total params

---

### 5. inspect_diff

Compare your proposed fix against the original buggy code.

**Best for:**
- Reviewing your changes before submission
- Catching unintended modifications
- Verifying you fixed all identified bugs
- Counting lines changed

**Example:**
```python
action = {
    "action_type": "inspect_diff",
    "proposed_code": """
import torch
import torch.nn as nn

# Fixed: Changed lr from 10.0 to 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Fixed: Correct order - backward before step
loss.backward()
optimizer.step()
"""
}
```

**Returns:**
- `diff`: Unified diff format (like `git diff`)
- `lines_changed`: Total lines modified
- `additions`: Lines added (prefixed with +)
- `deletions`: Lines removed (prefixed with -)

**Pro Tips:**
- Review diff for unintended changes (typos, removed seed)
- Verify all bug fixes are visible in diff
- Keep changes minimal - don't refactor unrelated code

---

### 6. submit_fix

Submit your final solution for grading.

**This is a terminal action** - after calling this, the episode ends.

**Example:**
```python
action = {
    "action_type": "submit_fix",
    "fixed_code": """
import torch
import torch.nn as nn

torch.manual_seed(42)

# Complete fixed training script...
# Must print LOSSES:[v1, v2, ...]
# For some tasks: VAL_ACC:X.XX
""",
    "explanation": "Fixed two bugs: 1) Changed lr from 10.0 to 0.01, 2) Moved step() after backward()"
}
```

**Returns:**
- `reward`: Score from 0.0 to 1.0
- `episode_done`: Always True
- `error_log`: stdout/stderr from execution
- `grader_details`: Task-specific grading info

---

## Common Debugging Patterns

### Pattern 1: Shape Mismatch Debugging

```python
# Step 1: Check input shapes
action1 = {
    "action_type": "get_variable_state",
    "setup_code": buggy_code,
    "expressions": ["X.shape", "y.shape", "model(X[:1]).shape"]
}

# Step 2: Inspect specific layer
action2 = {
    "action_type": "inspect_tensor",
    "setup_code": buggy_code,
    "target_expression": "model.fc.weight"
}
```

### Pattern 2: Gradient Flow Debugging

```python
# Step 1: Check if gradients exist
action1 = {
    "action_type": "run_training_probe",
    "code": buggy_code,
    "steps": 3
}
# Look at grad_norms - any zeros?

# Step 2: Check specific layer
action2 = {
    "action_type": "inspect_tensor",
    "setup_code": buggy_code + "\nloss.backward()",
    "target_expression": "backbone[0].weight.grad"
}
```

### Pattern 3: NaN Loss Debugging

```python
# Step 1: Find where NaN appears
action1 = {
    "action_type": "execute_snippet",
    "code": """
import torch
pred = torch.tensor([0.0, 0.5, 1.0])
print("log(pred):", torch.log(pred))
print("Any NaN?:", torch.isnan(torch.log(pred)).any())
"""
}

# Step 2: Test fix
action2 = {
    "action_type": "execute_snippet", 
    "code": """
import torch
pred = torch.tensor([0.0, 0.5, 1.0])
pred_safe = pred.clamp(min=1e-7)
print("log(pred_safe):", torch.log(pred_safe))
print("Any NaN?:", torch.isnan(torch.log(pred_safe)).any())
"""
}
```

### Pattern 4: Loss Function Debugging

```python
# Check what loss function expects vs what model outputs
action = {
    "action_type": "get_variable_state",
    "setup_code": buggy_code,
    "expressions": [
        "criterion",  # What loss is being used
        "out.shape",  # Model output shape
        "y.shape",    # Label shape
        "y.dtype",    # Label type (long vs float)
        "y[:3]"       # Sample labels
    ]
}
```

---

## Tips for Efficient Tool Use

1. **Start broad, then narrow**: Use `get_variable_state` first to understand the code, then `inspect_tensor` for specific issues.

2. **Limit turns**: You have max 10 turns per episode. Plan your debugging strategy.

3. **Test fixes early**: Use `run_training_probe` with `steps=2-3` to quickly verify if a fix works.

4. **Always inspect_diff**: Before `submit_fix`, always review your changes.

5. **Read error messages**: Tool outputs include stderr - read it carefully.

6. **Keep setup_code minimal**: Don't include the entire script - just what's needed to evaluate the expression.

7. **Use multiple expressions**: `get_variable_state` can evaluate up to 10 expressions at once - use it!

---

## Security Restrictions

Tools run in a sandboxed environment with these restrictions:

**Allowed imports:**
- torch, torch.nn, torch.optim, torch.utils.data
- numpy, sklearn, pandas, matplotlib, scipy
- math, random, os (read-only), sys
- collections, itertools, functools
- json, re, typing, copy, dataclasses

**Blocked imports:**
- socket, requests, httpx, urllib (no network)
- subprocess, shutil (no shell access)

**Other restrictions:**
- 30 second timeout per tool call
- File writes only to /tmp
- No GPU access (CPU only)
