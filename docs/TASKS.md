# WhipStudio Task Documentation

Detailed documentation for all 6 debugging tasks in WhipStudio.

---

## Task Overview

| Task | Difficulty | Bug Count | Key Skills Tested |
|------|------------|-----------|-------------------|
| task1 | 🟢 Easy | 2 | Basic PyTorch training loop |
| task2 | 🟡 Medium | 1 | Numerical stability |
| task3 | 🔴 Hard | 2 | Memory management, data handling |
| task4 | 🟡 Medium | 2 | Loss function selection, evaluation |
| task5 | 🟡 Medium | 1 | Transfer learning, parameter freezing |
| task6 | 🔴 Hard | 4 | Data preprocessing, CNN architecture |

---

## Task 1: Broken Training Loop

**Difficulty:** 🟢 Easy  
**Category:** Basic Training Loop

### Description
A 2-class linear classifier training loop has two bugs preventing convergence.

### Bugs
1. **Learning rate too high** (`lr=10.0` instead of `lr=0.01`)
2. **Wrong optimizer order** (`optimizer.step()` before `loss.backward()`)

### Success Criteria
- Final loss < 0.3
- Validation accuracy > 0.85
- Losses should decrease monotonically

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| Both bugs fixed, VAL_ACC > 0.90 | 0.85 - 1.0 |
| One bug fixed, some improvement | 0.40 - 0.70 |
| No bugs fixed, code runs | 0.15 - 0.25 |
| Code crashes | 0.0 - 0.10 |

### Hints
- Check the learning rate value
- Compare the order of `backward()` and `step()` calls
- Standard pattern: `zero_grad()` → forward → loss → `backward()` → `step()`

---

## Task 2: Silent NaN Loss

**Difficulty:** 🟡 Medium  
**Category:** Numerical Stability

### Description
A custom binary classification loss produces NaN values silently, causing training to fail without obvious errors.

### Bugs
1. **Unprotected log(0)** - The loss uses `torch.log(pred)` where `pred` can be 0.0

### Success Criteria
- No NaN values in loss
- Training completes successfully
- Validation accuracy > 0.80

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| NaN fixed, VAL_ACC > 0.85 | 0.85 - 1.0 |
| NaN fixed, lower accuracy | 0.50 - 0.75 |
| Still has NaN | 0.10 - 0.25 |
| Code crashes | 0.0 - 0.10 |

### Hints
- Look for `torch.log()` calls
- Use `torch.clamp(pred, min=1e-7)` before taking log
- Or use `torch.log1p(pred - 1 + 1e-7)` for more stability
- Alternative: use built-in `F.binary_cross_entropy` with `eps` parameter

### Common Mistakes
- Only fixing one of the log calls (there may be multiple)
- Using too small epsilon (1e-15 can still cause issues)
- Removing the loss entirely instead of fixing it

---

## Task 3: OOM + Data Leakage

**Difficulty:** 🔴 Hard  
**Category:** Memory Management, Data Pipeline

### Description
A CNN training script has memory issues and data leakage between train/validation sets.

### Bugs
1. **Graph accumulation** - `total_loss += loss` keeps the computation graph, causing OOM
2. **Data leakage** - Augmentation applied before train/val split, leaking information

### Success Criteria
- No OOM errors
- Training completes all epochs
- Validation accuracy > 0.90 (proper separation)

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| Both bugs fixed, VAL_ACC > 0.92 | 0.85 - 1.0 |
| One bug fixed | 0.40 - 0.65 |
| Bugs not fixed, VAL_ACC < 0.70 | 0.10 - 0.25 |
| OOM error | 0.0 - 0.10 |

### Hints
- Use `total_loss += loss.item()` or `total_loss += loss.detach()` to prevent graph retention
- Split data BEFORE applying augmentation
- Apply augmentation only to training set, not validation

### Common Mistakes
- Only fixing one bug
- Using `.detach()` incorrectly (must call `.item()` for scalar losses)
- Applying different augmentations to train/val but still from same augmented source

---

## Task 4: Wrong Loss Function

**Difficulty:** 🟡 Medium  
**Category:** Loss Function Selection

### Description
A multi-label classification task incorrectly uses CrossEntropyLoss instead of BCEWithLogitsLoss.

### Bugs
1. **Wrong loss function** - CrossEntropyLoss is for single-label, not multi-label
2. **Wrong evaluation** - Predictions should be sigmoid > 0.5, not argmax

### Success Criteria
- F1 score > 0.70
- Correct multi-hot predictions
- Training shows improvement

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| Both bugs fixed, F1 > 0.75 | 0.85 - 1.0 |
| Loss fixed, eval partially correct | 0.50 - 0.75 |
| Wrong loss still used | 0.10 - 0.30 |
| Code crashes | 0.0 - 0.10 |

### Hints
- Multi-label: each sample can have multiple labels (e.g., image tagged with [cat, dog, outdoor])
- Use `nn.BCEWithLogitsLoss()` for multi-label
- For predictions: `(torch.sigmoid(output) > 0.5).float()`
- Don't use softmax/argmax - that's for single-label

### Common Mistakes
- Using BCELoss without sigmoid (need BCEWithLogitsLoss or explicit sigmoid)
- Keeping argmax evaluation (should be threshold-based)
- Incorrect target format (should be float, not long)

---

## Task 5: Frozen Backbone

**Difficulty:** 🟡 Medium  
**Category:** Transfer Learning

### Description
A transfer learning setup freezes the backbone but still passes frozen parameters to the optimizer.

### Bugs
1. **Frozen parameters in optimizer** - Backbone is frozen but all params passed to Adam

### Success Criteria
- Only trainable parameters in optimizer, OR
- Backbone unfrozen and training
- Model shows improvement (loss decreases)

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| Bug properly fixed | 0.85 - 1.0 |
| Partial fix (some improvement) | 0.40 - 0.70 |
| No fix (backbone still frozen + in optimizer) | 0.10 - 0.25 |
| Code crashes | 0.0 - 0.10 |

### Two Valid Solutions

**Solution A: Only optimize trainable params**
```python
# Only pass parameters that require gradients
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)
```

**Solution B: Unfreeze the backbone**
```python
# Remove the freezing, train everything
for param in model.backbone.parameters():
    param.requires_grad = True
```

### Hints
- Check which parameters have `requires_grad=True`
- Filter parameters when creating optimizer
- Or change `requires_grad` to True for backbone

### Common Mistakes
- Unfreezing after optimizer creation (optimizer already captured frozen params)
- Only unfreezing some layers
- Forgetting to also handle bias parameters

---

## Task 6: Input-Output Mismatch

**Difficulty:** 🔴 Hard  
**Category:** Data Preprocessing, CNN Architecture

### Description
A CNN for image classification has multiple bugs related to data format and model architecture mismatches.

### Bugs (4 total)
1. **Image size mismatch** - Data is 32x32, model expects 28x28 (fc layer size wrong)
2. **Channel order** - Data is HWC (Height, Width, Channels), model expects CHW
3. **Label encoding** - Labels are one-hot encoded, CrossEntropyLoss expects class indices
4. **Batch dimension** - Single samples missing batch dimension during validation

### Success Criteria
- All 4 bugs fixed
- Model trains without shape errors
- Validation accuracy > 0.85

### Grading Rubric
| Condition | Score Range |
|-----------|-------------|
| All 4 bugs fixed, VAL_ACC > 0.90 | 0.90 - 1.0 |
| 3 bugs fixed | 0.65 - 0.80 |
| 2 bugs fixed | 0.40 - 0.55 |
| 1 bug fixed | 0.20 - 0.35 |
| No bugs fixed | 0.0 - 0.15 |

### Hints

**Bug 1: Image size**
- Option A: Resize images to 28x28 (`F.interpolate` or change data generation)
- Option B: Adjust fc layer input size (7*7*32 → 8*8*32 for 32x32 input)

**Bug 2: Channel order**
```python
# Convert HWC to CHW
images = images.permute(0, 3, 1, 2)  # [N, H, W, C] → [N, C, H, W]
```

**Bug 3: Label encoding**
```python
# Convert one-hot to class indices
labels = one_hot_labels.argmax(dim=1)
```

**Bug 4: Batch dimension**
```python
# Add batch dimension for single samples
x = x.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
```

### Common Mistakes
- Only fixing 1-2 of the 4 bugs
- Fixing in wrong order (need to be careful with dimensions)
- Using wrong dimension for permute/argmax
- Forgetting to handle validation differently from training

---

## Grading Philosophy

### Continuous Rewards
WhipStudio uses continuous scoring (0.0-1.0) rather than binary pass/fail:
- Partial credit for fixing some bugs
- Partial credit for code that runs but doesn't meet thresholds
- Higher scores require meeting stricter thresholds

### Why Continuous?
1. **Better RL training signal** - Agents can learn from small improvements
2. **Differentiates solutions** - Distinguishes between "almost correct" and "completely wrong"
3. **Encourages incremental progress** - Reward shaping guides learning

### Score Interpretation
| Score Range | Meaning |
|-------------|---------|
| 0.90 - 1.00 | Excellent - All bugs fixed, exceeds targets |
| 0.70 - 0.89 | Good - Main bugs fixed, meets basic targets |
| 0.40 - 0.69 | Partial - Some bugs fixed, shows improvement |
| 0.15 - 0.39 | Minimal - Code runs but bugs remain |
| 0.00 - 0.14 | Failed - Crashes or no meaningful output |

---

## Tips for Agents

1. **Read the task description carefully** - It tells you what metrics to print
2. **Keep seed values** - Don't remove `torch.manual_seed()` calls
3. **Print required metrics** - Must output `LOSSES:[...]` and task-specific metrics
4. **Test with tools first** - Use debugging tools before submitting
5. **Fix all bugs** - Partial fixes get partial credit
6. **Don't over-engineer** - Minimal fixes are better than rewrites
