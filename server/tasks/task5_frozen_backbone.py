TASK_DESCRIPTION = """
This is a standard transfer learning setup classifying 10 categories.
The developer froze the backbone during testing, but forgot to unfreeze it while still passing its parameters to the optimizer.
This wastes memory and computation as frozen params don't need optimizer state.

Fix the code so EITHER:
1. The backbone actually trains (unfreeze it), OR
2. Only pass trainable parameters to the optimizer

The grader checks:
- BACKBONE_GRAD_NORM: >0 means backbone is training, =0 means properly frozen
- OPTIMIZER_PARAM_COUNT: Should be reduced if only passing head params
"""

BUGGY_CODE = """
import torch
import torch.nn as nn

torch.manual_seed(42)

# Dummy dataset
X = torch.randn(32, 512)
y = torch.randint(0, 10, (32,))

# A simulated pre-trained backbone
backbone = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU()
)

# BUG: backbone is frozen, but passed to optimizer (wastes memory/compute)
backbone.requires_grad_(False)

head = nn.Linear(512, 10)

# BUG: passing frozen backbone params to optimizer
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(head.parameters()), lr=0.001
)
criterion = nn.CrossEntropyLoss()

# Count params in optimizer (for grading)
optimizer_param_count = sum(p.numel() for g in optimizer.param_groups for p in g['params'])

losses = []

# Take one step to check gradients
optimizer.zero_grad()
features = backbone(X)
logits = head(features)

loss = criterion(logits, y)
loss.backward()

# Calculate gradient norm on backbone to see if it's training
backbone_grad_norm = sum(
    p.grad.norm().item() for p in backbone.parameters() if p.grad is not None
)

optimizer.step()
losses.append(loss.item())

print('##METRICS_START##')
print('FINAL_LOSS:' + str(losses[-1]))
print('BACKBONE_GRAD_NORM:' + str(backbone_grad_norm))
print('OPTIMIZER_PARAM_COUNT:' + str(optimizer_param_count))
print('##METRICS_END##')
"""
