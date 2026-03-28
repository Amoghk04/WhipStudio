TASK_DESCRIPTION = """
This 2-class linear classifier training loop has bugs preventing convergence.
Fix it so that after 50 steps the loss is below 0.75 and decreasing.
Model: nn.Linear(10, 2), dataset: random 2-class, 32 samples/batch.
Print losses as: LOSSES:[val1, val2, ...]
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
torch.manual_seed(0)
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=10.0)  # BUG 1: lr too high
criterion = nn.CrossEntropyLoss()
losses = []
for step in range(50):
    x = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    optimizer.step()   # BUG 2: step before backward
    loss.backward()    # BUG 3: backward after step
    losses.append(loss.item())
print('##METRICS_START##')
print('LOSSES:' + str(losses))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "optimizer.step() called before loss.backward()",
    "learning rate 10.0 should be ~0.001",
]
