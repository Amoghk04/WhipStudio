TASK_DESCRIPTION = """
This binary regression trainer produces NaN loss around step 15.
Fix the numerical instability so loss stays finite for all 60 steps
and the final loss is below 0.5.
Print losses as: LOSSES:[val1, val2, ...]
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
torch.manual_seed(42)
model = nn.Linear(16, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for step in range(60):
    x = torch.randn(64, 16)
    y = torch.rand(64, 1)
    optimizer.zero_grad()
    pred = torch.sigmoid(model(x))
    # BUG: log(pred) can be -inf when pred rounds to 0.0
    loss = -torch.mean(y * torch.log(pred) + (1 - y) * torch.log(1 - pred))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
print('##METRICS_START##')
print('LOSSES:' + str(losses))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "torch.log(pred) when pred can be 0.0 after sigmoid — use F.binary_cross_entropy or clamp",
]
