TASK_DESCRIPTION = """
This binary classification trainer produces NaN loss after a few epochs.
Fix the numerical instability so loss stays finite for all 60 epochs
and the final loss is below 0.4 with validation accuracy above 0.75.
Print losses as: LOSSES:[val1, val2, ...]
Print validation accuracy as: VAL_ACC:X.XX
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

# Generate fixed training and validation datasets with learnable pattern
# y = 1 if sum of first 3 features > 0, else 0
X_train = torch.randn(320, 16)
y_train = (X_train[:, :3].sum(dim=1, keepdim=True) > 0).float()
X_val = torch.randn(80, 16)
y_val = (X_val[:, :3].sum(dim=1, keepdim=True) > 0).float()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = nn.Linear(16, 1)
# BUG AMPLIFIER: Higher learning rate makes predictions more extreme, causing log(0)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

losses = []
for epoch in range(60):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = torch.sigmoid(model(x))
        # BUG: log(pred) can be -inf when pred rounds to 0.0 due to extreme weights
        # This happens because SGD with high LR pushes weights to extreme values
        loss = -torch.mean(y * torch.log(pred) + (1 - y) * torch.log(1 - pred))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Validation
model.eval()
with torch.no_grad():
    val_pred = torch.sigmoid(model(X_val))
    val_binary = (val_pred > 0.5).float()
    val_acc = (val_binary == y_val).float().mean().item()

print('##METRICS_START##')
print('LOSSES:' + str(losses))
print('VAL_ACC:' + str(round(val_acc, 4)))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "torch.log(pred) when pred can be 0.0 after sigmoid — use F.binary_cross_entropy or clamp",
    "High learning rate (0.5) causes extreme predictions",
]
