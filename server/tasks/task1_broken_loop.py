TASK_DESCRIPTION = """
This 2-class linear classifier training loop has bugs preventing convergence.
Fix it so that after 50 epochs the loss is below 0.5 and validation accuracy is above 0.80.
Model: nn.Linear(10, 2), dataset: fixed 2-class (160 train, 40 val samples).
Print losses as: LOSSES:[val1, val2, ...]
Print validation accuracy as: VAL_ACC:X.XX
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

# Generate fixed training and validation datasets with learnable pattern
# y = 1 if first feature > 0, else 0
X_train = torch.randn(160, 10)
y_train = (X_train[:, 0] > 0).long()
X_val = torch.randn(40, 10) 
y_val = (X_val[:, 0] > 0).long()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=10.0)  # BUG 1: lr too high
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(50):
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.step()   # BUG 2: step before backward
        loss.backward()    # BUG 3: backward after step
        losses.append(loss.item())

# Validation
model.eval()
with torch.no_grad():
    val_logits = model(X_val)
    val_preds = val_logits.argmax(dim=1)
    val_acc = (val_preds == y_val).float().mean().item()

print('##METRICS_START##')
print('LOSSES:' + str(losses))
print('VAL_ACC:' + str(round(val_acc, 4)))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "optimizer.step() called before loss.backward()",
    "learning rate 10.0 should be ~0.001",
]
