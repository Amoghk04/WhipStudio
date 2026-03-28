TASK_DESCRIPTION = """
This trainer has TWO independent bugs:
1. A memory leak causing OOM crash before epoch 5 on CPU.
2. Data leakage inflating validation accuracy.
Fix both. After 20 epochs: val_acc > 0.70, no OOM, no suspicious early accuracy spike.
Print as: VAL_ACCS:[v1,v2,...] and FINAL_LOSS:X.XX
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(42)
X = torch.randn(1000, 20)
y = (X[:, 0] > 0).float()
# BUG 1: augmentation before split — val set gets augmented
X = X + torch.randn_like(X) * 0.1
train_ds, val_ds = random_split(TensorDataset(X, y), [800, 200])
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
train_losses, val_accs = [], []
total_loss = torch.tensor(0.0)  # BUG 2: keeps computation graph alive
for epoch in range(20):
    model.train()
    for xb, yb in DataLoader(train_ds, batch_size=32):
        optimizer.zero_grad()
        out = model(xb).squeeze()
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss  # BUG 2: graph accumulates
    model.eval()
    with torch.no_grad():
        idx = val_ds.indices
        xv, yv = X[idx], y[idx]
        preds = (torch.sigmoid(model(xv)) > 0.5).float()
        acc = (preds == yv).float().mean().item()
    val_accs.append(round(acc, 4))
print('##METRICS_START##')
print('VAL_ACCS:' + str(val_accs))
print('FINAL_LOSS:' + str(total_loss.item()))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "Augmentation applied before split — move after split, apply to train only",
    "total_loss += loss retains graph — use total_loss += loss.item()",
]
