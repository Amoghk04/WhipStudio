TASK_DESCRIPTION = """
This binary classification trainer has a bug causing validation accuracy around 50%.
The bug inverts the labels during training. Fix it so after 20 epochs:
- VAL_ACC > 0.90 (the primary goal)
- FINAL_LOSS < 0.3

Print as: VAL_ACCS:[v1,v2,...] and FINAL_LOSS:X.XX
Wrap output in ##METRICS_START## and ##METRICS_END##
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
X_train = torch.randn(800, 20)
y_train = (X_train[:, 0] > 0).float()
X_val = torch.randn(200, 20)
y_val = (X_val[:, 0] > 0).float()

train_ds = TensorDataset(X_train, y_train)
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()
val_accs = []
losses = []
for epoch in range(20):
    model.train()
    for xb, yb in DataLoader(train_ds, batch_size=32, shuffle=True):
        optimizer.zero_grad()
        out = model(xb).squeeze()
        # BUG: Wrong label transformation - should use yb directly
        loss = criterion(out, 1 - yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_val).squeeze()) > 0.5).float()
        acc = (preds == y_val).float().mean().item()
    val_accs.append(round(acc, 4))
print('##METRICS_START##')
print('VAL_ACCS:' + str(val_accs))
print('FINAL_LOSS:' + str(sum(losses[-25:])/25))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "Label inversion: criterion(out, 1 - yb) inverts the labels — use criterion(out, yb)",
]
