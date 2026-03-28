TASK_DESCRIPTION = """
This is a multi-label classification problem where each sample can have multiple active classes.
However, the model is currently using `CrossEntropyLoss`, which is meant for single-label (mutually exclusive) classes.
Because of this, the loss trains but the predictions are essentially garbage (treating it as a single-label problem).

Fix the loss function so it correctly handles multi-label classification.
The grader will check:
1. Loss convergence (loss < 0.5)
2. Model predictions are actually multi-hot (avg > 1 label/sample)
3. F1 Score > 0.6
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

torch.manual_seed(42)

# Generate synthetic multi-label data (100 samples, 20 features, 5 classes)
X = torch.randn(100, 20)
# Each sample has a 30% chance of having each class active
y = (torch.rand(100, 5) > 0.7).float() 

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 5)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# BUG: CrossEntropyLoss is for single-label classification
criterion = nn.CrossEntropyLoss()

losses = []
for step in range(100):
    optimizer.zero_grad()
    logits = model(X)
    
    # CrossEntropyLoss expects class indices, not one-hot/multi-hot vectors for the target
    loss = criterion(logits, y)
    
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Evaluation
with torch.no_grad():
    logits = model(X)
    # Using sigmoid and 0.5 threshold for multi-label prediction
    preds = (torch.sigmoid(logits) > 0.5).float()
    
    avg_labels = preds.sum(dim=1).mean().item()
    f1 = f1_score(y.numpy(), preds.numpy(), average='micro')

print('##METRICS_START##')
print('FINAL_LOSS:' + str(losses[-1]))
print('AVG_LABELS:' + str(avg_labels))
print('F1_SCORE:' + str(f1))
print('##METRICS_END##')
"""
