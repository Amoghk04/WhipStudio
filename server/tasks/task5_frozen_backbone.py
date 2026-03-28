TASK_DESCRIPTION = """
This is a standard transfer learning setup classifying 10 categories.
The developer froze the backbone during testing, but forgot to unfreeze it while still passing its parameters to the optimizer.
Fix the code so the backbone actually trains, or only pass the head parameters.
The grader checks the gradient norm of the backbone from the first backward pass.
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

# BUG: backbone is frozen, but passed to optimizer
backbone.requires_grad_(False)

head = nn.Linear(512, 10)

# passing both backbone and head to optimizer even though backbone is frozen
optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(head.parameters()), lr=0.001
)
criterion = nn.CrossEntropyLoss()

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

# Note: if backbone is properly frozen and only head is passed, backbone_grad_norm will be 0 but optimizer won't complain.
# If backbone is unfrozen, backbone_grad_norm will be > 0.
# The grader handles both valid solutions.
print('##METRICS_START##')
print('FINAL_LOSS:' + str(losses[-1]))
print('BACKBONE_GRAD_NORM:' + str(backbone_grad_norm))
print('##METRICS_END##')
"""
