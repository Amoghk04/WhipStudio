TASK_DESCRIPTION = """
This image classification script has multiple input-output mismatch bugs that cause 
silent failures or crashes. The model is a simple CNN trained on synthetic "images".

There are 4 BUGS to fix:
1. Shape mismatch: The model expects 28x28 images but data generator creates 32x32
2. Channel order mismatch: Model expects CHW but data is HWC format  
3. Label encoding mismatch: Model expects class indices but labels are one-hot encoded
4. Batch dimension mismatch: A validation step processes unbatched data

Fix all bugs so that:
- Training runs without errors for 30 epochs
- VAL_ACC > 0.85
- FINAL_LOSS < 0.5

Print as: LOSSES:[l1,l2,...], VAL_ACC:X.XX, FINAL_LOSS:X.XX
Wrap output in ##METRICS_START## and ##METRICS_END##
"""

BUGGY_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 30

# BUG 1: Create 32x32 images but model expects 28x28
# Generate synthetic image data (HWC format - common from PIL/OpenCV)
def generate_data(n_samples):
    # Creates images in HWC format (Height x Width x Channels)
    images = torch.randn(n_samples, 32, 32, 1)  # BUG: Wrong size & HWC format
    # Each class has a distinct pattern based on mean pixel value region
    class_indices = torch.randint(0, NUM_CLASSES, (n_samples,))
    for i, c in enumerate(class_indices):
        images[i] += c * 0.5  # Add class-dependent offset
    
    # BUG 3: Return one-hot labels instead of class indices
    labels = F.one_hot(class_indices, NUM_CLASSES).float()
    return images, labels

X_train, y_train = generate_data(800)
X_val, y_val = generate_data(200)

# BUG 2: Model expects CHW format (Channels x Height x Width) and 28x28 images
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Expecting input: (batch, 1, 28, 28) in CHW format
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # After two pooling ops on 28x28: 28->14->7, so 7*7*32 = 1568
        self.fc = nn.Linear(7 * 7 * 32, NUM_CLASSES)
    
    def forward(self, x):
        # Expects x to be (batch, channels, height, width)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Expects class indices, not one-hot

train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        
        # Missing: permute from HWC to CHW format
        # Missing: resize from 32x32 to 28x28
        outputs = model(images)
        
        # BUG: criterion expects class indices but labels are one-hot
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(train_loader))

# Validation
model.eval()
with torch.no_grad():
    # BUG 4: Process single sample without batch dimension
    sample = X_val[0]  # Shape: (32, 32, 1) - missing batch dim
    single_pred = model(sample)  # Will crash: expects (batch, C, H, W)
    
    # Full validation (also has format issues)
    val_outputs = model(X_val)
    val_preds = val_outputs.argmax(dim=1)
    val_labels = y_val.argmax(dim=1)  # Convert one-hot back to indices for comparison
    val_acc = (val_preds == val_labels).float().mean().item()

print('##METRICS_START##')
print('LOSSES:' + str([round(l, 4) for l in losses]))
print('VAL_ACC:' + str(round(val_acc, 4)))
print('FINAL_LOSS:' + str(round(losses[-1], 4)))
print('##METRICS_END##')
"""

GROUND_TRUTH_BUGS = [
    "Shape mismatch: Images are 32x32 but model expects 28x28 - need to resize or fix model architecture",
    "Channel order mismatch: Data is in HWC format but model expects CHW - use .permute(0, 3, 1, 2)",
    "Label encoding mismatch: Labels are one-hot but CrossEntropyLoss expects class indices - use .argmax(dim=1) or change generate_data",
    "Batch dimension mismatch: single sample missing batch dimension - use sample.unsqueeze(0)",
]

# Expected fixes (for grader reference):
# 1. Either resize images to 28x28 OR change model to expect 32x32 (fc layer: 8*8*32)
# 2. Add: images = images.permute(0, 3, 1, 2)  # HWC -> CHW
# 3. Either: labels = class_indices (return indices) OR labels = labels.argmax(dim=1) before criterion
# 4. Add: sample = sample.unsqueeze(0).permute(0, 3, 1, 2) before model(sample)
