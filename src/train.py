
"""Training pipeline for YOLOv1.

Defines the training and validation loops, optimizer configuration,
loss computation, and best-model checkpointing.
"""

import os
import torch
from torch.utils.data import DataLoader

from model import YOLOv1
from dataset import VOCDataset
from loss import YOLOv1Loss

# Configuration
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

voc2007_dir = "data/VOCdevkit/VOC2007"
voc2012_dir = "data/VOCdevkit/VOC2012"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Training: VOC 2007 + VOC 2012 train splits
train_dataset = VOCDataset(
    [voc2007_dir, voc2012_dir],
    split="train",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Validation: VOC 2012 val split only
val_dataset = VOCDataset(
    [voc2012_dir],
    split="val",
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Model, loss, optimizer
net = YOLOv1().to(device)
criterion = YOLOv1Loss()

optimizer = torch.optim.SGD(
    net.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4,
)

# Checkpoint configuration
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")
checkpoint_path = "checkpoints/yolov1_best.pth"

# Training and validation
for epoch in range(EPOCHS):
    # Training
    net.train()
    train_total_loss = 0.0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = net(X)
        loss = criterion(y_pred, y)

        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()

    avg_train_loss = train_total_loss / len(train_loader)

    # Validation
    net.eval()
    val_total_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            y_pred = net(X)
            loss = criterion(y_pred, y)

            val_total_loss += loss.item()

    avg_val_loss = val_total_loss / len(val_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {avg_train_loss:.4f} "
        f"| Val Loss: {avg_val_loss:.4f}"
    )

    # Save only when validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save(
            net.state_dict(),
            checkpoint_path,
        )

print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
print(f"Best model saved at: {checkpoint_path}")