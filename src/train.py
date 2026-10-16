
"""Training pipeline for YOLOv1.

Defines the training and validation loops, optimizer configuration,
loss computation, and model checkpointing.
"""

import torch
from torch.utils.data import DataLoader
import os

from model import model
from dataset import VOCDataset
from loss import YOLOv1Loss

# Configuration
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-3

voc_dirs = [
    "../data/VOCdevkit/VOC2007",
    "../data/VOCdevkit/VOC2012",
]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset and DataLoader
train_dataset = VOCDataset(voc_dirs, split="train")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Model, loss, optimizer
net = model().to(device)
criterion = YOLOv1Loss()
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4,
)

# Training loop
for epoch in range(EPOCHS):
    net.train()
    total_loss = 0.0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        y_pred = net(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"- Train Loss: {average_loss:.4f}"
    )

# Save model weights
os.makedirs("../checkpoints", exist_ok=True)
torch.save(net.state_dict(), "../checkpoints/yolov1_weights.pth")
print("Model weights saved.")