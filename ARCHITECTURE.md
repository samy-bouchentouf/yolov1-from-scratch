````
# YOLOv1 Architecture

---

# Project Structure

```text
yolov1-from-scratch/
│
├── checkpoints/
|   └── yolov1_best.pth
│
├── data/
│   └── VOCdevkit/
│       ├── VOC2007/
│       └── VOC2012/
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── inference.ipynb
│   └── train.ipynb
│
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── loss.py
│   ├── model.py
│   └── train.py
│
├── ARCHITECTURE.md
└── README.md
````

# Notebooks

## dataset_exploration.ipynb

Interactive notebook for exploring the dataset, inspecting images and annotations, visualizing bounding boxes, and analyzing class distributions.

## inference.ipynb

Interactive notebook for running inference on images, visualizing predicted bounding boxes and class labels, and inspecting detection confidence scores.

## train.ipynb

Interactive training notebook for running and monitoring training, inspecting batches, visualizing loss curves, and evaluating model predictions.

# Src

## dataset.py

"""Dataset loading and preprocessing for YOLOv1.

Handles image loading, annotation parsing, image preprocessing, and
target encoding for object detection on the PASCAL VOC dataset.
"""

## inference.py

"""Inference utilities for YOLOv1.

Decodes model predictions, filters low-confidence detections,
and applies class-wise Non-Maximum Suppression (NMS).
"""

## loss.py

"""YOLOv1 loss function.

Implements the multi-part loss used to train YOLOv1, including
bounding box coordinate regression, object confidence prediction,
no-object confidence penalization, and classification.
"""

## model.py

"""YOLOv1 model architecture implemented in PyTorch.

Defines the convolutional neural network and fully connected layers
used to predict bounding boxes, object confidence scores, and class
probabilities for each grid cell.
"""

## train.py

"""Training pipeline for YOLOv1.

Defines the training and validation loops, optimizer configuration,
loss computation, and best-model checkpointing.
"""