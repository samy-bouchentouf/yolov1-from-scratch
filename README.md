# YOLOv1 from Scratch

A PyTorch implementation of YOLOv1 built from scratch for object detection on the PASCAL VOC dataset.

The project covers the main stages of an object detection pipeline, from dataset preparation and model training to inference and bounding box visualization. The goal is to understand and implement the core components of YOLOv1 rather than rely on a pretrained detector.

## Overview

YOLOv1 (*You Only Look Once*) is a single-stage object detection model that predicts bounding boxes and class probabilities directly from an image in a single forward pass.

This project implements a YOLOv1-style architecture trained from random initialization, with a focus on understanding the model architecture, loss function, training process, and inference pipeline.

### Features

- PASCAL VOC 2007 and 2012 dataset loading and preprocessing
- YOLOv1 architecture implemented in PyTorch
- Multi-component YOLOv1 loss function
- Training and validation pipeline with best-model checkpointing
- Bounding box decoding and confidence-based filtering
- Class-wise Non-Maximum Suppression (NMS)
- Inference visualization with predicted bounding boxes, class labels, and confidence scores
- Visual comparison between predictions and ground-truth annotations

## Architecture

The model takes a `448 × 448` RGB image as input and produces a `7 × 7 × 30` prediction tensor.

The image is divided into a `7 × 7` grid. Each grid cell predicts:

- 2 bounding boxes, each represented by 4 coordinates and 1 confidence score
- 20 class probabilities

The output tensor therefore has the shape:

`7 × 7 × (2 × 5 + 20) = 7 × 7 × 30`

The network consists of convolutional layers followed by fully connected layers, with Leaky ReLU activations.

## Dataset

The project uses the PASCAL VOC 2007 and VOC 2012 datasets, which provide images annotated with bounding boxes for 20 object classes.

The classes are:

`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

### Expected directory structure

Download and extract the VOC datasets so that the directory structure is:

```text
data/
└── VOCdevkit/
    ├── VOC2007/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   └── ImageSets/
    └── VOC2012/
        ├── Annotations/
        ├── JPEGImages/
        └── ImageSets/
```

Images are resized to `448 × 448`, and their annotations are encoded into `7 × 7 × 30` target tensors.

The `data/` directory is excluded from version control. The datasets must be downloaded separately before training or running dataset-dependent notebooks.

## Project Structure

```text
yolov1-from-scratch/
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── inference.ipynb
│   └── train.ipynb
├── src/
│   ├── dataset.py
│   ├── inference.py
│   ├── loss.py
│   ├── model.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/samy-bouchentouf/yolov1-from-scratch.git
cd yolov1-from-scratch
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure the PASCAL VOC datasets are available in the expected `data/VOCdevkit/` directory before training.

A CUDA-enabled GPU is recommended to speed up training.

## Training

The training pipeline is implemented in `src/train.py`, with an interactive notebook available in `notebooks/train.ipynb`.

The current training configuration uses:

| Parameter | Value |
|---|---|
| Input resolution | `448 × 448` |
| Batch size | 24 |
| Number of epochs | 20 |
| Optimizer | SGD |
| Initial learning rate | `1e-4` |
| Momentum | `0.9` |
| Weight decay | `5e-4` |

The model is trained from random initialization. VOC 2007 and VOC 2012 training images are used for training, while VOC 2012 validation images are used to monitor validation loss.

The model checkpoint is saved to `checkpoints/yolov1_best.pth` when the validation loss improves. The `checkpoints/` directory is excluded from version control, so the checkpoint must be generated locally by running the training pipeline.

## Inference

The inference utilities are implemented in `src/inference.py`, and the workflow can be explored in `notebooks/inference.ipynb`.

The inference pipeline includes:

1. Loading the trained model checkpoint.
2. Running a forward pass on input images.
3. Decoding the predicted bounding boxes and class scores.
4. Filtering detections by confidence.
5. Applying class-wise Non-Maximum Suppression (NMS).
6. Visualizing predicted bounding boxes, class labels, and confidence scores.
7. Comparing predictions with ground-truth bounding boxes.

The inference utilities support both individual predictions and batches of images.

To run inference with the trained model, first generate the checkpoint by running the training pipeline. The inference notebook also requires the relevant dataset images and annotations.

## Current Limitations

This is an initial implementation focused on understanding and building the YOLOv1 pipeline from scratch.

- The model is trained from random initialization rather than pretrained weights.
- Model assessment currently relies on validation loss and visual inspection of predictions.
- Mean Average Precision (mAP) evaluation has not yet been implemented.
- Some architectural or training details may differ from the original YOLOv1 paper.

## Future Improvements

- Implement mAP evaluation at IoU 0.5.
- Experiment with training hyperparameters and loss components.
- Investigate training stability and detection quality.
- Compare model performance across different configurations.

## References

- Redmon, J. et al. [*You Only Look Once: Unified, Real-Time Object Detection*](https://arxiv.org/abs/1506.02640), 2016.
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/).