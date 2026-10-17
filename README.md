# YOLOv1 from Scratch

A PyTorch implementation of YOLOv1 built from scratch for object detection on the PASCAL VOC dataset.

The project covers the main stages of an object detection pipeline, from dataset preparation and model training to inference and bounding box visualization.

## Overview

YOLOv1 (*You Only Look Once*) is a single-stage object detection model that predicts bounding boxes and class probabilities directly from an image in a single forward pass.

This project implements a YOLOv1-style architecture from random initialization, with the goal of understanding the model's architecture, loss function, training process, and inference pipeline.

### Features

- PASCAL VOC 2007 and 2012 dataset loading and preprocessing
- YOLOv1 architecture implemented in PyTorch
- Multi-component YOLOv1 loss function
- Training and validation pipeline with best-model checkpointing
- Bounding box decoding and confidence-based filtering
- Class-wise Non-Maximum Suppression (NMS)
- Inference on images with bounding box and class label visualization

## Architecture

The model takes a `448 × 448` RGB image as input and produces a `7 × 7 × 30` prediction tensor.

Each grid cell predicts:
- 2 bounding boxes, each containing 4 coordinates and 1 confidence score
- 20 class probabilities

The output size is `7 × 7 × (2 × 5 + 20) = 7 × 7 × 30`.

The network consists of convolutional layers followed by fully connected layers, with Leaky ReLU activations.

## Dataset

The project uses the PASCAL VOC 2007 and VOC 2012 datasets, which contain images annotated with bounding boxes for 20 object classes.

The classes are:

`aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`.

Expected dataset structure:

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

Images are resized to `448 × 448`, and annotations are encoded into a `7 × 7 × 30` target tensor.

## Project Structure

```text
yolov1-from-scratch/
├── checkpoints/
│   └── yolov1_best.pth
├── data/
│   └── VOCdevkit/
│       ├── VOC2007/
│       └── VOC2012/
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
├── ARCHITECTURE.md
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

Make sure the PASCAL VOC dataset is available in the expected `data/VOCdevkit/` directory before training.

## Training

The training pipeline is implemented in `src/train.py`, with an interactive notebook available in `notebooks/train.ipynb`.

The current training configuration uses:

- Input resolution: `448 × 448`
- Batch size: 24
- Number of epochs: 50
- Optimizer: SGD
- Initial learning rate: `1e-4`
- Momentum: `0.9`
- Weight decay: `5e-4`

Training uses VOC 2007 and VOC 2012 training images, with VOC 2012 validation images used to monitor validation loss.

The best model is saved to `checkpoints/yolov1_best.pth` when the validation loss improves.

A CUDA-enabled GPU is recommended to speed up training.

## Inference

The inference pipeline is implemented in `src/inference.py` and can be explored in `notebooks/inference.ipynb`.

It includes:

1. Loading the trained model checkpoint.
2. Running a forward pass on input images.
3. Decoding the predicted bounding boxes and class scores.
4. Filtering detections by confidence.
5. Applying class-wise Non-Maximum Suppression (NMS).
6. Visualizing the final bounding boxes, predicted classes, and confidence scores.

The inference pipeline can be used on individual images or batches of images.

## Current Limitations

This is an initial implementation focused on understanding and building the YOLOv1 pipeline from scratch.

- The model is trained from random initialization.
- The current evaluation workflow focuses on validation loss and visual inspection of predictions.
- Mean Average Precision (mAP) evaluation has not yet been implemented.
- The implementation may differ from the original YOLOv1 paper in certain architectural or training details.

## Future Improvements

- Implement mAP evaluation at IoU 0.5.
- Compare predictions quantitatively across model versions.
- Experiment with training hyperparameters and loss components.
- Improve training stability and detection quality.

## References

- Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection* (2016). [Paper](https://arxiv.org/abs/1506.02640)
- [PASCAL VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/)