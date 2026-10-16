
"""Inference utilities for YOLOv1.

Decodes model predictions, filters low-confidence detections,
and applies class-wise Non-Maximum Suppression (NMS).
"""

import torch


S = 7
B = 2
C = 20


def decode_predictions(prediction, conf_threshold=0.25):
    """Decode one (7, 7, 30) prediction into class-specific detections.

    Returns detections with class_id, score, and normalized
    bounding boxes in (x_center, y_center, width, height) format.
    """
    detections = []

    for row in range(S):
        for col in range(S):
            cell = prediction[row, col]
            class_probs = cell[10:30]

            for box_idx in range(B):
                start = box_idx * 5
                x_cell, y_cell, w, h, confidence = cell[start:start + 5]

                x = (col + x_cell.item()) / S
                y = (row + y_cell.item()) / S
                width = w.item()
                height = h.item()

                for class_id in range(C):
                    score = confidence.item() * class_probs[class_id].item()

                    if score < conf_threshold:
                        continue

                    detections.append({
                        "class_id": class_id,
                        "score": score,
                        "box": (x, y, width, height),
                    })

    return detections


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in (x_center, y_center, w, h) format."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    intersection_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersection_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection = intersection_w * intersection_h

    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections, iou_threshold=0.5):
    """Apply class-wise NMS to decoded detections."""
    kept_detections = []

    class_ids = {detection["class_id"] for detection in detections}

    for class_id in class_ids:
        class_detections = [
            detection for detection in detections
            if detection["class_id"] == class_id
        ]
        class_detections.sort(
            key=lambda detection: detection["score"],
            reverse=True,
        )

        while class_detections:
            best = class_detections.pop(0)
            kept_detections.append(best)

            class_detections = [
                detection for detection in class_detections
                if calculate_iou(best["box"], detection["box"]) <= iou_threshold
            ]

    return sorted(
        kept_detections,
        key=lambda detection: detection["score"],
        reverse=True,
    )


def predict(net, images, conf_threshold=0.25, iou_threshold=0.5):
    """Run inference on a batch of images.

    Args:
        net: YOLOv1 model.
        images: Tensor of shape (N, 3, 448, 448).
        conf_threshold: Minimum confidence score.
        iou_threshold: IoU threshold for NMS.

    Returns:
        A list containing the detections for each image.
    """
    net.eval()

    with torch.no_grad():
        predictions = net(images)

    batch_detections = []

    for prediction in predictions:
        detections = decode_predictions(
            prediction,
            conf_threshold=conf_threshold,
        )
        detections = non_max_suppression(
            detections,
            iou_threshold=iou_threshold,
        )
        batch_detections.append(detections)

    return batch_detections