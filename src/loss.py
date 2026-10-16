"""YOLOv1 loss function.

Implements the multi-part loss used to train YOLOv1, including
bounding box coordinate regression, object confidence prediction,
no-object confidence penalization, and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


S = 7
B = 2
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5


def xy_loss(predictions, targets, responsible):
    """Compute the loss on bounding box center coordinates."""
    pred_xy = predictions[..., :2]
    target_xy = targets[..., :2]

    return ((pred_xy - target_xy).square().sum(dim=-1) * responsible).sum()


def wh_loss(predictions, targets, responsible):
    """Compute the loss on bounding box width and height."""
    pred_wh = predictions[..., 2:4].clamp(min=0)
    target_wh = targets[..., 2:4]

    return (
        (torch.sqrt(pred_wh) - torch.sqrt(target_wh)).square().sum(dim=-1)
        * responsible
    ).sum()


def no_object_confidence_loss(pred_conf, no_object_mask):
    """Penalize confidence predictions for non-responsible boxes."""
    return (pred_conf.square() * no_object_mask).sum()


def classification_loss(predictions, targets, object_mask):
    """Compute class probability loss for cells containing objects."""
    pred_classes = predictions[..., 10:30]
    target_classes = targets[..., 10:30]

    return (
        (pred_classes - target_classes).square().sum(dim=-1) * object_mask
    ).sum()


class YOLOv1Loss(nn.Module):
    """Original YOLOv1 multi-part loss."""

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (N, S, S, 30).
            targets: Tensor of shape (N, S, S, 30).

        Returns:
            Scalar loss.
        """
        # Split predictions into two boxes and class probabilities
        pred_boxes = predictions[..., :10].reshape(
            *predictions.shape[:-1], B, 5
        )
        pred_xywh = pred_boxes[..., :4]
        pred_conf = pred_boxes[..., 4]

        # Targets contain the same ground-truth box in both slots
        target_boxes = targets[..., :10].reshape(
            *targets.shape[:-1], B, 5
        )
        target_xywh = target_boxes[..., :4]

        # Identify cells containing objects
        object_mask = (targets[..., 10:30].sum(dim=-1) > 0).float()

        # Convert cell-relative centers to image-relative coordinates for IoU
        device = predictions.device
        rows = torch.arange(S, device=device).view(1, S, 1, 1)
        cols = torch.arange(S, device=device).view(1, 1, S, 1)

        pred_xy = pred_xywh[..., :2].clone()
        target_xy = target_xywh[..., :2].clone()

        pred_xy[..., 0] = (pred_xy[..., 0] + cols) / S
        pred_xy[..., 1] = (pred_xy[..., 1] + rows) / S
        target_xy[..., 0] = (target_xy[..., 0] + cols) / S
        target_xy[..., 1] = (target_xy[..., 1] + rows) / S

        # Widths and heights are already normalized by image dimensions
        pred_wh = pred_xywh[..., 2:4].clamp(min=0)
        target_wh = target_xywh[..., 2:4]

        # Compute IoU
        pred_min = pred_xy - pred_wh / 2
        pred_max = pred_xy + pred_wh / 2
        target_min = target_xy - target_wh / 2
        target_max = target_xy + target_wh / 2

        intersection_min = torch.maximum(pred_min, target_min)
        intersection_max = torch.minimum(pred_max, target_max)
        intersection_wh = (intersection_max - intersection_min).clamp(min=0)
        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]

        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]
        union = pred_area + target_area - intersection
        iou = intersection / (union + 1e-6)

        # Assign responsibility to the box with the highest IoU
        responsible_idx = iou.argmax(dim=-1)
        responsible = F.one_hot(responsible_idx, num_classes=B).float()
        responsible = responsible * object_mask.unsqueeze(-1)

        # Localization loss
        coord_loss = (
            xy_loss(pred_xywh, target_xywh, responsible)
            + wh_loss(pred_xywh, target_xywh, responsible)
        )

        # Confidence loss for responsible boxes, with IoU as target
        responsible_iou = (iou.detach() * responsible).sum(dim=-1)
        responsible_conf = (pred_conf * responsible).sum(dim=-1)

        obj_conf_loss = (
            (responsible_conf - responsible_iou).square() * object_mask
        ).sum()

        # Penalize all non-responsible boxes
        no_object_mask = 1.0 - responsible
        noobj_conf_loss = no_object_confidence_loss(
            pred_conf, no_object_mask
        )

        # Classification loss
        class_loss = classification_loss(predictions, targets, object_mask)

        total_loss = (
            LAMBDA_COORD * coord_loss
            + obj_conf_loss
            + LAMBDA_NOOBJ * noobj_conf_loss
            + class_loss
        )

        return total_loss