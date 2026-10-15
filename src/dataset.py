"""Dataset loading and preprocessing for YOLOv1.

Handles image loading, annotation parsing, image preprocessing, and
target encoding for object detection on the PASCAL VOC dataset.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# YOLOv1 parameters
S = 7
B = 2
C = 20
IMAGE_SIZE = 448

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}


def parse_annotation(xml_path):
    """Read bounding boxes and class labels from a VOC XML file."""
    root = ET.parse(xml_path).getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    objects = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text

        if class_name not in CLASS_TO_IDX:
            continue

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        objects.append({
            "class_id": CLASS_TO_IDX[class_name],
            "bbox": (xmin, ymin, xmax, ymax)
        })

    return objects, width, height


def get_cell(x_center, y_center):
    """Return the grid row and column containing a normalized center."""
    col = min(int(x_center * S), S - 1)
    row = min(int(y_center * S), S - 1)
    return row, col


def encode_target(objects, image_width, image_height):
    """Encode VOC objects into a YOLOv1 target of shape (S, S, 30)."""
    target = torch.zeros((S, S, B * 5 + C), dtype=torch.float32)

    for obj in objects:
        xmin, ymin, xmax, ymax = obj["bbox"]
        class_id = obj["class_id"]

        # Normalize box coordinates to the image
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        row, col = get_cell(x_center, y_center)

        # Center coordinates relative to the grid cell
        x_cell = x_center * S - col
        y_cell = y_center * S - row

        # YOLOv1 target: both boxes share the same ground truth
        target[row, col, 0:5] = torch.tensor(
            [x_cell, y_cell, box_width, box_height, 1.0]
        )
        target[row, col, 5:10] = torch.tensor(
            [x_cell, y_cell, box_width, box_height, 1.0]
        )

        # One-hot class vector
        target[row, col, 10 + class_id] = 1.0

    return target


class VOCDataset(Dataset):
    """PASCAL VOC dataset for YOLOv1."""

    def __init__(self, voc_dirs, split=None):
        """
        Args:
            voc_dirs: A directory path or list of VOC2007/VOC2012 directories.
            split: None, "train", "val", or "trainval".
                   None loads all annotations.
        """
        if isinstance(voc_dirs, (str, Path)):
            voc_dirs = [voc_dirs]

        if split not in (None, "train", "val", "trainval"):
            raise ValueError("split must be None, 'train', 'val', or 'trainval'")

        self.samples = []

        for voc_dir in voc_dirs:
            voc_dir = Path(voc_dir)
            image_dir = voc_dir / "JPEGImages"
            annotation_dir = voc_dir / "Annotations"

            if split is None:
                xml_paths = sorted(annotation_dir.glob("*.xml"))
            else:
                split_file = voc_dir / "ImageSets" / "Main" / f"{split}.txt"

                if not split_file.exists():
                    raise FileNotFoundError(
                        f"Split file not found: {split_file}"
                    )

                image_ids = [
                    line.strip()
                    for line in split_file.read_text().splitlines()
                    if line.strip()
                ]

                xml_paths = [
                    annotation_dir / f"{image_id}.xml"
                    for image_id in image_ids
                ]

            for xml_path in xml_paths:
                image_path = image_dir / f"{xml_path.stem}.jpg"

                if image_path.exists() and xml_path.exists():
                    self.samples.append((image_path, xml_path))

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, xml_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        objects, width, height = parse_annotation(xml_path)

        image = self.transform(image)
        target = encode_target(objects, width, height)

        return image, target