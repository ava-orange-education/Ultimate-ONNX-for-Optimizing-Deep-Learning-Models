"""Postprocessing functions for YOLO outputs."""

import os
import requests
from pathlib import Path
import json
import numpy as np
import torch
import torchvision
from typing import List, Optional, Tuple, Union


# Constants
CLASS_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/refs/heads/master/data/coco.names"
LOCAL_CLASS_PATH = "coco.names"


def get_class_names(
    file_path: str = LOCAL_CLASS_PATH, download_url: str = CLASS_NAMES_URL
) -> List[str]:
    """
    Load class names, downloading default COCO names if not present.

    Args:
        file_path: Local path to class names file
        download_url: URL to download default classes

    Returns:
        List of class names

    Example:
        >>> class_names = get_class_names()
        >>> print(class_names[:5])  # First 5 COCO classes
    """
    try:
        # Create parent directory if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Download if file doesn't exist
        if not Path(file_path).exists():
            if download_url is None:
                raise ValueError("No local class file and no download URL provided")

            response = requests.get(download_url)
            response.raise_for_status()

            with open(file_path, "w") as f:
                f.write(response.text)

        # Load and return class names
        with open(file_path, "r") as f:
            data = f.read().splitlines()

        return data

    except Exception as e:
        raise RuntimeError(f"Failed to load class names: {e}") from e


def xywh_to_xyxy(
    boxes: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Input boxes in xywh format

    Returns:
        Converted boxes in xyxy format

    Raises:
        AssertionError: If last dimension isn't 4
    """
    assert boxes.shape[-1] == 4, f"Expected 4 values per box, got {boxes.shape[-1]}"

    converted = (
        torch.empty_like(boxes)
        if isinstance(boxes, torch.Tensor)
        else np.empty_like(boxes)
    )

    centers = boxes[..., :2]
    half_wh = boxes[..., 2:] / 2
    converted[..., :2] = centers - half_wh  # top-left
    converted[..., 2:] = centers + half_wh  # bottom-right

    return converted


def non_max_suppression(
    predictions: Union[np.ndarray, torch.Tensor],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    max_detections: int = 300,
) -> List[torch.Tensor]:
    """
    Perform NMS on model outputs.

    Args:
        predictions: Raw model outputs
        conf_thresh: Minimum confidence threshold
        iou_thresh: NMS IoU threshold
        max_detections: Maximum detections per image

    Returns:
        List of filtered detections per image [x1,y1,x2,y2,conf,cls]
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.from_numpy(predictions)

    bs, _, num_preds = predictions.shape
    num_classes = predictions.shape[1] - 4

    # Filter by confidence
    conf_mask = predictions[:, 4 : 4 + num_classes].amax(1) > conf_thresh
    predictions = predictions.transpose(-1, -2)  # (bs, nc+4, n) -> (bs, n, nc+4)
    predictions[..., :4] = xywh_to_xyxy(predictions[..., :4])

    outputs = [torch.zeros((0, 6), device=predictions.device)] * bs

    for i, pred in enumerate(predictions):
        pred = pred[conf_mask[i]]  # Apply confidence threshold

        if not pred.shape[0]:
            continue

        # Get boxes and classes
        boxes, cls = pred.split((4, num_classes), 1)
        conf, cls_idx = cls.max(1, keepdim=True)
        detections = torch.cat((boxes, conf, cls_idx.float()), 1)
        detections = detections[conf.view(-1) > conf_thresh]

        # Apply NMS
        if detections.shape[0] > max_detections:
            detections = detections[
                detections[:, 4].argsort(descending=True)[:max_detections]
            ]

        keep = torchvision.ops.nms(detections[:, :4], detections[:, 4], iou_thresh)
        outputs[i] = detections[keep[:max_detections]]

    return outputs


def rescale_boxes(
    boxes: torch.Tensor,
    current_shape: Tuple[int, int],
    original_shape: Tuple[int, int],
    scale_pad: Optional[Tuple[float, float, float]] = None,
) -> torch.Tensor:
    """
    Rescale boxes from current image size back to original size.

    Args:
        boxes: Detected boxes [x1,y1,x2,y2,...]
        current_shape: (height, width) of processed image
        original_shape: (height, width) of original image
        scale_pad: Optional (scale, pad_w, pad_h)

    Returns:
        Rescaled boxes in original image coordinates
    """
    if scale_pad is None:
        scale = min(
            current_shape[0] / original_shape[0], current_shape[1] / original_shape[1]
        )
        pad_w = (current_shape[1] - original_shape[1] * scale) / 2
        pad_h = (current_shape[0] - original_shape[0] * scale) / 2
    else:
        scale, pad_w, pad_h = scale_pad

    boxes[:, [0, 2]] -= pad_w  # x coordinates
    boxes[:, [1, 3]] -= pad_h  # y coordinates
    boxes[:, :4] /= scale

    # Clip to image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_shape[0])

    return boxes
