"""Visualization functions for detection results."""

import cv2
import numpy as np
from typing import List, Optional, Tuple


def draw_detections(
    image: np.ndarray,
    detections: np.ndarray,
    class_names: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw detection boxes and labels on image.

    Args:
        image: Input image (BGR format)
        detections: Array of detections [x1,y1,x2,y2,conf,cls]
        class_names: Optional list of class names
        color: Box color (BGR)
        thickness: Box thickness

    Returns:
        Image with drawn detections
    """
    if detections.size == 0:
        return image

    img_height, img_width = image.shape[:2]
    font_scale = min(img_width, img_height) / 1000
    font_thickness = max(1, int(font_scale))

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)

        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

        # Create label
        label = f"{class_names[cls_id]} {conf:.2f}" if class_names else f"{conf:.2f}"

        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Draw text background
        cv2.rectangle(
            image,
            (int(x1), int(y1) - text_height - 5),
            (int(x1) + text_width, int(y1)),
            color,
            -1,  # Filled rectangle
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness,
            cv2.LINE_AA,
        )

    return image
