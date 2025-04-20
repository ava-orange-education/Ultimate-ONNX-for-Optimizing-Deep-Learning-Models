"""Image preprocessing functions for YOLO models."""

import cv2
import numpy as np
from typing import Tuple


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize image with letterbox padding while maintaining aspect ratio.

    Args:
        image: Input image (H,W,C)
        target_size: (width, height) to resize to
        color: Padding color (RGB)

    Returns:
        Tuple of (padded_image, scale_ratio, (pad_width, pad_height))

    Example:
        >>> img, ratio, (dw, dh) = letterbox_resize(image, (640, 640))
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate scaling and padding
    scale = min(target_width / width, target_height / height)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    left, top = (target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2

    # Resize and pad
    if (width, height) != new_size:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    right = target_width - new_size[0] - left
    bottom = target_height - new_size[1] - top

    # Add padding
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, scale, (left, top)


def preprocess_image(
    image: np.ndarray, target_size: Tuple[int, int] = (640, 640)
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float, float]]:
    """
    Full preprocessing pipeline for YOLO input.

    Args:
        image: Input image (H,W,C)
        target_size: Target dimensions (width, height)

    Returns:
        Tuple of (processed_tensor, original_shape, (scale, pad_w, pad_h))
    """
    orig_h, orig_w = image.shape[:2]

    # Letterbox resize
    img, scale, (pad_w, pad_h) = letterbox_resize(image, target_size)

    # Convert to CHW format and normalize
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img, (orig_h, orig_w), (scale, pad_w, pad_h)
