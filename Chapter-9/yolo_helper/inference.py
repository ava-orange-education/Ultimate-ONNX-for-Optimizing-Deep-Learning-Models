"""Main inference pipeline for YOLO models."""

import time
import cv2
import numpy as np
import onnxruntime
from typing import Optional, Tuple
from yolo_helper.preprocessing import preprocess_image
from yolo_helper.postprocessing import non_max_suppression, rescale_boxes


class YOLOInference:
    def __init__(self, model_path: str):
        """
        Initialize YOLO inference session.

        Args:
            model_path: Path to ONNX model
        """
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(
        self, image_path: str, conf_thresh: float = 0.25, iou_thresh: float = 0.45
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Run full inference pipeline.

        Args:
            image_path: Path to input image
            conf_thresh: Confidence threshold
            iou_thresh: NMS IoU threshold

        Returns:
            Tuple of (image, detections) or None if no detections
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor, orig_shape, scale_pad = preprocess_image(image_rgb)

        # Run inference
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time:.3f}s")

        # Process outputs
        detections = non_max_suppression(outputs[0], conf_thresh, iou_thresh)

        if detections[0].shape[0] > 0:
            detections[0][:, :4] = rescale_boxes(
                detections[0][:, :4], input_tensor.shape[2:], orig_shape, scale_pad
            )
            return image, detections[0].cpu().numpy()

        return None
