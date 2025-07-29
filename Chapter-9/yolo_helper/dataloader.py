"""Data loading functions used during quantization."""

import os
import numpy as np
import glob
from typing import Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
import cv2
from onnxruntime.quantization import CalibrationDataReader
from torch.utils.data import Subset
from yolo_helper.preprocessing import preprocess_image
import random

# Set random seed for reproducibility
random.seed(42)


class COCODataset(Dataset):
    """Custom Dataset class for loading and preprocessing COCO dataset images.

    Args:
        root_dir (str): Path to directory containing images
        sample (int): Number of images to randomly sample from the directory.
                     If None, uses all images. Default: 100.
    """

    def __init__(self, root_dir: str, sample: Optional[int] = 100):
        self.root_dir = root_dir

        # Validate root directory exists
        if not os.path.isdir(self.root_dir):
            raise RuntimeError(f"Image root directory not found at {self.root_dir}")

        # Get all JPEG images in directory
        self.img_paths = glob.glob(os.path.join(self.root_dir, "*.jpg"))

        # Randomly sample images if sample size is specified
        if sample is not None:
            if sample > len(self.img_paths):
                raise ValueError(
                    f"Sample size {sample} exceeds available images ({len(self.img_paths)})"
                )
            self.img_paths = random.sample(self.img_paths, sample)

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Loads and preprocesses an image at the given index.

        Args:
            index (int): Index of the image to load

        Returns:
            dict: Dictionary containing the preprocessed image with key 'images'
        """
        img_path = self.img_paths[index]

        # Read and convert image color space
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess image using YOLO helper function
        img, _, _ = preprocess_image(img)

        return {"images": img}


class YoloDataReader(CalibrationDataReader):
    """Adapter class to make PyTorch DataLoader work with ONNX Runtime calibration.

    Args:
        data_loader: PyTorch DataLoader yielding batches of preprocessed data
    """

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(data_loader)
        self.length = len(data_loader)

    def get_next(self) -> Optional[Dict[str, Any]]:
        """Gets the next batch of data for calibration.

        Returns:
            dict: A dictionary mapping input names to numpy arrays, or None if no more data
        """
        try:
            return next(self.iter)
        except StopIteration:
            return None

    def __len__(self) -> int:
        """Returns the number of batches in the data loader."""
        return self.length

    def set_range(self, start_index: int, end_index: int) -> None:
        """Sets the data range for calibration.

        Args:
            start_index: Starting index of the subset
            end_index: Ending index of the subset (exclusive)
        """
        dl = Subset(self.data_loader, indices=range(start_index, end_index))
        self.iter = iter(dl)
        self.length = len(dl)  # Update length to reflect subset size

    def rewind(self) -> None:
        """Resets the iterator to the beginning of the dataset."""
        self.iter = iter(self.data_loader)
        self.length = len(self.data_loader)
