"""
UDMS Image Preprocessing — Standard preprocessing matching MobileNetV2 expectations.

This module is used by BOTH the training pipeline AND the inference runtime.
Do NOT create a separate src/inference/preprocessing.py — this is the single source.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
PIXEL_RANGE = (0.0, 1.0)


def load_and_preprocess(image_path: str | Path) -> np.ndarray:
    """Load image from disk and preprocess for model input.

    Steps:
    1. Open with PIL
    2. Convert to RGB (handles grayscale, RGBA)
    3. Resize to 224x224 using LANCZOS resampling
    4. Convert to numpy float32 array
    5. Normalize pixel values to [0, 1]

    Returns: numpy array shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_bytes(image_bytes: bytes) -> np.ndarray:
    """Same as load_and_preprocess but accepts raw bytes (for API endpoint).

    Returns: numpy array shape (224, 224, 3), dtype float32, values in [0, 1]
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def preprocess_image(image_input: str | Path | Image.Image) -> np.ndarray:
    """Preprocess an image from a file path or PIL Image for model input.

    Accepts either a file path (str/Path) or an already-opened PIL Image.
    Converts to RGB, resizes to 224x224, and normalizes to [0,1] float32.
    No augmentation is applied here — augmentation belongs in augmentation_pipeline.py.

    Args:
        image_input: A file path (str or Path) or a PIL Image instance.

    Returns:
        numpy array of shape (224, 224, 3), dtype float32, values in [0, 1].
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0

    with Image.open(image_input) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0


def add_batch_dimension(image: np.ndarray) -> np.ndarray:
    """Expand dims to (1, 224, 224, 3) for single-image inference."""
    return np.expand_dims(image, axis=0)

