"""
UDMS Data Augmentation Pipeline — Applied ONLY to training images. Never to val/test.

Uses the Albumentations library.
Transforms chosen per the project plan for citizen-submitted urban disorder photos.
"""

import numpy as np
import albumentations as A


def get_training_augmentation() -> A.Compose:
    """Return the training augmentation pipeline.

    Transforms (in order):
    1. HorizontalFlip(p=0.5)
    2. Rotate(limit=15, p=0.5)              — ±15 degrees
    3. RandomBrightnessContrast(±20%, p=0.5)
    4. RandomScale(0.8x–1.2x, p=0.3)
    5. GaussNoise(var 10–50, p=0.3)          — simulates phone camera noise
    6. Resize(224, 224)                       — always resize last
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.GaussNoise(std_range=(0.1, 0.5), p=0.3),
        A.Resize(224, 224),
    ])


def get_validation_transform() -> A.Compose:
    """Return validation/test transform. Resize ONLY, no augmentation."""
    return A.Compose([
        A.Resize(224, 224),
    ])


def augment_image(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply augmentation to a single image. Returns augmented image."""
    result = transform(image=image)
    return result["image"]


def get_augmentation_pipeline() -> A.Compose:
    """Return the standard training augmentation pipeline.

    This pipeline is ONLY for training data — never apply to validation or test sets.

    Transforms:
    1. HorizontalFlip(p=0.5)
    2. Rotate(limit=15, p=0.5)              — ±15 degrees
    3. RandomBrightnessContrast(limit=0.2, p=0.5)
    4. RandomScale(scale_limit=0.2, p=0.3)
    5. GaussNoise(p=0.3)                    — simulates phone camera noise
    6. Resize(224, 224)                      — always resize last

    Returns:
        An Albumentations Compose pipeline ready for training augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.RandomScale(scale_limit=0.2, p=0.3),
        A.GaussNoise(std_range=(0.1, 0.5), p=0.3),
        A.Resize(224, 224),
    ])

