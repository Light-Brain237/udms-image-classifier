"""Tests for src.data.augmentation_pipeline."""
import numpy as np
import pytest
import albumentations as A

from src.data.augmentation_pipeline import (
    augment_image,
    get_augmentation_pipeline,
    get_training_augmentation,
    get_validation_transform,
)


def _img() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (300, 300, 3), dtype=np.uint8)


class TestGetTrainingAugmentation:
    def test_returns_compose(self):
        assert isinstance(get_training_augmentation(), A.Compose)

    def test_output_shape(self):
        result = get_training_augmentation()(image=_img())["image"]
        assert result.shape == (224, 224, 3)

    def test_output_dtype(self):
        result = get_training_augmentation()(image=_img())["image"]
        assert result.dtype == np.uint8


class TestGetValidationTransform:
    def test_returns_compose(self):
        assert isinstance(get_validation_transform(), A.Compose)

    def test_output_shape(self):
        result = get_validation_transform()(image=_img())["image"]
        assert result.shape == (224, 224, 3)

    def test_deterministic(self):
        t = get_validation_transform()
        img = _img()
        np.testing.assert_array_equal(t(image=img)["image"], t(image=img)["image"])


class TestAugmentImage:
    def test_output_shape(self):
        result = augment_image(_img(), get_training_augmentation())
        assert result.shape == (224, 224, 3)

    def test_returns_ndarray(self):
        result = augment_image(_img(), get_training_augmentation())
        assert isinstance(result, np.ndarray)

    def test_validation_augment(self):
        result = augment_image(_img(), get_validation_transform())
        assert result.shape == (224, 224, 3)


class TestGetAugmentationPipeline:
    def test_returns_compose(self):
        assert isinstance(get_augmentation_pipeline(), A.Compose)

    def test_output_shape(self):
        result = get_augmentation_pipeline()(image=_img())["image"]
        assert result.shape == (224, 224, 3)
