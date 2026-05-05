"""Tests for src.data.preprocessing."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data.preprocessing import (
    IMG_SIZE,
    add_batch_dimension,
    load_and_preprocess,
    preprocess_bytes,
    preprocess_image,
)


def _jpeg(width=640, height=480, color="red") -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _tmp_jpg(tmp_path, w=640, h=480) -> str:
    p = tmp_path / "test.jpg"
    Image.new("RGB", (w, h), color="blue").save(str(p))
    return str(p)


class TestLoadAndPreprocess:
    def test_output_shape(self, tmp_path):
        assert load_and_preprocess(_tmp_jpg(tmp_path)).shape == (224, 224, 3)

    def test_output_dtype(self, tmp_path):
        assert load_and_preprocess(_tmp_jpg(tmp_path)).dtype == np.float32

    def test_pixel_range(self, tmp_path):
        arr = load_and_preprocess(_tmp_jpg(tmp_path))
        assert arr.min() >= 0.0 and arr.max() <= 255.0

    def test_grayscale_to_rgb(self, tmp_path):
        p = tmp_path / "g.png"
        Image.new("L", (300, 300), 100).save(str(p))
        assert load_and_preprocess(str(p)).shape == (224, 224, 3)

    def test_rgba_to_rgb(self, tmp_path):
        p = tmp_path / "a.png"
        Image.new("RGBA", (300, 300), (100, 100, 100, 128)).save(str(p))
        assert load_and_preprocess(str(p)).shape == (224, 224, 3)

    def test_accepts_pathlib(self, tmp_path):
        assert load_and_preprocess(Path(_tmp_jpg(tmp_path))).shape == (224, 224, 3)


class TestPreprocessBytes:
    def test_shape(self):
        assert preprocess_bytes(_jpeg()).shape == (224, 224, 3)

    def test_dtype(self):
        assert preprocess_bytes(_jpeg()).dtype == np.float32

    def test_pixel_range(self):
        arr = preprocess_bytes(_jpeg())
        assert arr.min() >= 0.0 and arr.max() <= 255.0

    def test_upscale_small(self):
        assert preprocess_bytes(_jpeg(50, 50)).shape == (224, 224, 3)

    def test_downscale_large(self):
        assert preprocess_bytes(_jpeg(1920, 1080)).shape == (224, 224, 3)

    def test_grayscale_bytes(self):
        buf = io.BytesIO()
        Image.new("L", (300, 300), 128).save(buf, format="JPEG")
        assert preprocess_bytes(buf.getvalue()).shape == (224, 224, 3)


class TestPreprocessImage:
    def test_from_string_path(self, tmp_path):
        arr = preprocess_image(_tmp_jpg(tmp_path))
        assert arr.shape == (224, 224, 3) and arr.dtype == np.float32

    def test_from_pathlib(self, tmp_path):
        assert preprocess_image(Path(_tmp_jpg(tmp_path))).shape == (224, 224, 3)

    def test_from_pil_image(self):
        arr = preprocess_image(Image.new("RGB", (300, 300), "green"))
        assert arr.shape == (224, 224, 3) and arr.dtype == np.float32

    def test_pil_grayscale(self):
        assert preprocess_image(Image.new("L", (300, 300), 50)).shape == (224, 224, 3)


class TestAddBatchDimension:
    def test_shape(self):
        arr = np.zeros((224, 224, 3), dtype=np.float32)
        assert add_batch_dimension(arr).shape == (1, 224, 224, 3)

    def test_dtype(self):
        arr = np.zeros((224, 224, 3), dtype=np.float32)
        assert add_batch_dimension(arr).dtype == np.float32

    def test_values_unchanged(self):
        arr = np.ones((224, 224, 3), dtype=np.float32) * 128.0
        np.testing.assert_array_equal(add_batch_dimension(arr)[0], arr)


def test_img_size_constant():
    assert IMG_SIZE == (224, 224)
