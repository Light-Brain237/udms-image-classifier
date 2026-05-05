"""Tests for src.training.model — architecture only, no training."""
import pytest

try:
    import tensorflow as tf
    tf.keras  # raises AttributeError if keras not wired into tf
    _KERAS_AVAILABLE = True
except (ImportError, AttributeError):
    _KERAS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _KERAS_AVAILABLE,
    reason="TensorFlow/Keras not available in this environment",
)

from src.training.model import build_model, get_model_summary, unfreeze_top_layers


@pytest.fixture(scope="module")
def frozen_model():
    return build_model(freeze_backbone=True)


class TestBuildModel:
    def test_input_shape(self, frozen_model):
        assert frozen_model.input_shape == (None, 224, 224, 3)

    def test_output_shape_5_classes(self, frozen_model):
        assert frozen_model.output_shape == (None, 5)

    def test_backbone_frozen(self, frozen_model):
        backbone = next(
            l for l in frozen_model.layers
            if hasattr(l, "layers") and "mobilenet" in l.name.lower()
        )
        assert not backbone.trainable

    def test_has_two_dropout_layers(self, frozen_model):
        drops = [l for l in frozen_model.layers if isinstance(l, tf.keras.layers.Dropout)]
        assert len(drops) == 2

    def test_has_dense_128(self, frozen_model):
        dense = [l for l in frozen_model.layers
                 if isinstance(l, tf.keras.layers.Dense) and l.units == 128]
        assert len(dense) == 1

    def test_output_activation_softmax(self, frozen_model):
        out_layer = next(
            l for l in frozen_model.layers
            if isinstance(l, tf.keras.layers.Dense) and l.units == 5
        )
        assert out_layer.activation.__name__ == "softmax"

    def test_custom_num_classes(self):
        m = build_model(num_classes=3)
        assert m.output_shape == (None, 3)

    def test_unfrozen_model(self):
        m = build_model(freeze_backbone=False)
        backbone = next(
            l for l in m.layers
            if hasattr(l, "layers") and "mobilenet" in l.name.lower()
        )
        assert backbone.trainable


class TestUnfreezeTopLayers:
    def test_backbone_becomes_trainable(self):
        model = build_model(freeze_backbone=True)
        model = unfreeze_top_layers(model, num_layers=10)
        backbone = next(
            l for l in model.layers
            if hasattr(l, "layers") and "mobilenet" in l.name.lower()
        )
        assert backbone.trainable

    def test_some_layers_stay_frozen(self):
        model = build_model(freeze_backbone=True)
        model = unfreeze_top_layers(model, num_layers=10)
        backbone = next(
            l for l in model.layers
            if hasattr(l, "layers") and "mobilenet" in l.name.lower()
        )
        frozen = [l for l in backbone.layers if not l.trainable]
        assert len(frozen) > 0


class TestGetModelSummary:
    def test_returns_string(self, frozen_model):
        summary = get_model_summary(frozen_model)
        assert isinstance(summary, str)

    def test_contains_mobilenet(self, frozen_model):
        assert "mobilenet" in get_model_summary(frozen_model).lower()
