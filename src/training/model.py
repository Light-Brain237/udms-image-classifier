"""
UDMS Model Architecture — MobileNetV2 transfer learning.

Architecture:
  MobileNetV2 (ImageNet weights, include_top=False)
  → mobilenetv2_preprocess (Lambda: scales [0,255] → [-1,1])
  → GlobalAveragePooling2D
  → Dropout(0.3)
  → Dense(128, activation='relu')
  → Dropout(0.2)
  → Dense(5, activation='softmax')

Input shape: (224, 224, 3) — raw float32 pixels in [0, 255]
Output shape: (5,) — probability distribution over 5 categories

The preprocessing Lambda layer is exported into the TFLite artifact, so callers
never need to apply mobilenet_v2.preprocess_input themselves.  Training and
inference both supply raw [0, 255] values — no train/serve skew is possible.
"""

import tensorflow as tf

from src.data.category_mapping import NUM_CLASSES

INPUT_SHAPE = (224, 224, 3)
DENSE_UNITS = 128
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2


def build_model(
    num_classes: int = NUM_CLASSES, freeze_backbone: bool = True
) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model.

    Args:
        num_classes: Number of output classes (default 7).
        freeze_backbone: If True, freeze all MobileNetV2 layers (Phase 1).

    Returns:
        Uncompiled tf.keras.Model.
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_backbone

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenetv2_preprocess",
    )(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(DROPOUT_1)(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT_2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def unfreeze_top_layers(
    model: tf.keras.Model, num_layers: int = 30
) -> tf.keras.Model:
    """Unfreeze the last ``num_layers`` layers of the backbone for fine-tuning.

    Steps:
    1. Access the MobileNetV2 base (model.layers[1]).
    2. Set base.trainable = True.
    3. Freeze all layers except the last ``num_layers``.
    4. Return the model (needs recompilation in train.py).
    """
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    return model


def get_model_summary(model: tf.keras.Model) -> str:
    """Return model summary as string."""
    lines: list[str] = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)

