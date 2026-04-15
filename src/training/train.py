"""
UDMS Training Pipeline — Two-phase transfer learning.

Phase 1: Freeze backbone, train classification head only.
  - Optimizer: Adam(lr=1e-3)
  - Loss: CategoricalCrossentropy
  - Epochs: 20
  - Callbacks: ModelCheckpoint, EarlyStopping, CSVLogger

Phase 2: Unfreeze last 30 layers, fine-tune.
  - Optimizer: Adam(lr=1e-5)  — MUST be very low
  - Epochs: 15
  - Callbacks: + ReduceLROnPlateau

Data loading: tf.keras.utils.image_dataset_from_directory
  - image_size: (224, 224)
  - batch_size: 32
  - label_mode: 'categorical'
"""

import tensorflow as tf

from src.training.model import build_model, unfreeze_top_layers
from src.training.callbacks import get_phase1_callbacks, get_phase2_callbacks

BATCH_SIZE = 32
PHASE1_LR = 1e-3
PHASE1_EPOCHS = 20
PHASE2_LR = 1e-5
PHASE2_EPOCHS = 15
IMG_SIZE = (224, 224)


def load_data(
    data_dir: str, split: str = "train", batch_size: int = BATCH_SIZE
) -> tf.data.Dataset:
    """Load image dataset from directory structure.

    Args:
        data_dir: Root of processed data (e.g. ``data/processed``).
        split: One of ``train``, ``val``, ``test``.
        batch_size: Batch size.

    Returns:
        tf.data.Dataset yielding (image_batch, label_batch).
    """
    return tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/{split}",
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=(split == "train"),
    )


def train_phase1(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = PHASE1_EPOCHS,
    checkpoint_dir: str = "models/",
) -> tf.keras.callbacks.History:
    """Phase 1: compile with Adam(lr=1e-3), train head only.

    Saves best model to ``models/phase1_best.h5``.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_phase1_callbacks(checkpoint_dir),
    )
    return history


def train_phase2(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = PHASE2_EPOCHS,
    checkpoint_dir: str = "models/",
) -> tf.keras.callbacks.History:
    """Phase 2: unfreeze top 30 layers, recompile with Adam(lr=1e-5), fine-tune.

    Saves best model to ``models/phase2_best.h5``.
    """
    unfreeze_top_layers(model, num_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_phase2_callbacks(checkpoint_dir),
    )
    return history


def train_full_pipeline(data_dir: str = "data/processed") -> tf.keras.Model:
    """Orchestrate: build_model → Phase 1 → unfreeze → Phase 2 → return model."""
    train_ds = load_data(data_dir, "train")
    val_ds = load_data(data_dir, "val")

    model = build_model(freeze_backbone=True)

    print("=" * 60)
    print("PHASE 1 — Training classification head (backbone frozen)")
    print("=" * 60)
    train_phase1(model, train_ds, val_ds)

    print("\n" + "=" * 60)
    print("PHASE 2 — Fine-tuning top 30 backbone layers")
    print("=" * 60)
    train_phase2(model, train_ds, val_ds)

    return model


if __name__ == "__main__":
    train_full_pipeline()

