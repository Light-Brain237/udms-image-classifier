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
TEMP_SPLIT = 0.30   # val + test together = 30%;  train = 70%
SEED = 42
DATA_DIR = "data/processed/all"


def load_data(
    data_dir: str = DATA_DIR,
    batch_size: int = BATCH_SIZE,
) -> "tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]":
    """Return (train_ds, val_ds, test_ds) with a 70 / 15 / 15 split.

    Strategy:
      1. ``validation_split=0.30, subset='training'``  → 70 % train
      2. ``validation_split=0.30, subset='validation'`` → 30 % temp
      3. temp split 50 / 50  → 15 % val  +  15 % test

    Args:
        data_dir: Flat directory with one subfolder per class.
        batch_size: Batch size.

    Returns:
        Tuple of (train_ds, val_ds, test_ds).
    """
    common = dict(
        image_size=IMG_SIZE,
        batch_size=None,
        label_mode="categorical",
        seed=SEED,
        validation_split=TEMP_SPLIT,
    )
    train_ds = (
        tf.keras.utils.image_dataset_from_directory(
            data_dir, subset="training", **common
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, subset="validation", **common
    )
    n_temp = sum(1 for _ in temp_ds)
    n_val = n_temp // 2
    val_ds = temp_ds.take(n_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = temp_ds.skip(n_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds


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


def train_full_pipeline(data_dir: str = DATA_DIR) -> tf.keras.Model:
    """Orchestrate: build_model → Phase 1 → unfreeze → Phase 2 → return model."""
    train_ds, val_ds, test_ds = load_data(data_dir)

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

