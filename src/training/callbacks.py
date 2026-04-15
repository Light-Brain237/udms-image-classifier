"""
UDMS Training Callbacks — centralises all callback configuration.

Phase 1 (head training):  ModelCheckpoint, EarlyStopping, CSVLogger
Phase 2 (fine-tuning):    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
"""

import tensorflow as tf


def get_phase1_callbacks(checkpoint_dir: str = "models/") -> list:
    """Return callbacks for Phase 1 (head training).

    Callbacks:
    1. ModelCheckpoint — save best model by val_accuracy.
    2. EarlyStopping — patience 5, restore best weights.
    3. CSVLogger — log metrics to CSV.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "phase1_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            checkpoint_dir + "phase1_training_log.csv",
        ),
    ]


def get_phase2_callbacks(checkpoint_dir: str = "models/") -> list:
    """Return callbacks for Phase 2 (fine-tuning).

    Callbacks:
    1. ModelCheckpoint — save best model by val_accuracy.
    2. EarlyStopping — patience 5, restore best weights.
    3. ReduceLROnPlateau — halve LR on plateau (patience 3, min_lr 1e-7).
    4. CSVLogger — log metrics to CSV.
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "phase2_best.h5",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            checkpoint_dir + "phase2_training_log.csv",
        ),
    ]

