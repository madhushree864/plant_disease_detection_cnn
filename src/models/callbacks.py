"""Callback factory kept separate from model construction."""

from __future__ import annotations

from .config import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_MIN_LEARNING_RATE,
    DEFAULT_REDUCE_LR_FACTOR,
    DEFAULT_REDUCE_LR_PATIENCE,
)


def make_training_callbacks(
    *,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    reduce_lr_patience: int = DEFAULT_REDUCE_LR_PATIENCE,
    reduce_lr_factor: float = DEFAULT_REDUCE_LR_FACTOR,
    min_learning_rate: float = DEFAULT_MIN_LEARNING_RATE,
):
    """Return early stopping, best-checkpoint, and learning-rate callbacks."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to create callbacks") from exc
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_learning_rate,
        ),
    ]
