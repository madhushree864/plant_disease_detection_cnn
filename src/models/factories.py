"""Model-definition factories for the plant disease classifier."""

from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_WEIGHTS,
)


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TensorFlow is required to build models") from exc
    return tf


def build_custom_cnn(
    num_classes: int = DEFAULT_NUM_CLASSES,
    *,
    input_shape: tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
):
    """Build a small CNN from scratch with a softmax classification head."""
    tf = _require_tensorflow()
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = inputs
    for filters, block in ((32, 1), (64, 2), (128, 3)):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu", name=f"conv{block}_1"
        )(x)
        x = tf.keras.layers.MaxPooling2D(name=f"pool{block}")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )(x)
    return tf.keras.Model(inputs, outputs, name="custom_cnn")


def build_efficientnetb0(
    num_classes: int = DEFAULT_NUM_CLASSES,
    *,
    input_shape: tuple[int, int, int] = DEFAULT_IMAGE_SIZE,
    weights: str | None = DEFAULT_WEIGHTS,
    dropout_rate: float = DEFAULT_DROPOUT_RATE,
    include_top: bool = False,
):
    """Build EfficientNetB0 with a replaceable classification head.

    ``weights`` deliberately defaults to ``None``. Pass ``"imagenet"`` only
    when downloading/preloading ImageNet weights is explicitly desired.
    """
    tf = _require_tensorflow()
    if include_top:
        raise ValueError("include_top must be False so the custom head is used")
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        name="efficientnetb0_backbone",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )(x)
    return tf.keras.Model(inputs, outputs, name="efficientnetb0_classifier")


def set_backbone_trainability(model: Any, trainable_layers: int = 0) -> Any:
    """Freeze the EfficientNet backbone except for its upper N layers.

    The classification head is left trainable. ``trainable_layers=0`` freezes
    the complete backbone; a negative value or a value larger than the
    backbone size is rejected to avoid accidentally training the full model.
    """
    backbone = model.get_layer("efficientnetb0_backbone")
    if not isinstance(trainable_layers, int) or trainable_layers < 0:
        raise ValueError("trainable_layers must be a non-negative integer")
    if trainable_layers > len(backbone.layers):
        raise ValueError("trainable_layers exceeds the number of backbone layers")
    # Set the nested model first; assigning ``trainable`` recursively would
    # otherwise undo the per-layer selection below.
    backbone.trainable = trainable_layers > 0
    split = len(backbone.layers) - trainable_layers
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= split
    return model
