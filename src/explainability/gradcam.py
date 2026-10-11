"""Grad-CAM generation for saved Keras image classifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for Grad-CAM") from exc
    return tf


def find_last_conv_layer(model: Any) -> Any:
    """Return the deepest 4-D convolution layer, including nested backbones."""
    candidates = []

    def visit(container):
        for layer in getattr(container, "layers", ()):
            if isinstance(layer, (tf.keras.Model, tf.keras.Sequential)):
                visit(layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                output = getattr(layer, "output", None)
                if output is not None and len(output.shape) == 4:
                    candidates.append(layer)

    tf = _require_tensorflow()
    visit(model)
    if not candidates:
        raise ValueError("model has no suitable 4-D Conv2D layer for Grad-CAM")
    return candidates[-1]


def make_gradcam_heatmap(model: Any, image: Any, class_index: int | None = None) -> tuple[np.ndarray, int]:
    """Create a normalized heatmap and return it with the selected class index."""
    tf = _require_tensorflow()
    layer = find_last_conv_layer(model)
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        with Image.open(image_path) as opened:
            array = preprocess_image(opened)
    else:
        array = preprocess_image(image)
    batch = tf.convert_to_tensor(np.expand_dims(array, 0), dtype=tf.float32)
    grad_model = tf.keras.models.Model(model.inputs, [layer.output, model.output])
    with tf.GradientTape() as tape:
        convolution, predictions = grad_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        score = predictions[:, class_index]
    gradients = tape.gradient(score, convolution)
    if gradients is None:
        raise ValueError("could not compute gradients for the selected convolution layer")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(convolution * weights[:, None, None, :], axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy().astype(np.float32), int(class_index)


def save_gradcam(
    model: Any,
    image: Any,
    *,
    actual: str | int | None = None,
    predicted: str | int | None = None,
    output_dir: str | Path = "outputs/evaluation/gradcam",
    filename: str | None = None,
) -> Path:
    """Generate and save a heatmap overlay, returning its output path."""
    heatmap, selected = make_gradcam_heatmap(model, image)
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            original = opened.convert("RGB")
    elif isinstance(image, Image.Image):
        original = image.convert("RGB")
    else:
        original = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    heat = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255)).resize(original.size, Image.Resampling.BILINEAR)
    # A compact red/yellow map avoids making matplotlib a runtime requirement.
    heat_rgb = np.asarray(heat)
    color = np.zeros((*heat_rgb.shape, 3), dtype=np.uint8)
    color[..., 0] = heat_rgb
    color[..., 1] = np.uint8(heat_rgb * 0.45)
    overlay = Image.blend(original, Image.fromarray(color), 0.45)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"gradcam_actual-{actual if actual is not None else 'unknown'}_predicted-{predicted if predicted is not None else selected}.png"
    path = destination / filename
    overlay.save(path)
    return path


# Descriptive aliases kept public for callers that prefer verb-based names.
identify_final_conv_layer = find_last_conv_layer
generate_gradcam = save_gradcam
