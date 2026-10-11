"""Reusable, single-image inference for saved Keras classifiers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_image


def load_class_mapping(path: str | Path) -> dict[int, str]:
    """Load the one authoritative index-to-class mapping from JSON or CSV."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"class mapping/manifest not found: {file_path}")
    if file_path.suffix.lower() == ".json":
        with file_path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        raw = raw.get("class_mapping", raw.get("classes", raw)) if isinstance(raw, dict) else raw
        if not isinstance(raw, dict):
            raise ValueError("mapping JSON must be an object or contain class_mapping")
        mapping = {}
        for key, value in raw.items():
            # Accept both {"0": "label"} and {"label": 0}.
            if str(key).lstrip("-").isdigit():
                mapping[int(key)] = str(value)
            elif str(value).lstrip("-").isdigit():
                mapping[int(value)] = str(key)
            else:
                raise ValueError("mapping JSON entries must pair indices and class names")
    elif file_path.suffix.lower() in {".csv", ".tsv"}:
        with file_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t" if file_path.suffix.lower() == ".tsv" else ","))
        if not rows:
            raise ValueError("class mapping CSV is empty")
        mapping = {}
        for row in rows:
            index = row.get("class_index", row.get("index"))
            label = row.get("label", row.get("class_name", row.get("class")))
            if index is not None and label is not None:
                mapping[int(index)] = str(label)
        if not mapping:
            raise ValueError("CSV must contain class_index/index and label/class_name columns")
    else:
        raise ValueError("class mapping must be JSON, CSV, or TSV")
    if not mapping or sorted(mapping) != list(range(len(mapping))):
        raise ValueError("class mapping indices must be a non-empty contiguous range starting at zero")
    return dict(sorted(mapping.items()))


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for inference") from exc
    return tf


def predict_image(
    model_path: str | Path,
    mapping_path: str | Path,
    image: str | Path | Image.Image | np.ndarray | bytes,
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Load a saved model and predict one image, returning stable metadata."""
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"saved Keras model not found: {model_file}")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    mapping = load_class_mapping(mapping_path)
    tf = _require_tensorflow()
    model = tf.keras.models.load_model(model_file)
    if isinstance(image, (str, Path)):
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        with Image.open(image_path) as opened:
            array = preprocess_image(opened)
    else:
        array = preprocess_image(image)
    probabilities = np.asarray(model.predict(np.expand_dims(array, 0), verbose=0))[0]
    if probabilities.ndim != 1 or len(probabilities) != len(mapping):
        raise ValueError("model output size does not match the class mapping")
    indices = np.argsort(probabilities)[::-1][: min(top_k, len(probabilities))]
    top = [
        {"class_index": int(index), "class_name": mapping[int(index)], "confidence": float(probabilities[index])}
        for index in indices
    ]
    predicted = int(indices[0])
    return {
        "class_index": predicted,
        "class_name": mapping[predicted],
        "confidence": float(probabilities[predicted]),
        "top_k": top,
    }
