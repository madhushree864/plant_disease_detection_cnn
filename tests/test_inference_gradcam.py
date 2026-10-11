import json

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from src.inference import predict_image
from src.explainability.gradcam import find_last_conv_layer, save_gradcam


def _tiny_model():
    inputs = tf.keras.Input((224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, activation="relu", name="features")(inputs)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(pooled)
    return tf.keras.Model(inputs, outputs)


def test_inference_returns_stable_structure(tmp_path):
    model_path = tmp_path / "model.keras"
    mapping_path = tmp_path / "classes.json"
    image_path = tmp_path / "image.png"
    _tiny_model().save(model_path)
    mapping_path.write_text(json.dumps({"0": "healthy", "1": "diseased"}), encoding="utf-8")
    Image.new("RGB", (32, 32), "white").save(image_path)

    result = predict_image(model_path, mapping_path, image_path, top_k=2)

    assert set(result) == {"class_index", "class_name", "confidence", "top_k"}
    assert len(result["top_k"]) == 2
    assert result["class_name"] in {"healthy", "diseased"}


def test_gradcam_discovers_and_saves_overlay(tmp_path):
    model = _tiny_model()
    assert find_last_conv_layer(model).name == "features"
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    output = save_gradcam(model, image, output_dir=tmp_path, actual="healthy", predicted="healthy")
    assert output.is_file()
