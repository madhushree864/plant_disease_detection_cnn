"""Streamlit front end for the saved plant-disease classifier."""

from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference import predict_image


DEFAULT_MODEL_PATH = "outputs/training/custom_cnn_baseline/best_model.keras"
DEFAULT_MAPPING_PATH = "outputs/data_split_manifest.csv"


@st.cache_data(show_spinner=False)
def _predict_cached(model_path: str, mapping_path: str, image_bytes: bytes) -> dict:
    """Cache deterministic inference results, including the uploaded image."""
    return predict_image(model_path, mapping_path, image_bytes, top_k=3)


def main() -> None:
    st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
    st.title("Plant Disease Detection")
    st.write(
        "Upload a leaf image to classify it with the saved Custom CNN baseline. "
        "EfficientNet ImageNet weights were unavailable in this environment."
    )

    with st.sidebar:
        st.header("Inference configuration")
        model_path = st.text_input(
            "Model path",
            value=os.getenv("AML_MODEL_PATH", DEFAULT_MODEL_PATH),
            help="Can also be set with AML_MODEL_PATH.",
        ).strip()
        mapping_path = st.text_input(
            "Class mapping path",
            value=os.getenv("AML_CLASS_MAPPING_PATH", DEFAULT_MAPPING_PATH),
            help="Can also be set with AML_CLASS_MAPPING_PATH.",
        ).strip()

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    image_bytes = uploaded.getvalue()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption=uploaded.name, use_container_width=True)
    except (OSError, ValueError) as exc:
        st.error(f"Unable to read the uploaded image: {exc}")
        return

    missing = []
    if not model_path or not Path(model_path).is_file():
        missing.append(f"model file not found: {model_path}")
    if not mapping_path or not Path(mapping_path).is_file():
        missing.append(f"class mapping file not found: {mapping_path}")
    if missing:
        st.error(
            "Inference is unavailable because the required model or class mapping "
            "is missing. " + "; ".join(missing)
        )
        return

    try:
        result = _predict_cached(model_path, mapping_path, image_bytes)
    except Exception as exc:
        st.error(
            "Inference is unavailable. Check that the saved model, class mapping, "
            f"and TensorFlow installation are compatible. Details: {exc}"
        )
        return

    st.subheader("Prediction")
    st.metric("Predicted class", result["class_name"])
    st.metric("Confidence", f"{result['confidence']:.2%}")
    st.subheader("Top-3 predictions")
    st.dataframe(
        [
            {
                "Class": item["class_name"],
                "Confidence": f"{item['confidence']:.2%}",
            }
            for item in result["top_k"]
        ],
        hide_index=True,
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
