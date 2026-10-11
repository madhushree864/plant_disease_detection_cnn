# Plant Disease Detection Using Transfer Learning

Advanced Machine Learning project for Coimbatore Institute of Technology (19MAM51).

The original dataset is intentionally kept outside this repository. The
unaugmented labelled ZIP is read in place by the Phase 2 utilities under
`src/data`; see [reports/dataset_report.md](reports/dataset_report.md) and
[reports/project_status.md](reports/project_status.md) for verified details.

Training is not configured to run automatically. Phase 2 prepares and validates
a reproducible metadata manifest; Phase 3 provides bounded EDA via
`python -m src.eda ARCHIVE --max-images 1000`, saving plots under
`outputs/eda`. Phase 4 provides reusable 224x224 RGB decoding, normalization,
train-only flip augmentation, and `tf.data` adapters in
`src/preprocessing.py`. The extraction cache in `data/extracted` is created
only from the unaugmented ZIP, verifies the source SHA-256 and manifest counts
(55,448 images, 39 classes), and preserves the archive's class-directory
structure. No ZIP is modified. No model is trained. See
[reports/implementation_plan.md](reports/implementation_plan.md) for the
CPU-oriented plan.

Model definitions are available under `src/models`: `build_custom_cnn()` creates a small CNN from scratch and `build_efficientnetb0()` creates the transfer-learning variant. EfficientNet weights default to `None`; pass `weights="imagenet"` only when an explicit download is intended. `set_backbone_trainability()` supports gradual fine-tuning, and `make_training_callbacks()` creates the standard validation callbacks. Phase 6 adds opt-in orchestration under `src/training` and `src/evaluation`.
Given an existing archive and split manifest, `train_model()` verifies or
creates the extraction cache, then compiles a supplied Keras model with sparse
class indices and loads files from that cache. It saves `history.json`,
loss/accuracy plots, and a best checkpoint. The ZIP-backed loader remains
available for compatibility. For example:

```text
python -m src.training.cli ARCHIVE outputs/data_split_manifest.csv --classes 39
python -m src.evaluation.cli ARCHIVE outputs/data_split_manifest.csv outputs/training/best_model.keras
```

The file training loader uses a bounded shuffle buffer (512 examples by
default), seed 42, 224x224 RGB preprocessing, train-only augmentation, and
deterministic evaluation. Extract and verify explicitly with:

```text
python -m src.data.extract ARCHIVE --manifest outputs/data_split_manifest.csv
```

The extracted file loader uses TensorFlow-native file reads, image decoding,
resize, parallel mapping, and prefetching; Pillow remains the API used for
single-image and ZIP compatibility preprocessing. A bounded CPU benchmark
(20--50 batches, and never a full epoch) measures preprocessing, batch
delivery, and Custom CNN forward/backward cost:

```text
python -m src.profiling data/extracted outputs/data_split_manifest.csv \
  --batches 20 --batch-size 16 --threads 1 \
  --output outputs/profiling_benchmark.json
```

The benchmark does not start EfficientNet or full training.

Validate loading without starting training with:

```text
python -m src.training.smoke_test ARCHIVE outputs/data_split_manifest.csv --split validation
```

Both commands are explicit (nothing runs on import), use CPU-safe defaults,
and never download weights (`weights=None`). Evaluation writes accuracy,
macro precision/recall/F1, a classification report, and a confusion matrix to
`outputs/evaluation/metrics.json`. The manifest's `archive_path`, `label`, and
`class_index` are used consistently; evaluation can continue to use the
read-only ZIP adapter when an extracted cache is not needed.

Reusable single-image inference is available in `src/inference.py`:
`predict_image("model.keras", "class_mapping.json", "image.jpg")` loads a
saved model and one authoritative JSON/CSV class mapping, then returns class,
confidence, and top-k metadata. `src/explainability/gradcam.py` discovers the
final convolution layer (including nested EfficientNetB0-like backbones) and
saves overlays under `outputs/evaluation/gradcam`. These utilities are
explicit and do not train, download weights, or build Streamlit.

## Streamlit application

The optional Streamlit UI uses the same `src.inference.predict_image` path and
does not train a model or invent predictions. Install the requirements and
launch it from the repository root with:

```text
streamlit run app/app.py
```

By default it expects `outputs/training/custom_cnn_baseline/best_model.keras`
and the generated
`outputs/data_split_manifest.csv` (which contains the class mapping columns).
Configure these paths in the sidebar, or set
`AML_MODEL_PATH` and `AML_CLASS_MAPPING_PATH` before launching. The class names
come exclusively from the configured JSON/CSV/TSV mapping. A missing or
incompatible model, mapping, TensorFlow installation, or image is reported as
an actionable error in the UI.

The official EfficientNetB0 ImageNet file
`efficientnetb0_backbone_notop.h5` was not available locally, and the official
Keras URL returned HTTP 403. Any fallback run is labelled
`EfficientNetB0_from_scratch`, not transfer learning. The app therefore
defaults to the valid Custom CNN checkpoint.

The final measured comparison and limitations are in
`outputs/comparison/model_comparison.json`,
`outputs/evaluation/error_analysis.md`, and
`reports/research_report.md`. Grad-CAM was attempted for both saved models but
the existing implementation was incompatible with their reloaded functional
graphs, so no visualization is claimed. The original ImageNet transfer-learning
research question remains unresolved.
