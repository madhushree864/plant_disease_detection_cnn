# Project Status

## Phase 2: archive inspection and reproducible split

Implemented read-only data utilities under `src/data` for the unaugmented ZIP
archive. They:

- discover labels from image-containing directories (the supplied archive has
  39 labels);
- validate image readability directly from ZIP streams without extracting or
  modifying the archive; and
- create a deterministic, per-class 70/15/15 split using seed `42` and a
  stable alphabetical class-index mapping.

`src.data.build_manifest` writes a CSV manifest to
`outputs/data_split_manifest.csv` by default. The manifest contains archive
member paths and labels only; image files are never copied. The CLI refuses to
write a manifest if any selected image is unreadable.

Quick tests cover class discovery, readability, split reproducibility,
split leakage, and class mapping consistency. Model training and application
work are intentionally out of scope for this phase.

## Phases 3-4: bounded EDA and preprocessing

`src/eda.py` computes manifest/class counts and bounded, actual image
dimension/channel summaries directly from the unaugmented ZIP, and saves plots
to `outputs/eda`. `src/preprocessing.py` provides lazy ZIP-member decoding to
224x224 RGB float32 arrays, with train-only seeded/random horizontal flips and
deterministic evaluation preprocessing. TensorFlow is imported only when the
optional `make_tf_dataset` adapter is used. No training or full-dataset scan
is performed by default.

## Phase 5: model definitions

`src/models` provides a small from-scratch CNN, an EfficientNetB0 transfer-learning factory, a helper that unfreezes only the upper backbone layers, and factories for standard training callbacks. EfficientNet weights default to `None`, so constructing a model does not download weights.

## Phase 6: reusable training and evaluation

`src/training` provides an explicit, CPU-safe `TrainingConfig`, manifest
validation, and `train_model()` orchestration for a supplied model. Training
uses the existing lazy ZIP-backed preprocessing adapter and saves a best
checkpoint, JSON history, and lightweight loss/accuracy plots. `src/evaluation`
provides pure metric helpers plus held-out model evaluation, writing accuracy,
macro precision/recall/F1, a classification report, and confusion matrix.
CLI entry points are available with `python -m src.training.cli` and
`python -m src.evaluation.cli`; they never run automatically and do not
download weights. Tests use tiny in-memory metric inputs; no dataset scan is
performed.

## Phase 7: reusable inference and Grad-CAM

`src/inference.py` loads a saved Keras model plus one authoritative JSON/CSV
class mapping, applies shared 224x224 RGB preprocessing, and returns
predicted class, confidence, and top-k metadata for one image. Missing model
or mapping files fail explicitly. `src/explainability/gradcam.py` locates the
deepest suitable convolution layer, including nested EfficientNetB0-like
backbones, and saves supplied-image overlays to
`outputs/evaluation/gradcam`. These modules are opt-in: they do not train, download weights, or run
Streamlit.

## Phase 8: Streamlit application

`app/app.py` provides an upload-based Streamlit interface over
`src.inference.predict_image`. It displays the uploaded image, predicted class,
confidence, and the available top-three predictions. Model and class-mapping
paths are configurable through the sidebar or the `AML_MODEL_PATH` and
`AML_CLASS_MAPPING_PATH` environment variables; class names are never
hardcoded. Inference results are cached by model path, mapping path, and image
bytes. The UI reports missing or incompatible artifacts clearly and never
trains a model or fabricates output. The app requires the optional `streamlit`
dependency and is launched with `streamlit run app/app.py`; it was not started
as part of repository validation.

## Phase 9: bounded streaming training loader

The compatibility ZIP-backed `tf.data` loader uses a configurable 512-example
shuffle buffer (seed 42), opens the archive once per generator iteration, and
keeps validation/test deterministic with no augmentation. The smoke utility
`python -m src.training.smoke_test ARCHIVE MANIFEST` now exercises the verified
file-backed path and reports image and label shapes without starting a model.

## Phase 10: experimental training attempt

### Dataset manifest

- Source: unaugmented archive only.
- Manifest: `outputs/data_split_manifest.csv`
- Verified rows: 55,448
- Verified classes: 39
- Verified readable images: 55,448
- Split seed: 42
- Split: deterministic 70/15/15 per class.

### Custom CNN baseline

- Status: **stopped before the first epoch completed**.
- Configuration: existing Custom CNN, 224x224 input, batch size 16, one CPU
  thread, maximum 10 epochs, and existing callbacks.
- No checkpoint, history, plot, or evaluation result was produced.
- Actual bottleneck: the previous lazy ZIP-backed training pipeline filled a
  shuffle buffer of 38,814 training records before processing began. On the CPU-only
  machine this remained in buffer-filling after more than two minutes, with no
  completed epoch and no saved artifact.
- No EfficientNetB0 training, fine-tuning, evaluation, error analysis, or
  Grad-CAM experiment has started.

No experimental metric is reported because no model completed an epoch or
produced a valid checkpoint.

## Phase 13: Custom CNN baseline experiment

- Dataset: verified unaugmented archive only.
- Manifest: `outputs/data_split_manifest.csv` (unchanged; SHA-256
  `3810761BB92CF7E2C288EC4A53310797BCB2C63474813E5F27E79749D6925D84`)
- Split: 38,814 train / 8,318 validation / 8,316 test.
- Model: existing `custom_cnn`, unchanged.
- Configuration: seed 42, 224x224 RGB, batch size 16, shuffle buffer 512,
  CPU threads 4, maximum 3 epochs, existing Adam/sparse-cross-entropy
  configuration, train-only augmentation, early stopping, ReduceLROnPlateau,
  and ModelCheckpoint.
- Status: **completed 3 epochs successfully**.
- Epoch durations from the actual run: 2,204 s, 2,269 s, and 2,209 s.
- Total training time: approximately 6,682 s (111.4 minutes).
- Best validation epoch: **epoch 1**, based on lowest validation loss.

### Actual training history

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.496807 | 0.618231 | 7.652438 | 0.028613 |
| 2 | 0.964688 | 0.690086 | 9.899536 | 0.031137 |
| 3 | 0.823050 | 0.719199 | 11.178065 | 0.031137 |

The best checkpoint restored by the callback is from epoch 1. Validation loss
increased after epoch 1 despite improving training accuracy.

### Test evaluation

The saved best checkpoint was evaluated once on the held-out test split only.

- Accuracy: **0.028740**
- Macro precision: **0.000737**
- Macro recall: **0.025641**
- Macro F1: **0.001433**

Artifacts:

- Checkpoint: `outputs/training/custom_cnn_baseline/best_model.keras`
- History: `outputs/training/custom_cnn_baseline/history.json`
- Accuracy plot: `outputs/training/custom_cnn_baseline/accuracy.png`
- Loss plot: `outputs/training/custom_cnn_baseline/loss.png`
- Test metrics/classification report: `outputs/evaluation/custom_cnn_baseline/metrics.json`
- Test confusion matrix: `outputs/evaluation/custom_cnn_baseline/confusion_matrix.png`

No EfficientNetB0 training or fine-tuning has started. No model comparison,
error analysis, or Grad-CAM experiment has been run yet.

## Phase 14: bounded Custom CNN baseline diagnosis

The completed baseline was inspected without retraining and without changing
the manifest, split, or extracted data. `src/diagnostics.py` is a read-only
utility; it checks the manifest/index contract, extracted paths and cache
marker, bounded samples, split counts, preprocessing behavior, and bounded
checkpoint predictions. It does not construct or start EfficientNet.

Verified findings:

- 55,448 manifest rows and 39 classes match the expected dataset contract;
  split counts are train 38,814, validation 8,318, and test 8,316.
- Every manifest path (55,448/55,448) resolves to an extracted file. The
  extraction marker reports 55,448 members and 39 classes. The original ZIP is
  not present in this working tree, so source-hash revalidation could not be
  performed; no cache files were modified.
- Every manifest label has the stable alphabetical `class_index` used by the
  split code. Bounded train/validation/test records include path, label,
  index, and existence status.
- Evaluation preprocessing is byte-for-byte deterministic on a sampled image.
  Horizontal flip augmentation is enabled only for training; validation/test
  loaders use `training=False`, `shuffle=False`, and deterministic options.
- On bounded 128-record validation and test batches, the saved checkpoint
  predicted class 38 for all 128 records in each split (0.000 bounded
  accuracy; dominant actual classes were 0 and 1). This is consistent with
  the existing full-test metrics (2.874% accuracy and near-zero macro F1):
  the collapse is a genuine model-performance failure, not a label/path
  mapping or evaluation-augmentation bug.
- The training orchestration supplies finite datasets without explicit
  `steps_per_epoch`; partial final batches are retained. An
  “input ran out” warning is therefore not expected from this code path.
  The earlier observed stall/slow progress is attributable to image
  decoding and delivery on CPU, rather than exhausted input.

Diagnostic command and output:

```text
python -m src.diagnostics
rows=55448 classes=39 mapping_ok=True
split_counts={'test': 8316, 'train': 38814, 'validation': 8318}
extracted_paths={'checked': 55448, 'missing': 0, 'all_exist': True}
evaluation_deterministic=True
validation: records=128 accuracy=0.0 predicted={38: 128} unique=[38]
test: records=128 accuracy=0.0 predicted={38: 128} unique=[38]
```

### Bounded-loader retry

- Smoke test: passed for train and validation splits; batches were produced
  immediately with shape `(4, 224, 224, 3)`, integer labels, and a bounded
  shuffle buffer of 512.
- Custom CNN retry: stopped after approximately 120 seconds before epoch 1
  completed.
- Progress observed: approximately 70 of 2,425 batches in epoch 1.
- No checkpoint, history, plot, or evaluation result was produced.
- The previous full-buffer stall was fixed, but CPU image decoding/resizing and
  ZIP-backed sample delivery remain too slow for a practical full epoch under
  the current implementation.
- EfficientNetB0 was not started.

## Phase 10: extracted file-cache training retry

- Cache location: `data/extracted`
- Source: unaugmented ZIP only.
- Cache verification: passed against source SHA-256, manifest paths, 55,448
  images, and 39 classes.
- Manifest SHA-256 before training:
  `3810761BB92CF7E2C288EC4A53310797BCB2C63474813E5F27E79749D6925D84`
- Real-data train smoke test: passed with batches shaped
  `(4, 224, 224, 3)` and integer labels.
- Real-data validation smoke test: passed with the same shape and deterministic
  preprocessing.
- Custom CNN retry: stopped before epoch 1 completed.
- Configuration was unchanged: 224x224, batch size 16, shuffle buffer 512,
  seed 42, one CPU thread, maximum 10 epochs, existing callbacks.
- Measured progress: 4 batches in approximately 11 seconds after startup.
  The epoch contains approximately 2,425 batches, so this remains impractical
  on the available CPU even with local files.
- No checkpoint, history, plots, or evaluation metrics were produced.
- EfficientNetB0 was not started.

## Phase 11: verified file extraction cache

`src.data.extraction` provides a reusable cache at `data/extracted` for only
the unaugmented ZIP. It verifies the source SHA-256, preserves archive
class-directory paths, and refuses incomplete caches. With the authoritative
manifest it verifies 55,448 image rows and 39 classes; the ZIP is never
modified. `python -m src.data.extract ARCHIVE --manifest MANIFEST` is the
explicit extraction utility. Training now uses the verified file-backed
loader while the ZIP loader remains available for compatibility. Both loaders
preserve manifest labels and class indices, use seed 42 and a bounded 512
example shuffle buffer, preprocess 224x224 RGB images, augment only training,
and keep evaluation deterministic.

## Phase 12: bounded CPU profiling and safe loader optimization

The file-backed loader was profiled using the existing extracted, unaugmented
data and manifest, 224x224 inputs, batch size 16, seed 42, and one CPU thread.
The controlled run used exactly 20 batches (320 samples) and did not start an
epoch or EfficientNet:

| Stage | Wall time | Per unit |
| --- | ---: | ---: |
| File read/decode/resize preprocessing (320 images) | 1.273 s | 3.98 ms/image |
| Dataset batch delivery (20 batches) | 0.619 s | 30.96 ms/batch |
| Custom CNN `train_on_batch` (forward/backward, 20 batches) | 32.268 s | 1.613 s/batch |

The measured bottleneck is Custom CNN CPU forward/backward compute (about 94%
of the measured stage time), not file I/O or dataset delivery. Based on the
measurement, the extracted loader was safely optimized to use TensorFlow-native
`read_file`/decode/resize, parallel `map`, prefetch, and a bounded shuffle
buffer. Train-only flipping is preserved and evaluation remains deterministic.
The optimization cannot remove the dominant model compute cost.

Benchmark artifact: `outputs/profiling_benchmark.json`. With approximately
2,425 training batches, the measured compute rate implies roughly 65 minutes
for one epoch before validation and callback overhead. Therefore a full
baseline run is **not recommended as a practical CPU run**; no full epoch,
10-epoch training, or EfficientNet run was started.
