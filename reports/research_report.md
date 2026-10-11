# Plant Disease Detection Using Transfer Learning

## Abstract

This project evaluates whether transfer learning improves 39-class plant
disease classification over a CNN trained from scratch. The verified local
dataset contains 55,448 unaugmented RGB images with a deterministic 70/15/15
split. The Custom CNN completed its prescribed experiment. The intended
EfficientNetB0 ImageNet experiment was blocked because the official Keras
weight URL returned HTTP 403 and no local official file was available. Any
fallback EfficientNet run is reported as from-scratch initialization, not
transfer learning.

## Problem statement and research question

Plant disease recognition is a multiclass image classification problem. The
research question is: “Does transfer learning provide a meaningful improvement
over a CNN trained from scratch for multiclass plant disease classification?”

## Dataset and preprocessing

Only the unaugmented archive is used. It contains 55,448 readable images in
39 classes. The authoritative manifest is
`outputs/data_split_manifest.csv`, with 38,814 training, 8,318 validation, and
8,316 test records, generated deterministically with seed 42. Images are read
as RGB, resized to 224x224, and normalized to float32 [0, 1]. Augmentation is
train-only; validation and test preprocessing is deterministic. Provenance is
qualified in `reports/dataset_source_verification.md`.

## Models and configuration

The baseline Custom CNN has three convolutional blocks (32, 64, and 128
filters), pooling, global average pooling, dropout, and a 39-unit softmax
head. It was trained from scratch for three epochs with batch size 16, four
CPU threads, seed 42, shuffle buffer 512, and the existing callbacks.

The intended advanced model is EfficientNetB0 with ImageNet weights, a frozen
backbone, and a new classification head. Keras expected
`efficientnetb0_backbone_notop.h5` with MD5
`50bc09e76180e00e4465e1a485ddc09d`; the official URL returned HTTP 403 and no
local official cache was found. Consequently, no random-initialized run is
called transfer learning.

## Results

Only measured values in saved artifacts are included:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Custom CNN from scratch | 0.028740 | 0.000737 | 0.025641 | 0.001433 |
| EfficientNetB0_from_scratch | 0.0966810967 | 0.0024790025 | 0.0256410256 | 0.0045209177 |

The baseline and EfficientNetB0_from_scratch reports and 39x39 confusion
matrices are in their respective `outputs/evaluation/` directories. The
machine-readable comparison is
`outputs/comparison/model_comparison.json`. These results compare two
from-scratch runs only; they do not demonstrate an effect of transfer
learning.

## Error analysis and explainability

The baseline diagnostic found no manifest, path, class-index, split, or
preprocessing mismatch. The saved Custom CNN checkpoint predicted one class
on bounded held-out samples. The EfficientNetB0_from_scratch confusion matrix
shows class 36 receiving all 8,316 test predictions. The complete report and
confusion matrices are preserved. Grad-CAM was attempted for both checkpoints,
but the existing implementation raised an unconnected-output error for the
reloaded Keras graphs; no Grad-CAM result is claimed.

## Application

The Streamlit application accepts an uploaded image, applies the shared
preprocessing, and displays the predicted class, confidence, and top three
predictions. It defaults to the existing Custom CNN checkpoint because no
ImageNet transfer-learning checkpoint exists. Import and syntax validation
passed without launching a server.

## Limitations, ethics, and future work

The ImageNet transfer-learning comparison could not be completed while the
official weights were inaccessible. CPU-only execution limits throughput.
Predictions are experimental and require expert verification before
agricultural use; domain shift and class imbalance must be considered.
Future work is to obtain and MD5-verify the official Keras weights through an
authorized channel, then run frozen transfer learning and controlled
fine-tuning without changing the manifest or preprocessing. The original
research question remains unresolved.

## Conclusion

EfficientNetB0_from_scratch achieved higher measured test accuracy and macro
F1 than the Custom CNN in these saved runs, but both models were initialized
from scratch. Therefore the project does not conclude that transfer learning
improved performance.

## Reproducibility

Use the existing unaugmented archive, manifest, verified `data/extracted`
cache, Python 3.13, TensorFlow 2.21, and the documented CPU settings. Launch
the application with `streamlit run app/app.py`; it defaults to the valid
Custom CNN checkpoint.
