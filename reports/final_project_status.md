# Final Project Status

## Completed

- Verified unaugmented dataset: 55,448 images and 39 classes.
- Preserved authoritative 70/15/15 manifest and deterministic seed 42 split.
- Completed Custom CNN from-scratch baseline without rerunning it.
- Completed `EfficientNetB0_from_scratch` using `weights=None`; it is not a
  transfer-learning result.
- Saved held-out test metrics, reports, and 39x39 confusion matrices.
- Created `outputs/comparison/model_comparison.json`.
- Created `outputs/evaluation/error_analysis.md`.
- Verified the Streamlit module imports and defaults to the existing Custom CNN
  checkpoint.
- Created the research report and this final status document.

## Exact measured metrics

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Custom CNN | 0.028740 | 0.000737 | 0.025641 | 0.001433 |
| EfficientNetB0_from_scratch | 0.0966810967 | 0.0024790025 | 0.0256410256 | 0.0045209177 |

## Blocked component

The intended EfficientNetB0 ImageNet transfer-learning experiment was not
completed. The official Keras file
`efficientnetb0_backbone_notop.h5` could not be downloaded because the
official URL returned HTTP 403, and no official local cache file was found.
No result in this project claims ImageNet transfer learning.

## Explainability and application

Grad-CAM was attempted using both saved checkpoints and the existing
implementation. Reloaded Keras graphs produced an unconnected-output error, so
no Grad-CAM images were fabricated or saved. Streamlit import validation
succeeded; the app uses the available Custom CNN checkpoint by default.

## Validation

Python compilation, the existing lightweight pytest suite, Streamlit import,
manifest integrity, model-file existence, and confusion-matrix dimensions were
validated without training.

No training was rerun and no fabricated metrics were used. The research
question about ImageNet transfer learning remains unresolved.
