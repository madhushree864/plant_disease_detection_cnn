# Next Steps

## Current dataset status

The candidate directory is:

`C:\Users\senso\Downloads\kaggle\working`

It contains 30,000 flat PNG images named `train1.png` through `train30000.png`. The images have no class folders or associated label metadata in the dataset area. A focused search of Downloads found no dataset-associated CSV, JSON, TXT, Excel, README, Kaggle metadata, or PlantVillage-related file. PlantVillage provenance is therefore not verified.

## What is missing

The current collection is not usable for supervised plant-disease classification as-is. The following are missing:

- A verified dataset source and citation
- Class names
- A mapping from every image filename to its class
- A documented train/validation/test split, or enough labels to create one reproducibly
- Confirmed PlantVillage metadata if PlantVillage is required for the assignment

Labels must not be inferred from image appearance.

## Exact files or metadata needed

Obtain one of the following, without modifying the current 30,000-image directory:

1. A directory structure such as `class_name\image.png`, with every image placed under its verified class folder; or
2. A CSV/JSON manifest containing at least:
   - image filename or relative path
   - class name or class ID
   - a documented class-ID-to-class-name mapping
   - optional official split (`train`, `validation`, or `test`)

Also retain the official dataset source URL/citation and any README or metadata file that establishes provenance.

## Recommended next action tomorrow

Locate the original download/archive or obtain the approved labeled PlantVillage dataset manually. Do not begin training until the labels and source can be verified. Once available, compare the manifest against the 30,000 filenames, report missing/extra entries, validate all images, and then create a reproducible stratified split.

## Planned model architecture

- **Baseline:** small custom CNN with normalization, two or three convolutional blocks, pooling, dropout or batch normalization, and a softmax output layer.
- **Advanced AML model:** EfficientNetB0 with ImageNet transfer learning, a new classification head, frozen-backbone training followed by controlled upper-layer fine-tuning.
- CPU constraints: small/moderate batches, limited epochs, early stopping, checkpoint restoration, and learning-rate scheduling.

## Planned evaluation metrics

For both models, report:

- Accuracy
- Per-class, macro, and weighted precision
- Per-class, macro, and weighted recall
- Per-class, macro, and weighted F1-score
- Confusion matrix
- Training/validation curves
- Error analysis of false positives and false negatives
- Grad-CAM explanations for EfficientNetB0

No model training or application development is planned until dataset verification is complete.
