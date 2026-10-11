# Implementation Plan

## Project

**Plant Disease Detection Using Transfer Learning**  
Coimbatore Institute of Technology, III M.Sc. AI & ML, V Semester  
Advanced Machine Learning - 19MAM51

## Current gate

Phase 2 has verified the supplied unaugmented ZIP layout: labels are encoded
by 39 class directories and can be read without extracting the archive.
Training remains out of scope until later phases and must use the generated
metadata split without copying the source images.

## Dataset Acquisition Requirement

Supervised training must wait until a verified labeled dataset is available. Before implementation proceeds, obtain the authoritative dataset source/citation and a label mapping for every image, preferably through class-name directories or a documented CSV/JSON manifest containing image filenames and class names. Verify that the mapping covers the complete image collection, then validate files and create a reproducible stratified split. Do not infer disease labels from image appearance, create sample labels, or claim PlantVillage provenance without source metadata.

## Reproducibility and CPU constraints

- Use a fixed random seed for Python, NumPy, and the selected framework.
- Run on CPU only; do not require CUDA or NVIDIA packages.
- Use a small/moderate batch size, initially 16 or 32 depending on memory.
- Use image inputs around 224 × 224 and avoid unnecessarily large architectures.
- Keep initial experiments short and use callbacks.
- Record package versions, hardware, split seed, hyperparameters, and excluded files.
- Do not copy the original dataset into the repository. Store only configuration, code, reports, and generated artifacts.

## Phase 1: data verification and preparation

1. Obtain the authoritative class labels and source citation.
2. Validate every image, record corrupt/unreadable paths, and confirm dimensions/formats.
3. Convert RGBA to RGB consistently.
4. Create a reproducible stratified train/validation/test split, preferably 70/15/15 unless an official split is supplied.
5. Save split manifests outside the original dataset or in a small repository-safe metadata file.
6. Produce exploratory summaries: class counts, sample images, dimensions, and imbalance.

## Phase 2: baseline model

Implement a small custom CNN with:

- Rescaling/normalization
- Two or three convolutional blocks
- Batch normalization or dropout where useful
- Global average pooling or a compact dense head
- Softmax classification output

Train with early stopping and learning-rate reduction. Record training time, parameter count, accuracy, precision, recall, F1-score, and confusion matrix results.

## Phase 3: advanced AML model

Use **EfficientNetB0** with transfer learning:

1. Load ImageNet-pretrained convolutional weights without the classification head.
2. Freeze the backbone and train a small regularized classification head.
3. Unfreeze only the upper backbone layers for controlled fine-tuning.
4. Use a lower fine-tuning learning rate than the head-training rate.
5. Use early stopping, best-checkpoint restoration, and a learning-rate scheduler.

If pretrained weights cannot be obtained in the execution environment, document that limitation rather than silently substituting a different experiment.

## Phase 4: practical hyperparameter tuning

Use a small, CPU-feasible search over a few choices, such as:

- Batch size: 16 or 32
- Learning rate: two or three carefully chosen values
- Dropout: two values
- Augmentation strength: conservative versus moderate

Limit trials and epochs, use validation performance for selection, and keep the test set isolated.

## Phase 5: evaluation and error analysis

For both models, report:

- Accuracy
- Macro and weighted precision
- Macro and weighted recall
- Macro and weighted F1-score
- Per-class metrics
- Confusion matrix
- Training curves
- Parameter count and approximate CPU training time

Inspect representative false positives and false negatives, especially confused disease pairs and visually difficult examples. Keep error-analysis samples tied to their original paths and labels.

## Phase 6: Grad-CAM explainability

Generate Grad-CAM visualizations for correctly and incorrectly classified test images using the final convolutional layer of EfficientNetB0. Explain whether highlighted regions correspond to plausible leaf symptoms and record cases where attention is on background or artifacts.

## Expected deliverables

- Reproducible preprocessing and split code
- Baseline CNN and EfficientNetB0 training code
- Saved model checkpoints outside version control where appropriate
- EDA, training, evaluation, prediction, and Grad-CAM outputs
- Comparison table and discussion of the baseline versus transfer-learning model
- Dataset citation, limitations, and CPU execution notes
