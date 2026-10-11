# Dataset Report

## Inspection status

This report contains only values verified by inspection on 2026-09-22. The original files were read in place and were not copied, modified, or deleted.

## Dataset Verification Status

**Status: VERIFIED LABELED IMAGE ARCHIVES; SUITABLE FOR SUPERVISED MULTICLASS CLASSIFICATION.**

The specified dataset directory contains two ZIP archives:

- `Plant_leaf_diseases_dataset_without_augmentation.zip`
- `Plant_leaf_diseases_dataset_with_augmentation.zip`

Each archive has one dataset root directory followed by 39 class-name directories. The class directory name is the label; no separate CSV, JSON, or metadata mapping is required for the archived layout. The archives were inspected without extraction or modification.

The unaugmented archive contains 55,448 images. The augmented archive contains 61,486 images. These are alternative dataset variants and must not be combined indiscriminately: augmented images derived from a source image must not cross train/validation/test boundaries, or evaluation leakage may occur.

## Source and detected layout

- Inspected path: `C:\Users\senso\Downloads\Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network`
- Top level: two ZIP files listed above; no extracted train/validation/test directories are present at this path.
- Within each ZIP: one root folder, then class folders, then image files.
- Example layout: `Plant_leave_diseases_dataset_without_augmentation\Apple___Apple_scab\image (1).JPG`
- Labels are represented by the class directory names.
- No official train/validation/test directories were found inside either archive.
- The archive contents establish a labeled plant-leaf dataset, but the inspected files do not independently establish the original PlantVillage citation/provenance.

## Source and Citation

The project materials do not contain an accessible paper, DOI, dataset README, or authoritative source record for this collection. Therefore no formal bibliographic citation is asserted. The verified source description is: *Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network*, local archive directory containing the unaugmented (55,448 images) and augmented (61,486 images) variants. Original authors, publication, DOI, and source repository are not stated in the available materials. PlantVillage provenance is not verified.

## Classes and distribution

Both archives contain the same 39 class names. Counts are shown as `without augmentation / with augmentation`:

| Class | Images |
|---|---:|
| Apple___Apple_scab | 630 / 1,000 |
| Apple___Black_rot | 621 / 1,000 |
| Apple___Cedar_apple_rust | 275 / 1,000 |
| Apple___healthy | 1,645 / 1,645 |
| Background_without_leaves | 1,143 / 1,143 |
| Blueberry___healthy | 1,502 / 1,502 |
| Cherry___Powdery_mildew | 1,052 / 1,052 |
| Cherry___healthy | 854 / 1,000 |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 513 / 1,000 |
| Corn___Common_rust | 1,192 / 1,192 |
| Corn___Northern_Leaf_Blight | 985 / 1,000 |
| Corn___healthy | 1,162 / 1,162 |
| Grape___Black_rot | 1,180 / 1,180 |
| Grape___Esca_(Black_Measles) | 1,383 / 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 / 1,076 |
| Grape___healthy | 423 / 1,000 |
| Orange___Haunglongbing_(Citrus_greening) | 5,507 / 5,507 |
| Peach___Bacterial_spot | 2,297 / 2,297 |
| Peach___healthy | 360 / 1,000 |
| Pepper,_bell___Bacterial_spot | 997 / 1,000 |
| Pepper,_bell___healthy | 1,478 / 1,478 |
| Potato___Early_blight | 1,000 / 1,000 |
| Potato___Late_blight | 1,000 / 1,000 |
| Potato___healthy | 152 / 1,000 |
| Raspberry___healthy | 371 / 1,000 |
| Soybean___healthy | 5,090 / 5,090 |
| Squash___Powdery_mildew | 1,835 / 1,835 |
| Strawberry___Leaf_scorch | 1,109 / 1,109 |
| Strawberry___healthy | 456 / 1,000 |
| Tomato___Bacterial_spot | 2,127 / 2,127 |
| Tomato___Early_blight | 1,000 / 1,000 |
| Tomato___Late_blight | 1,909 / 1,909 |
| Tomato___Leaf_Mold | 952 / 1,000 |
| Tomato___Septoria_leaf_spot | 1,771 / 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 / 1,676 |
| Tomato___Target_Spot | 1,404 / 1,404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 / 5,357 |
| Tomato___Tomato_mosaic_virus | 373 / 1,000 |
| Tomato___healthy | 1,591 / 1,591 |

The unaugmented variant is strongly imbalanced (152 to 5,507 images per class). The augmented variant is less sparse for several classes but remains imbalanced because some classes were not increased to 1,000.

## Image formats and dimensions

- Unaugmented archive: 55,446 `.jpg`, 1 `.png`, and 1 `.jpeg` by filename extension.
- Augmented archive: 61,484 `.jpg`, 1 `.png`, and 1 `.jpeg` by filename extension.
- A representative sample of 30 images from each archive decoded as JPEG, RGB, 256 × 256 pixels.
- Complete archive-level readability scan: 0 unreadable files in the unaugmented archive (55,448 checked) and 0 unreadable files in the augmented archive (61,486 checked).

## Missing/corrupt image information

- No missing class directory was observed in the ZIP manifests.
- Representative scan: 0 unreadable images out of 30 sampled per archive.
- Complete corruption status: no corrupted/unreadable images detected in either archive.

## Recommended preprocessing

These are recommendations, not measured dataset results:

1. Obtain or verify class labels before creating a supervised dataset.
2. Convert RGBA images to RGB consistently, documenting how the alpha channel is handled.
3. Resize to the selected model input size (for example, 224 × 224 for EfficientNetB0).
4. Normalize using the preprocessing function expected by the selected backbone.
5. Apply training-only augmentation such as small rotations, translations, horizontal flips, and moderate zoom/contrast changes.
6. Check all files for decoding errors before training and record the excluded paths.
7. Recompute class counts after labels are available.

## Recommended split strategy

After labels are available, use a reproducible, stratified split, for example:

- 70% training
- 15% validation
- 15% test

Keep the test set untouched until final evaluation. Prevent near-duplicate or augmented versions of the same source image from crossing splits. If the verified source already supplies an official split, preserve it and document the exact split instead.

## Suitability assessment

- Suitable for supervised multiclass plant disease classification: **Yes**, because labels are encoded by 39 class-name directories.
- Suitable for the planned small Custom CNN baseline: **Yes**.
- Suitable for EfficientNetB0 transfer learning and fine-tuning: **Yes**, subject to CPU-feasible batch sizes/epochs and preprocessing.
- Recommended experiment source: use one archive variant only, preferably the unaugmented archive with augmentation applied only to training data. Keep validation and test images free of generated counterparts from training.
- Create a reproducible stratified split after checking for duplicate/source relationships. Exclude or explicitly justify `Background_without_leaves` depending on the project question.
- PlantVillage provenance: **not independently verified from the archive filenames alone**.
