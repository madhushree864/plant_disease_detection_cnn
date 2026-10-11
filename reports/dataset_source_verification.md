# Dataset Source Verification

## 1. Paper Information

### Paper search scope

The project repository, the candidate dataset directory, and the checked Desktop/Downloads research materials were initially inspected without finding the relevant paper. A subsequent inspection of `C:\Users\senso\Downloads\IEEE Papers` found three relevant IEEE PDFs. The 2025 paper explicitly cites the requested dataset record as Mendeley Data V1, DOI `10.17632/tywbtsjrjv.1`, but the dataset record and the nine-layer paper itself are not present among the accessible PDFs.

### Exact paper title

The exact dataset citation identified in the accessible IEEE PDF is **“Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network.”** This is a Mendeley Data record, not the title of an IEEE paper available in the checked folder.

### Authors

J. A. Pandian and G. Geetharamani are stated as the dataset-record authors in the 2025 IEEE paper's reference list.

### IEEE conference/journal name

Mendeley Data, V1, 2019, DOI `10.17632/tywbtsjrjv.1`, as stated in the 2025 IEEE paper's reference list.

### Publication year

10.17632/tywbtsjrjv.1.

### DOI

Not stated in the accessible project/research materials.

Because the relevant paper was not accessible, no paper-specific bibliographic facts have been inferred from general web knowledge or from the dataset directory name.

## 2. Dataset Information Reported in the Paper

The dataset record itself could not be inspected. The accessible 2025 IEEE paper reports:

- Dataset name: PlantVillage.
- Dataset source/citation: Mendeley Data V1, DOI `10.17632/tywbtsjrjv.1`.
- Number of classes/species: 38 species of plant diseases.
- Number of images: 54,303.
- Augmentation: rotations, shifts, shear, zoom, horizontal flipping, and brightness adjustment are described in the IEEE paper.
- Exact relationship between those figures and the local ZIP manifests: Not stated in the accessible paper.

The comparison is therefore a **probable match but not completely verified**, not an exact match.

## 3. Local Dataset Information

The verified local candidate is:

`C:\Users\senso\Downloads\Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network`

It contains two ZIP archives:

- `Plant_leaf_diseases_dataset_without_augmentation.zip`
- `Plant_leaf_diseases_dataset_with_augmentation.zip`

Verified local facts:

- The unaugmented archive contains **55,448 images**.
- The augmented archive contains **61,486 images**.
- Each archive contains **39 class directories**.
- Directory names are the class labels.
- Images are reported as 256 × 256 RGB images based on the prior archive inspection.
- The complete archive-level readability scan found **0 unreadable files** in each archive.
- No official train/validation/test directories were found inside either archive.
- The inspected dataset folder contains the two archives but no paper, README, citation, or source metadata.

## 4. Paper vs Local Dataset Comparison

Because no relevant paper was accessible, a paper-to-local numerical comparison is not possible.

| Item | Paper | Verified local dataset | Comparison status |
|---|---|---|---|
| Paper title and citation | Not available | Directory name only | Cannot compare |
| Dataset name | PlantVillage; cited Mendeley Data V1 record | Archive names identify unaugmented and augmented variants | Probable provenance relationship |
| Dataset source | Mendeley Data V1, DOI `10.17632/tywbtsjrjv.1` | No embedded citation metadata | Citation identified from paper |
| Number of classes | 38 species of plant diseases | 39 directory-based classes | Not an exact count match |
| Number of images | 54,303 | 55,448 unaugmented; 61,486 augmented | Not an exact count match |
| Image format/size | Not stated in the accessible paper | 256 × 256 RGB; extension mix documented in `dataset_report.md` | Cannot compare |
| Preprocessing/augmentation | Augmentation transformations explicitly listed in the paper | Augmented archive exists; exact author method is not documented locally | Compatible in concept; exact correspondence unresolved |

### 55,448-image archive

The 55,448-image archive is a **probable but not exact match** to the dataset cited by the 2025 IEEE paper: the paper cites the same nine-layer dataset title/Mendeley record and identifies PlantVillage, but reports 54,303 images and 38 species rather than the local 55,448 images and 39 classes.

### 61,486-image archive

The 61,486-image archive has the same 39 class names and is explicitly named as the “with augmentation” variant of the local unaugmented archive. This supports treating it as a variant of the same local collection. The 2025 IEEE paper describes augmentation, but does not confirm this exact archive or explain the local count/class discrepancy.

## 5. Dataset Provenance

The only provenance established by the available materials is the local directory and its archive contents. The local files do **not** independently establish:

 - the exact relationship to the cited Mendeley manifest;
 - the reason for the difference between the paper's 54,303/38 figures and the local 55,448/39 figures; or
 - the exact augmentation procedure used to create the local augmented archive.

The accessible 2025 IEEE paper does provide a provenance lead: the Mendeley Data record cited above, and it explicitly calls the dataset PlantVillage.

The archive contents do establish directory-based labels, 39 classes, the two image totals, and the absence of unreadable files in the verified scan.

## 6. Final Verification Status

**Status: Local dataset structure verified; probable paper/source match identified, exact archive correspondence unresolved.**

The candidate is suitable for continued dataset inspection as a labeled 39-class image collection. The cited Mendeley record is the correct dataset provenance lead, but the local archive should be described as a probable rather than exact match until its manifest/count discrepancy is resolved. PlantVillage may be stated only with the qualification that the 2025 IEEE paper explicitly identifies the cited dataset as PlantVillage.

No model was trained, the original dataset archives were not modified, and no Streamlit application was created.

## 7. Correct Citation for Our Project

The dataset citation explicitly stated in the accessible IEEE paper is:

> J. A. Pandian and G. Geetharamani, “Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network,” *Mendeley Data*, V1, 2019, DOI: 10.17632/tywbtsjrjv.1.

The local ZIP filenames and counts should be reported separately as verified local observations, with the unresolved 54,303/38 versus 55,448/39 discrepancy clearly noted.
