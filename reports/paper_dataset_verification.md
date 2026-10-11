# Paper and Dataset Verification

## 1. Available IEEE Papers

Every PDF found in `C:\Users\senso\Downloads\IEEE Papers` was inspected locally:

1. `Hybrid_CNN_and_Transformer-Based_Sequential_Learning_Techniques_for_Plant_Disease_Classification.pdf`
2. `Improving_Plant_Disease_Classification_With_Deep-Learning-Based_Prediction_Model_Using_Explainable_Artificial_Intelligence.pdf`
3. `Plant_Disease_Identification_Using_a_Novel_Convolutional_Neural_Network.pdf`

All three are relevant to plant disease detection/classification and CNN-based methods. The first two also discuss transfer learning or advanced deep-learning models. The PDFs were the only source used; no external web sources were consulted.

## 2. Relevant Papers

### 2.1 Hybrid CNN and Transformer-Based Sequential Learning Techniques for Plant Disease Classification

- **Authors:** Anuruk Prommakhot, Jakkrit Onshaunjit, Wichian Ooppakaew, Grianggai Samseemoung, and Jakkree Srinonchat.
- **IEEE journal:** IEEE Access, Volume 13, 2025.
- **Publication information stated in the PDF:** Accepted 28 June 2025; publication 7 July 2025; current version 18 July 2025.
- **DOI:** 10.1109/ACCESS.2025.3586285.
- **Dataset:** PlantVillage.
- **Dataset source stated in the paper:** PlantVillage is described as an open-source dataset. Reference [33] identifies the cited dataset record as Mendeley Data.
- **Dataset URL:** No dataset URL is stated for reference [33] in this PDF.
- **Dataset citation stated in the PDF:** J. A. Pandian and G. Geetharamani, “Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network,” *Mendeley Data*, V1, 2019, DOI: 10.17632/tywbtsjrjv.1.
- **Number of images:** 54,303.
- **Number of classes/species:** The abstract says 38 species of plant diseases; the dataset section reports 54,303 images. The paper does not provide a complete machine-readable class list in the inspected text.
- **Class names explicitly mentioned:** Examples in the results include `Apple_Black Rot`, `Peach_Healthy`, `Tomato_Healthy`, `Cherry_Healthy`, `Corn_Gray Leaf Spot`, and `Raspberry_Healthy`. These are not a complete class list.
- **Models:** TwoConV-BiLSTM, TransConV-BiLSTM, TransNet-BiLSTM, TransNet-LSTM, TransNet-Seq2Seq, and TransNet-GRU. The proposed approach combines convolutional operators, transformer/sequential learning, LSTM/BiLSTM, Seq2Seq, and GRU components.
- **Preprocessing:** The PDF describes feature extraction/refinement through convolutional layers, pooling/global average pooling, transformer processing, and sequential layers. A separate complete image preprocessing specification is not stated in the inspected text.
- **Data augmentation:** Rotations of 20 degrees, width shift 0.2, height shift 0.2, shear 0.2, zoom 0.2, horizontal flipping at 50%, and brightness adjustment at 20%.
- **Train/test split:** 80% training and 20% testing.
- **Evaluation metrics:** Accuracy, precision, recall, loss, training time, AUC, and confusion matrix.
- **Main results:** TwoConV-BiLSTM is reported as the strongest model, with approximately 97.88% accuracy, 97.93% precision, 97.62% recall, and 0.0696 training loss. The conclusion also reports 97.36% accuracy on test data and a training time of 96.03 minutes for the best model. The PDF contains both 97.88% and 97.89% in different passages; both values are reported without reconciliation.

### 2.2 Improving Plant Disease Classification With Deep-Learning-Based Prediction Model Using Explainable Artificial Intelligence

- **Authors:** Natasha Nigar, Hafiz Muhammad Faisal, Muhammad Umer, Olukayode Oki, and Jose Manappattukunnel Lukose.
- **IEEE journal:** IEEE Access, Volume 12, 2024.
- **Publication information stated in the PDF:** Accepted 11 July 2024; publication 15 July 2024; current version 26 July 2024.
- **DOI:** 10.1109/ACCESS.2024.3428553.
- **Dataset:** “New Plant Diseases” dataset.
- **Dataset source:** Kaggle.
- **Dataset URL stated in the PDF:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Dataset citation stated in the PDF:** S. Bhattarai, “New Plant Diseases Dataset,” online Kaggle source.
- **Number of images:** 87,000.
- **Number of classes:** 38 distinct classes across 14 plant species.
- **Class names:** The paper discusses crop/disease categories and shows dataset tables/figures, but does not provide a complete class list in the extractable text used for this report.
- **Models:** EfficientNetB0 as the main model; four pre-trained models are mentioned for comparison, but their complete names are not all stated in the inspected methodology passage.
- **Preprocessing:** Background removal to isolate the region of interest; images separated into distinct classes based on crops.
- **Data augmentation:** No explicit augmentation procedure is stated in the inspected methodology.
- **Train/test split:** 80% training and 20% validation.
- **Evaluation metrics:** Accuracy, precision, recall, confusion matrix, and ANOVA analysis. Explanations are generated using LIME.
- **Main results:** 99.69% accuracy, 98.27% precision, and 98.26% recall for 38 diseases. The paper also describes a mobile application named PlantCare and TensorFlow Lite conversion.

### 2.3 Plant Disease Identification Using a Novel Convolutional Neural Network

- **Authors:** Sk Mahmudul Hassan and Arnab Kumar Maji.
- **IEEE journal:** IEEE Access, Volume 10, 2022.
- **Publication information stated in the PDF:** Accepted 30 December 2021; publication 7 January 2022; current version 14 January 2022.
- **DOI:** 10.1109/ACCESS.2022.3141371.
- **Datasets:** PlantVillage, a rice disease dataset, and a cassava plant disease dataset.
- **PlantVillage source:** The paper calls PlantVillage a large and available plant disease dataset and states that the selected PlantVillage images concern corn, potato, and tomato diseases. No PlantVillage URL is stated in this paper.
- **Dataset citation/reference:** The paper's reference [17] cites G. Geetharamani and P. J. Arun, “Identification of plant leaf diseases using a nine-layer deep convolutional neural network,” *Computers & Electrical Engineering*, Volume 76, pages 323–338, June 2019. This is a paper citation, not the complete dataset citation.
- **Number of images:** The paper explicitly states 5,932 rice images and 5,656 cassava images. A total PlantVillage image count is not stated in the extractable text; the PlantVillage data description is presented in a table.
- **Number of classes:** Rice has four categories: bacterial blight, blast, brown spot, and tungro. Cassava has five categories: healthy cassava and four diseased-leaf categories. For PlantVillage, the selected crops are corn, potato, and tomato; a complete class count is not stated in the extractable text.
- **Class names explicitly provided:** Rice: bacterial blight, blast, brown spot, and tungro. Cassava: healthy cassava, cassava bacterial blight, cassava brown streak, and other diseased categories described in the dataset table. PlantVillage: corn, potato, and tomato disease groups.
- **Models:** A novel lightweight CNN using Inception and residual connections, depthwise separable convolution, batch normalization, ReLU, pooling, global average pooling, and dropout. Comparisons include VGG16, VGG19, InceptionV3, ResNet50, and DenseNet201.
- **Preprocessing:** The paper discusses controlled/uniform PlantVillage backgrounds and field-condition rice/cassava images. The proposed architecture uses batch normalization, ReLU, pooling, global average pooling, and dropout.
- **Data augmentation:** No explicit image-augmentation procedure is stated.
- **Train/test split:** An 80%-20% training/validation split is stated. The paper also reports five-fold cross-validation.
- **Evaluation metrics:** Accuracy, precision, recall, F1-score, training/validation loss, confusion matrix, and training time.
- **Main results:** 99.39% accuracy on the selected PlantVillage data, 99.66% on rice, and 76.59% on imbalanced cassava data. The proposed model has 428,100 parameters and is presented as smaller than the compared networks.

## 3. Exact Dataset Information from Papers

The strongest direct provenance statement for the requested dataset is in reference [33] of the 2025 paper:

> J. A. Pandian and G. Geetharamani, “Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network,” *Mendeley Data*, V1, 2019, DOI: **10.17632/tywbtsjrjv.1**.

The same 2025 paper explicitly describes the experimental dataset as **PlantVillage**, reports **54,303 images**, and reports **38 species of plant diseases**. Therefore:

- The paper-supported dataset source is the cited **Mendeley Data V1 record**, DOI 10.17632/tywbtsjrjv.1.
- The paper explicitly identifies that dataset as PlantVillage.
- The PDF does not state the local archive filenames, the local 55,448-image count, the local 61,486-image count, or the reason for any count/class discrepancy.
- The PDF does not state a separate dataset URL for the Mendeley record.

The 2022 paper independently cites the related nine-layer research article, but its citation is not a complete dataset provenance record and does not report the local archive totals.

## 4. Local Dataset Information

The verified local dataset is located at:

`C:\Users\senso\Downloads\Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network`

It contains:

- `Plant_leaf_diseases_dataset_without_augmentation.zip`: 55,448 images.
- `Plant_leaf_diseases_dataset_with_augmentation.zip`: 61,486 images.
- 39 directory-based classes in each archive.
- Directory names used as labels.
- Approximately 256 × 256 RGB images.
- 0 corrupted/unreadable images in the verified archive scans.
- No official train/validation/test directories in the archive layout.

These local statistics come from the prior archive inspection, not from any external source and not from the PDFs.

## 5. Paper-Dataset Comparison

### 5.1 Comparison with the 2025 Hybrid CNN paper

| Attribute | 2025 paper | Verified local archive | Result |
|---|---:|---:|---|
| Dataset name | PlantVillage | Dataset directory named for the nine-layer dataset | Same provenance lead, but not identical naming evidence |
| Dataset citation | Mendeley Data V1, DOI 10.17632/tywbtsjrjv.1 | No embedded citation metadata | Probable relationship |
| Images | 54,303 | 55,448 unaugmented; 61,486 augmented | Not an exact count match |
| Classes/species | 38 | 39 | Not an exact class-count match |
| Augmentation | Explicit transformations listed | Separate augmented archive | Compatible in concept; exact generation not verified |
| Split | 80% train / 20% test | No official split directories | Not directly comparable |

**Classification: B. Probable match but not completely verified.**

The title, authors, Mendeley DOI, and PlantVillage identification provide a strong provenance lead. However, the local unaugmented archive does not exactly match the paper's reported image/class totals, and the PDFs do not explain the discrepancy. The 61,486-image archive is plausibly a generated variant of the same local source because it is the local “with augmentation” archive with the same directory labels, but the paper does not state that exact archive or confirm its generation.

### 5.2 Comparison with the 2024 EfficientNetB0/XAI paper

The paper reports a Kaggle “New Plant Diseases” dataset with 87,000 images, 38 classes, and 14 plant species. The local archive has 55,448 unaugmented images and 39 classes.

**Classification: C. No match.**

### 5.3 Comparison with the 2022 Novel CNN paper

The paper uses selected PlantVillage corn, potato, and tomato data together with separate rice and cassava datasets. It does not report the local 55,448-image archive or the local 39-class layout.

**Classification: C. No match established from the PDF.**

## 6. Recommended Case Study Paper

The most suitable single paper for the 20-mark AML Research Article Critique is:

**“Improving Plant Disease Classification With Deep-Learning-Based Prediction Model Using Explainable Artificial Intelligence” by Natasha Nigar et al., IEEE Access, 2024, DOI 10.1109/ACCESS.2024.3428553.**

It is suitable because it combines an advanced transfer-learning model (EfficientNetB0) with an explicit XAI method (LIME), reports a clearly described 87,000-image/38-class Kaggle dataset, gives an 80:20 training/validation design, defines evaluation measures, reports quantitative results, discusses model comparison and ANOVA, and describes a mobile deployment artifact. These provide enough technical and methodological material for a critical AML discussion. Its dataset is not the local project dataset, so the case study and project dataset must not be conflated.

## 7. Supporting Literature for Project

The following papers can support the planned project *Plant Disease Detection Using Transfer Learning*:

### Primary supporting paper

**Improving Plant Disease Classification With Deep-Learning-Based Prediction Model Using Explainable Artificial Intelligence** supports:

- EfficientNetB0 transfer learning;
- frozen feature extraction and classification head design;
- 80:20 validation methodology;
- accuracy, precision, recall, confusion matrix, and ANOVA;
- explainability with LIME;
- practical mobile deployment considerations.

### Additional supporting paper

**Plant Disease Identification Using a Novel Convolutional Neural Network** supports:

- a custom CNN baseline;
- comparison against transfer-learning architectures including VGG16, VGG19, InceptionV3, ResNet50, and DenseNet201;
- Inception, residual connections, depthwise separable convolution, batch normalization, dropout, and global average pooling;
- accuracy, precision, recall, F1-score, confusion matrices, and cross-validation;
- analysis of controlled versus field image conditions.

### Advanced architecture and augmentation supporting paper

**Hybrid CNN and Transformer-Based Sequential Learning Techniques for Plant Disease Classification** supports:

- advanced CNN/transformer and sequential-learning architectures;
- explicit image-augmentation transformations;
- 80:20 train/test splitting;
- accuracy, precision, recall, loss, AUC, training time, and confusion matrix;
- discussion of PlantVillage limitations and controlled backgrounds;
- provenance tracing to the Mendeley Data record associated with the nine-layer dataset title.

Together, these papers support the planned custom CNN baseline, transfer learning, augmentation, fine-tuning-oriented discussion, learning/evaluation methodology, error analysis, confusion matrices, and explainability. Grad-CAM is not the primary explanation method in the 2024 paper, which uses LIME; the project should present Grad-CAM as the project's planned additional method rather than claim it was used by that paper.

## 8. Dataset Citation

The citation explicitly stated in the available IEEE papers for the dataset matching the requested title is:

> J. A. Pandian and G. Geetharamani, “Data for: Identification of plant leaf diseases using a 9-layer deep convolutional neural network,” *Mendeley Data*, V1, 2019, DOI: **10.17632/tywbtsjrjv.1**.

The 2022 IEEE paper separately cites the related research article as:

> G. Geetharamani and P. J. Arun, “Identification of plant leaf diseases using a nine-layer deep convolutional neural network,” *Computers & Electrical Engineering*, Volume 76, pp. 323–338, June 2019.

The first citation is the appropriate dataset citation supported by the PDFs. The second is a related research-paper citation and should not replace the dataset citation.

## 9. Remaining Issues

- The dataset record itself and the nine-layer research article are not among the three PDFs in `C:\Users\senso\Downloads\IEEE Papers`; only their citations appear in the PDFs.
- The local unaugmented total (55,448) differs from the 2025 paper's reported total (54,303), and the local class count (39) differs from the paper's reported 38. The PDFs do not resolve this discrepancy.
- The exact relationship between the local `Background_without_leaves` directory and the paper's 38 reported disease species is not stated in the papers and must not be asserted as fact.
- The paper does not confirm the local ZIP filenames or the local 61,486-image augmented archive.
- No exact local archive checksum or Mendeley-record file manifest was available for byte-level verification.
- The 2024 paper's Kaggle dataset is a different dataset and should not be cited as the source of the local archive.
- No model training, dataset modification, or Streamlit implementation was performed.
