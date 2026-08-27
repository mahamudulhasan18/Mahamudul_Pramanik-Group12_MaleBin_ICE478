# Related Work — Group00_MaleBin_CSE475 (Track 3: CNN + Attention)

Seven peer-reviewed / archival papers, 2023–2025, on image-based malware
classification. Columns follow brief §6.3, plus the **attention type**
column Track 3 requires.

`Comparable` marks the rows §6.4 permits us to compare our result against:
same dataset family (Malimg), same input type (grayscale byte-plots), same
task (multi-class family classification). Rows marked *no* differ in dataset
or task and are context only.


## Summary table

| # | Paper (year) | Dataset | Method | Attention type | Headline metric | Comparable |
|---|---|---|---|---|---|---|
| 1 | S. Li et al. (2024) | Malimg (25 families, 9,435 images) | CNN with FFSE blocks = multi-scale feature fusion + Squeeze-and-Excitation channel attention | Channel (Squeeze-and-Excitation) inside a multi-scale fusion block | F1 = 99.27 | **yes** |
| 2 | M. Basak et al. (2024) | Custom (25 families, 49,374), Malimg, MaleVis | Dynamic Residual Involution Network (DRIN): involution kernels that are spatially specific and channel-agnostic, i.e. attention baked into the kernel | Involution (spatially specific, channel-agnostic) + residual | F1 = 99.05 | **yes** |
| 3 | P. Panda et al. (2023) | Malimg (25 families) | Stacked ensemble of autoencoder + GRU + MLP over 25 CNN-extracted features | None (stacked ensemble, not attention) | Accuracy = 99.43 | **yes** |
| 4 | M. Alshomrani et al. (2025) | Malimg + MaleVis + VirusMNIST combined (61 classes) | ConvNeXt-Tiny (local features) fused with Swin Transformer (global context) | Self-attention (shifted-window Swin) + convolutional local features | Accuracy = 94.04 | **yes** |
| 5 | S. J. Makkawy et al. (2025) | MalVis (>1.3M images, 9 malware classes + benign) | Entropy + N-gram enhanced visualisation | None | macro-F1 = 90.81 | no |
| 6 | Jayasudha M et al. (2023) | Malimg, a blended dataset, and MaleVis (three imbalance levels) | Six multi-class transfer-learning models | None (pure transfer learning) | Precision = 97.00 | no |
| 7 | D. Zhang et al. (2024) | Malimg and other image-based malware sets | Multi-scale Kernel (MK) block mixing large and small kernels plus an improved Squeeze-and-Excitation block | Improved Squeeze-and-Excitation channel attention | see below | **yes** |

## Full entries


### 1. PAFE: A lightweight visualization-based fast malware classification method

| Field | Value |
|---|---|
| Authors | S. Li, J. Wang, S. Wang, Y. Song |
| Year | 2024 |
| Venue | Heliyon 10, e35965 |
| DOI | `10.1016/j.heliyon.2024.e35965` |
| Dataset | Malimg (25 families, 9,435 images) |
| Application area | Windows PE malware family classification from grayscale byte-plots |
| Method / model | CNN with FFSE blocks = multi-scale feature fusion + Squeeze-and-Excitation channel attention; pixel-padding resize instead of interpolation; 256x256 input |
| **Attention type** | Channel (Squeeze-and-Excitation) inside a multi-scale fusion block |
| Reported metrics | Accuracy 99.25%, Precision 99.29%, Recall 99.25%, F1 99.27%, inference 10.04 ms, 721,913 params |
| Strengths | Best published Malimg accuracy/latency trade-off; tiny (0.72M params); pixel-padding avoids the texture distortion that bilinear resizing causes; reports timing and parameter count, not just accuracy |
| Limitations | Random (not source-grouped) split, so polymorphic near-duplicates can straddle train/test; no macro-F1 or per-class recall on the rare families; authors themselves note generalisation to new variants is unverified; no explainability |
| Research gap | Leakage-controlled evaluation and per-class (macro) reporting are missing; attention is channel-only, so byte-offset position is discarded |
| Relation to our work | Our Pillar-A target. We reproduce the same Malimg 25-family task as a subset of MaleBin and compare F1, but under a duplicate-grouped split |
| Comparable under §6.4 | yes |

### 2. Attention-Based Malware Detection Model by Visualizing Latent Features Through Dynamic Residual Kernel Network

| Field | Value |
|---|---|
| Authors | M. Basak, D.-W. Kim, M.-M. Han, G.-Y. Shin |
| Year | 2024 |
| Venue | Sensors 24(24), 7953 |
| DOI | `10.3390/s24247953` |
| Dataset | Custom (25 families, 49,374), Malimg, MaleVis |
| Application area | Malware family classification from visualised binaries |
| Method / model | Dynamic Residual Involution Network (DRIN): involution kernels that are spatially specific and channel-agnostic, i.e. attention baked into the kernel |
| **Attention type** | Involution (spatially specific, channel-agnostic) + residual |
| Reported metrics | Malimg: Acc 99.3%, P 0.992, R 0.989, F1 0.9905 | MaleVis: Acc 98.9%, F1 0.9892 | Custom: Acc 99.5%, F1 0.9948 |
| Strengths | Attention is intrinsic to the kernel rather than bolted on; validated on three datasets; heat-map visualisation attempted |
| Limitations | Authors state it still struggles on under-represented classes; heavier than lightweight CNNs; sensitive to preprocessing noise; scalability to unseen families untested; interpretability admitted to be limited |
| Research gap | The under-represented-class weakness is exactly what macro-F1 exposes and what class-weighted training plus a balanced dataset can address |
| Relation to our work | Second comparable Malimg baseline; its admitted rare-class weakness motivates our macro-F1-driven model selection and class-weighted loss |
| Comparable under §6.4 | yes |

### 3. Transfer Learning for Image-Based Malware Detection for IoT (SE-AGM)

| Field | Value |
|---|---|
| Authors | P. Panda, C. U. Om Kumar, S. Marappan, M. Suresh, S. Manimurugan, D. Veesani Nandi |
| Year | 2023 |
| Venue | Sensors 23(6), 3253 |
| DOI | `10.3390/s23063253` |
| Dataset | Malimg (25 families) |
| Application area | IoT malware detection from byte-plot images |
| Method / model | Stacked ensemble of autoencoder + GRU + MLP over 25 CNN-extracted features; each stage's output feeds the next; data augmentation studied |
| **Attention type** | None (stacked ensemble, not attention) |
| Reported metrics | Average accuracy 99.43% on Malimg |
| Strengths | Highest reported Malimg accuracy in our set; very cheap at inference because it classifies only 25 encoded features; ablates augmentation |
| Limitations | Reports accuracy only -- no macro-F1, no per-class recall, no confusion matrix on the rare families, so the number cannot be checked against the imbalance rule; feature extractor trained on the same data it later encodes; random split |
| Research gap | An accuracy-only headline on an imbalanced 25-class set is exactly what Sec. 6.2 warns about; the result is not verifiable per class |
| Relation to our work | Highest accuracy number we must acknowledge, but it is accuracy-only, so we compare our accuracy to it and explain why macro-F1 is the fairer basis |
| Comparable under §6.4 | yes |

### 4. An Explainable Hybrid CNN-Transformer Architecture for Visual Malware Classification

| Field | Value |
|---|---|
| Authors | M. Alshomrani, A. Albeshri, A. A. Alsulami, B. Alturki |
| Year | 2025 |
| Venue | Sensors 25(15), 4581 |
| DOI | `10.3390/s25154581` |
| Dataset | Malimg + MaleVis + VirusMNIST combined (61 classes); also Maldeb, Dumpware-10 |
| Application area | Visual malware classification across merged sources |
| Method / model | ConvNeXt-Tiny (local features) fused with Swin Transformer (global context); Grad-CAM for interpretability; real-time deployment demo |
| **Attention type** | Self-attention (shifted-window Swin) + convolutional local features |
| Reported metrics | Combined 61-class validation accuracy 94.04% (ConvNeXt-Tiny alone 92.45%, Swin alone 90.44%); Maldeb 98%; Dumpware-10 97% |
| Strengths | The only paper in our set that evaluates a *merged multi-source* label space, which is what MaleBin is; uses Grad-CAM and discusses it; shows the hybrid beats either half |
| Limitations | Accuracy on a validation split rather than an untouched test set; no macro-F1 on 61 imbalanced classes; Grad-CAM interpreted without acknowledging that byte-plots have no semantic regions; heavy backbones |
| Research gap | Merged-source label spaces drop ~5 points versus single-source Malimg, and nobody reports macro-F1 there; also no duplicate control across merged sources |
| Relation to our work | Closest analogue to our 39-class MaleBin setting (merged sources, more classes). This is the paper our full-MaleBin number is compared against |
| Comparable under §6.4 | yes |

### 5. MalVis: A Large-Scale Image-Based Framework and Dataset for Advancing Android Malware Classification

| Field | Value |
|---|---|
| Authors | S. J. Makkawy, M. J. De Lucia, K. E. Barner |
| Year | 2025 |
| Venue | arXiv:2505.12106 |
| DOI | `10.48550/arXiv.2505.12106` |
| Dataset | MalVis (>1.3M images, 9 malware classes + benign) |
| Application area | Android malware classification from bytecode visualisations |
| Method / model | Entropy + N-gram enhanced visualisation; MobileNetV2 / DenseNet201 / ResNet50 / InceptionV3 with eight ensemble strategies; undersampling |
| **Attention type** | None |
| Reported metrics | Accuracy 95.19%, macro-F1 90.81%, Precision 92.58%, Recall 89.10%, MCC 87.58%, ROC-AUC 98.06% |
| Strengths | Reports macro-F1, MCC and ROC-AUC -- the honest metric set for imbalanced data; huge scale; explicit imbalance handling |
| Limitations | Android bytecode, not Windows PE byte-plots, so not directly comparable; undersampling discards data; no attention module |
| Research gap | Shows how far macro-F1 sits below accuracy on imbalanced visual malware data (95.19 vs 90.81) -- a gap the PE-image papers never report |
| Relation to our work | Context, not a comparison target. It is our evidence that macro-F1 is the right headline metric and that accuracy overstates performance |
| Comparable under §6.4 | no |

### 6. Comparative Analysis of Imbalanced Malware Byteplot Image Classification using Transfer Learning

| Field | Value |
|---|---|
| Authors | Jayasudha M, A. Shaik, G. Pendharkar, S. Kumar, Muhesh Kumar B, S. Balaji |
| Year | 2023 |
| Venue | PEIS 2023, Lecture Notes in Electrical Engineering; arXiv:2310.02742 |
| DOI | `10.48550/arXiv.2310.02742` |
| Dataset | Malimg, a blended dataset, and MaleVis (three imbalance levels) |
| Application area | Byte-plot malware classification under class imbalance |
| Method / model | Six multi-class transfer-learning models; ResNet50, EfficientNetB0 and DenseNet169 were the strongest |
| **Attention type** | None (pure transfer learning) |
| Reported metrics | Max precision 97% (imbalanced), 95% (intermediate), 95% (balanced); more imbalance -> faster convergence but higher variance across models |
| Strengths | Directly studies the imbalance axis; uses a blended (multi-source) dataset like MaleBin; documents the convergence/variance trade-off |
| Limitations | Precision-only headline; no macro-F1 or per-class recall; no attention or custom architecture; no duplicate control across the blend |
| Research gap | Blended multi-source byte-plot data is under-studied and reported with the wrong metric |
| Relation to our work | Justifies our baseline pool (ResNet50, DenseNet121, EfficientNet-B0) and our decision to report variance across folds, not a single number |
| Comparable under §6.4 | no |

### 7. IMCMK-CNN: A lightweight convolutional neural network with Multi-scale Kernels for Image-based Malware Classification

| Field | Value |
|---|---|
| Authors | D. Zhang, Y. Song, Q. Xiang, Y. Wang |
| Year | 2024 |
| Venue | Alexandria Engineering Journal 111, 203-220 |
| DOI | `10.1016/j.aej.2024.10.055` |
| Dataset | Malimg and other image-based malware sets |
| Application area | Malware variant classification from byte-plot images |
| Method / model | Multi-scale Kernel (MK) block mixing large and small kernels plus an improved Squeeze-and-Excitation block; fusion strategy keeps the parameter cost of small kernels |
| **Attention type** | Improved Squeeze-and-Excitation channel attention |
| Reported metrics | [FILL FROM PDF -- read the results table of the published version and replace this string before you submit Task 1] |
| Strengths | Directly motivates multi-scale kernels for byte texture; explicitly targets the parameter cost of large kernels |
| Limitations | Channel attention only, so byte-offset position is not modelled; single-source evaluation |
| Research gap | Multi-scale + channel attention is established; direction-aware (positional) attention for byte-plots is not |
| Relation to our work | The architectural ancestor of our multi-scale dilated block. We keep its multi-scale idea and add the positional attention it lacks |
| Comparable under §6.4 | yes |

## The research gap we address

All seven papers share three properties:

1. **Every reported number comes from a random split.** None controls for
   polymorphic near-duplicates, and none controls for provenance overlap between
   merged corpora. So the 99.2–99.4 % Malimg figures are upper bounds under an
   optimistic protocol, not estimates of generalisation.
2. **The headline metric is almost always accuracy or precision**, on imbalanced
   data. Only MalVis (2025) publishes macro-F1, and there it sits **4.4 points
   below** accuracy (90.81 vs 95.19). The PE byte-plot papers never publish that
   gap — and Basak et al. (2024) explicitly admit their model "struggles with
   underrepresented classes".
3. **Attention, where present, discards position.** SE in PAFE and IMCMK-CNN,
   involution in DRIN, shifted-window self-attention in Alshomrani et al. None
   exploits the one structural fact a byte-plot actually has: **the row index is
   the byte offset in the file.** Channel attention global-pools it away; window
   self-attention treats it as a generic 2-D coordinate at much higher cost.

**Our contribution.** A small from-scratch CNN whose attention is
*direction-aware* — coordinate attention gives a per-row and a per-column gate,
so byte-offset position survives — stacked on multi-scale dilated convolutions,
trained with augmentation that never flips or rotates the file, and evaluated
under a **duplicate-grouped** split with **macro-F1** as the headline. We report
both a strictly fair comparison (Malimg-25 subset, same classes as the published
work) and an indicative one (full MaleBin-39), plus 5-fold CV and significance
tests, instead of a single flattering number.

## Note on entry 7

The `metrics` field for IMCMK-CNN is a placeholder. Download the published
version (DOI `10.1016/j.aej.2024.10.055`), read its results table, and replace
the string in `RELATED_WORK` inside `code/common/malebin_common.py`, then re-run
the generators. We have deliberately left it flagged rather than guessing a
number.

## Paper PDFs

Place the seven PDFs in `related_work/papers/`, named `01_PAFE_2024.pdf`,
`02_DRIN_2024.pdf`, `03_SEAGM_2023.pdf`, `04_Hybrid_2025.pdf`,
`05_MalVis_2025.pdf`, `06_Byteplot_2023.pdf`, `07_IMCMK_2024.pdf`.
