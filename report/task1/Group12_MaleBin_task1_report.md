# Task 1 Report — Data Understanding & Related Work

**Group12 · ICE478 Summer 2026 · Track 3 (CNN + Attention) · Dataset: MaleBin**

> Fill every `⟨…⟩` from your Kaggle run of
> `code/task1/Group00_MaleBin_task1_eda.ipynb`. Every number below is printed by
> that notebook. Export this file to PDF as
> `report/task1/Group00_MaleBin_task1_report.pdf`.

---

## 1. Dataset

| Property | Value |
|---|---|
| Name | MaleBin: Malware Binary Greyscale Images |
| Source | Kaggle `tashiee/malebin-malware-binary-greyscale-images` (CC BY 4.0) |
| Compiled from | Malimg (Nataraj et al., VizSec 2011) + a subset of `walt30/malware-images` (MalwareBazaar) |
| Application area | Static malware family triage from byte-plot images (cyber security) |
| Problem type | Single-label multi-class image classification |
| Samples | ⟨12,464⟩ |
| Classes | ⟨39⟩ — of which ⟨25⟩ Malimg families (⟨N⟩ images) and ⟨14⟩ others (⟨N⟩ images) |
| Format / resolution / channels | PNG · 256×256 · 1 (8-bit grayscale) |
| Samples per class | min ⟨…⟩ · median ⟨…⟩ · max ⟨…⟩ |
| Imbalance ratio | ⟨…⟩ : 1 |
| Corrupt / unreadable files | ⟨0⟩ |
| Subject / source identifier | **not provided** — derived, see §4 |

A byte-plot is produced by reading a binary as a byte stream, laying it out
row-wise into a fixed-width 2-D array and rendering it as an 8-bit grayscale
image. **Row *r* of a 256-px-wide plot therefore holds bytes `[256r, 256r+256)`
of the file.** This single fact drives every design decision in Tasks 2–3.

## 2. Class balance and its effect on the metrics

⟨Insert `task1_B_class_balance.png`.⟩

Majority class ⟨…⟩ (⟨…⟩ images, ⟨…⟩ %); minority ⟨…⟩ (⟨…⟩ images, ⟨…⟩ %);
imbalance ratio ⟨…⟩ : 1. A majority-only classifier would score accuracy ⟨…⟩ but
macro-F1 ⟨…⟩.

**Consequence.** Accuracy is not a safe headline. Per brief §6.2 we report
**macro-F1 and per-class recall** as primary, select models on *validation
macro-F1*, and use inverse-frequency class weights in the loss.

## 3. What the images look like

⟨Insert `task1_C_sample_grid.png` and `task1_C_within_family.png`.⟩

Two observations, both load-bearing:

1. **Byte-plots are horizontally banded, not object-like.** Each image is a stack
   of bands: dense speckle where executable code sits, strictly periodic patterns
   where a table or padding repeats, uniform black where the file is zero-padded,
   high-entropy noise where a section is packed or encrypted. *Vertical* position
   is meaningful (it is the byte offset); *horizontal* position is an artefact of
   the row width. → We choose **coordinate attention** in Task 2 and **forbid
   flips and rotations** in augmentation.
2. **Families in one lineage look nearly identical** (⟨e.g. Allaple.A/Allaple.L,
   Swizzor.gen!E/Swizzor.gen!I⟩), and images *within* a family are often
   near-copies. That is a leakage hazard, quantified in §4.

## 4. Data quality and the leakage risk — the decisive section

| Check | Result |
|---|---|
| Corrupt / unreadable | ⟨0⟩ |
| Constant images (std ≈ 0) | ⟨…⟩ |
| Near-constant (std < 3) | ⟨…⟩ |
| Exact pixel duplicates | ⟨…⟩ images in ⟨…⟩ groups |
| Duplicate hashes spanning >1 family | ⟨…⟩ (would be label noise) |
| Near-duplicate links (dHash ≤ 6/128) | ⟨…⟩ |
| **Duplicate groups** | ⟨…⟩ for ⟨12,464⟩ images (⟨…⟩ % collapse) |
| Largest group | ⟨…⟩ images |
| Groups spanning >1 family | ⟨0⟩ |

**The argument.** Brief §6.2 requires a subject/source-based split. MaleBin ships
no such column, so we construct one. A malware family is a set of polymorphic
variants of the same program: two variants' byte-plots differ by a shifted
section, a repacked region or a few patched bytes, and are otherwise the same
picture. MaleBin also merges two partly-overlapping corpora, so one original
sample can appear twice. If a variant lands in train and its near-twin in test,
the score measures memorisation.

**Our grouping key.** SHA-1 of the pixel buffer (exact duplicates) ∪ 128-bit
difference hash with Hamming distance ≤ 6 searched within each family (near
duplicates), unioned with union-find. All splits and all 5 CV folds in Tasks 2–3
are `StratifiedGroupKFold` on that key.

**Measured cost.** A 1-NN classifier on 20 hand-made features scores macro-F1
⟨…⟩ under a random split and ⟨…⟩ under the grouped split — an inflation of
⟨+…⟩ (⟨…⟩ % relative). A weak model doing well under a random split *is* the
proof of memorisation.

**Consequence.** Our effective sample size is the number of **distinct
binaries** (⟨…⟩), not the image count. Our reported scores will be lower than
the published 99 %+ figures, and §7 explains why that is the honest number.

⟨Insert `task1_F_duplicates.png` and `task1_F_dup_group_example.png`.⟩

## 5. Pixel and texture statistics

⟨Insert `task1_E_histograms.png` and `task1_E_violin.png`; attach
`task1_E_feature_stats.csv`.⟩

Twenty interpretable descriptors per image (byte statistics + texture): mean,
median, std, min/max, quartiles, IQR, skewness, kurtosis, Shannon byte entropy,
zero-fraction, 0xFF-fraction, printable-ASCII fraction, gradient magnitude,
row/column variance, row autocorrelation, and top/bottom band means.

Findings: distributions are strongly non-Gaussian — large positive skew and heavy
kurtosis on `zero_frac`, `high_frac` and `mean`, because most binaries contain
long runs of `0x00` padding and `0xFF` filler. The most class-discriminative
descriptors by η² are ⟨entropy …, row_autocorr …, zero_frac …⟩. No single scalar
separates 39 classes, which is exactly the job handed to the CNN — but the
non-zero η² confirms the task is learnable from texture rather than noise.

## 6. Structure without a classifier: correlation, PCA, t-SNE, UMAP

⟨Insert `task1_G_correlation.png` and `task1_G_projections.png`.⟩

Redundant feature pairs (|r| > 0.85): ⟨…⟩. Silhouette scores on the family
labels: PCA-features ⟨…⟩, PCA-pixels ⟨…⟩, t-SNE ⟨…⟩, UMAP ⟨…⟩.

Reading: many families form tight, well-separated islands in t-SNE/UMAP on raw
pixel PCs, several of them *sub-clustered* — and those sub-clusters are the
duplicate groups from §4. A few families overlap heavily (⟨…⟩), which predicts
where the Task-2 confusion matrix will be dark. Improvements should therefore be
judged on **per-class recall for those families**, not on the average.

Interactive versions: `task1_H1_class_balance.html`, `task1_H2_projection.html`,
`task1_H3_features.html`.

## 7. Related work and the research gap

See `related_work/Group00_MaleBin_related_work_table.pdf` for the full
seven-paper table (2023–2025) with the Track-3 *attention type* column.

| Paper | Dataset | Attention | Headline |
|---|---|---|---|
| Li et al. 2024 (PAFE) | Malimg 25 | SE inside multi-scale fusion | F1 99.27 |
| Basak et al. 2024 (DRIN) | Malimg, MaleVis | Involution | F1 99.05 |
| Panda et al. 2023 (SE-AGM) | Malimg 25 | none (stacked ensemble) | Acc 99.43 |
| Alshomrani et al. 2025 | Malimg+MaleVis+VirusMNIST, 61 cls | Swin self-attention | Acc 94.04 |
| Makkawy et al. 2025 (MalVis) | MalVis 1.3 M, 10 cls | none | macro-F1 90.81 |
| Jayasudha et al. 2023 | Malimg, blended, MaleVis | none | Prec ≤ 97 |
| Zhang et al. 2024 (IMCMK-CNN) | Malimg | improved SE | ⟨read from PDF⟩ |

**The gap.** All seven share three properties:

1. **Every number comes from a random split.** None controls for polymorphic
   near-duplicates or for provenance overlap between merged corpora. §4 measured
   what that is worth on this data.
2. **The headline metric is accuracy or precision** on imbalanced data. Only
   MalVis reports macro-F1, and there it sits 4.4 points *below* accuracy
   (90.81 vs 95.19). The byte-plot papers never publish that gap; Basak et al.
   explicitly admit their model "struggles with underrepresented classes".
3. **Attention, where used, discards position** — SE and involution global-pool,
   Swin treats position as a generic 2-D coordinate at much higher cost. Nobody
   exploits the fact that **the row index is the byte offset in the file**.

**What we will do.** A small from-scratch CNN with *direction-aware* attention
(coordinate attention: a per-row and a per-column gate) on top of multi-scale
dilated convolutions, byte-aware augmentation that never flips or rotates, a
**duplicate-grouped** split, and **macro-F1** as the headline — reported on both
the Malimg-25 subset (strictly fair comparison) and full MaleBin-39.

## 8. Dataset limitations we inherit

The uploader states three things that cap what any model can achieve here, and we
carry them into Tasks 2 and 3 rather than hiding them:

1. The **Malimg half is outdated malware** — reliable in a closed set, not
   generalisable to modern threats.
2. **Resizing distorts the images**: all samples were forced to 256×256, so
   65,536 pixels stand in for files of very different true lengths. Any
   byte-offset reading is approximate.
3. A newer **MaleBin 2.0 RGB** dataset exists and the uploader recommends it over
   v1; we use v1 because it is the version on the course dataset list.

## 9. Task 1 conclusions → Task 2 decisions

| Finding | Decision |
|---|---|
| Uniform 256×256×1, no corrupt files | Input is `1×224×224` after one resize; no colour pipeline |
| Imbalance ⟨…⟩ : 1 | macro-F1 headline; inverse-frequency class weights; select on val macro-F1 |
| No subject column but heavy near-duplication | Duplicate-grouped stratified splits and CV folds; effective *n* = distinct binaries |
| Row position = byte offset; images are banded | **Coordinate attention**; **no** flips or rotations |
| Signal at byte / block / section scale | **Multi-scale dilated** conv block |
| A few overlapping variant pairs | Judge improvements on those families' recall |
| All related work: random split + accuracy headline | Our numbers will be lower; the fair comparison is the Malimg-25 scope |
