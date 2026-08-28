# Group12_MaleBin_ICE478

**Course** · ICE478 Machine Learning, Summer 2026 — East West University, Dhaka
**Instructor** · Raihan Ul Islam
**Track** · **3 — CNN + Attention** (image classification)
**Dataset** · MaleBin: Malware Binary Greyscale Images
**Group** · `Group12` — *replace with your group number and add member names/IDs below*

| Name | Student ID | Contribution |

| Mahamudul Hasan Pramanik | 2023-1-50-013 | 100% |

---

## 1. The project in one paragraph

A malware binary can be read as a stream of bytes, laid out row-wise into a
fixed-width 2-D array and rendered as an 8-bit grayscale image — a *byte-plot*.
Different malware families produce visibly different textures because they share
code, packers and resource layouts. We build **ByteAttnNet**, a small
from-scratch CNN with a two-part attention stack, to classify 39 malware families
from such byte-plots. The design follows one observation from the EDA: **in a
byte-plot the row index is the byte offset in the file**, so vertical position is
semantically meaningful (PE header at the top, code sections next, padding at the
end) while horizontal position is an artefact of the chosen row width. We
therefore use *coordinate attention* — a per-row and per-column gate that
preserves position — on top of multi-scale dilated convolutions, and we train it
with augmentation that never flips or rotates the file. Everything is evaluated
under a **duplicate-grouped, family-stratified split** so that polymorphic
near-duplicates cannot straddle train and test, with **macro-F1** as the headline
metric.

## 2. Dataset

| | |
|---|---|
| Name | MaleBin: Malware Binary Greyscale Images |
| Source | Kaggle · [`tashiee/malebin-malware-binary-greyscale-images`](https://www.kaggle.com/datasets/tashiee/malebin-malware-binary-greyscale-images) |
| Licence | CC BY 4.0 |
| Size | 12,464 images · 39 malware families · 256×256 · 8-bit grayscale · ~582 MB |
| Compiled from | (a) **Malimg** — Nataraj et al., *VizSec* 2011 (25 families); (b) a subset of [`walt30/malware-images`](https://www.kaggle.com/datasets/walt30/malware-images), visualised from MalwareBazaar samples by the same byte-to-pixel method |
| Problem type | Single-label multi-class image classification |

### Limitations we inherit and report (not hide)

The dataset uploader states three things that cap what any model can achieve
here, and we repeat them in all three task reports:

1. **The Malimg half is outdated malware.** It classifies well in a closed set
   but will not generalise to modern threats.
2. **Resizing distorts the images.** All samples were forced to 256×256, i.e.
   65,536 pixels stand in for files of very different true lengths. Any
   byte-offset reading is therefore approximate.
3. **A newer MaleBin 2.0 RGB dataset exists** and the uploader recommends it over
   v1. We use v1 because it is the version on the course dataset list.

## 3. Repository layout

```
Mahamudul_Pramanik-Group12_MaleBin_ICE478/
├── README.md
├── .gitignore
├── code/
│   ├── common/                                   ← generators + shared module
│   │   ├── malebin_common.py                     the shared library (single source of truth)
│   │   ├── nbtool.py                             notebook builder
│   │   ├── make_nb_task*.py                      regenerate the notebooks
│   │   ├── run_notebooks.py                      execute the chain; figures to disk, not inline
│   │   └── collect_results.py                    build the results summary from the artifacts
│   ├── requirements.txt
│   ├── task1/Group12_MaleBin_task1_eda.ipynb
│   ├── task2/Group12_MaleBin_task2_baselines.ipynb
│   ├── task2/Group12_MaleBin_task2_proposed_model.ipynb
│   ├── task3/Group12_MaleBin_task3_improvement_ablation.ipynb
│   └── task3/Group12_MaleBin_task3_explainability.ipynb
├── report/
│   ├── REAL_RUN.md                               ← the run's budget, caveats and bug log
│   ├── task1/Group12_MaleBin_task1_report.md     → export to .pdf
│   ├── task2/Group12_MaleBin_task2_report.md     → export to .pdf
│   └── task3/Group12_MaleBin_task3_report.md     → export to .pdf
├── figures/                                      ← every figure, as a file
├── related_work/
│   ├── Group00_MaleBin_related_work_table.md     → export to .pdf
│   ├── Group00_MaleBin_related_work_table.csv
│   └── papers/                                   ← put the 7 paper PDFs here
└── models/
    ├── README.md                                 how to regenerate the checkpoints
    └── Group12_MaleBin_label_map.json
```

Not tracked, by design: `artifacts/` (regenerable metrics and prediction arrays)
and `*.pth` checkpoints (~111 MB, one file at 94.7 MB). `models/README.md` and
each report say how to reproduce them.

Each notebook writes `malebin_common.py` itself in its second cell, so **any
notebook can be run on its own** — no cross-notebook imports and no fixed run
order.

## 4. How to run on Kaggle

1. **New Notebook** → *File → Import Notebook* → upload one of the five `.ipynb`
   files from `code/`.
2. **Add Data** → search `MaleBin malware binary greyscale` → **Add**. The
   notebook finds the folder automatically; no path editing needed.
3. **Settings** → *Accelerator*: **GPU T4 ×2 or P100**. *Internet*: off is fine —
   every library used ships with the Kaggle image.
4. Edit the one line `CFG.group = "Group00"` in the boot cell to your group
   number.
5. **Run All.** Then *Save Version → Save & Run All (Commit)*, which stores the
   notebook **with its outputs** — that committed version is what you download
   and push to this repo.

### Run order and approximate GPU time

| # | Notebook | Time | Needs |
|---|---|---|---|
| 1 | `task1_eda.ipynb` | 10–15 min | dataset |
| 2 | `task2_baselines.ipynb` | 40–70 min | dataset |
| 3 | `task2_proposed_model.ipynb` | 15–25 min | dataset (+ notebook 2's output, optional) |
| 4 | `task3_improvement_ablation.ipynb` | 2–4 h | dataset (+ notebook 2's output, optional) |
| 5 | `task3_explainability.ipynb` | 10–20 min | dataset (+ notebook 4's output, optional) |

The "optional" inputs are a convenience: add the earlier notebook's output via
**Add Data → Notebook Output** and the later notebook reuses its baselines,
predictions and checkpoint. If you skip that, the later notebook retrains what it
needs so it still produces real results — it just takes longer.

Notebook 4 is the long one. If a session times out, set `RUN_ABLATION = True,
RUN_CV = False` in one session and `RUN_ABLATION = False, RUN_CV = True` in the
next; the ablation table is reloaded from disk.

### Quick wiring check first (recommended)

Add `os.environ["MALEBIN_FAST"] = "1"` **above** `import malebin_common`, or set
`CFG.fast = True` immediately after the boot cell. That shrinks images to 64 px,
epochs to 2 and folds to 2, so all five notebooks finish in ~15 minutes total.
The numbers are meaningless — it only proves the code runs. Turn it off before
producing anything you will quote.

### Running locally instead

```bash
pip install -r code/requirements.txt
export MALEBIN_DATA_ROOT=/path/to/malebin      # or let auto-discovery find ./data/malebin
jupyter lab code/task1/Group12_MaleBin_task1_eda.ipynb
```

## 5. Method summary

### Leakage control — the decision that shapes every number

The brief (§6.2) requires a *subject/source-based* split. MaleBin ships **no**
subject, hash or source column, so we construct one:

* **exact duplicates** — SHA-1 of the resized pixel buffer;
* **near duplicates** — 128-bit difference hash (horizontal + vertical
  gradients), Hamming distance ≤ 6, searched within each family;
* both relations unioned with **union-find** → a `dup_group` id per image.

Every split — the hold-out train/val/test and all 5 CV folds — is
`StratifiedGroupKFold` on `dup_group`, stratified on family. The justification:
a malware family is a set of polymorphic variants of one binary, whose byte-plots
are near-identical, and MaleBin additionally merges two partly-overlapping
corpora. A random split therefore scores memorisation, not classification.
Notebook 1 §F and notebook 2a §3 both *measure* what this costs us.

Our effective sample size is the number of **distinct binaries**, not the number
of images.

### ByteAttnNet

```
input 1×224×224
  → stem: Conv3×3 s2 + BN + ReLU                                    48×112×112
  → 4 × [ MSConvBlock(3×3 ∥ 3×3 d2 ∥ 3×3 d3 → 1×1 fuse + skip)
          → CBAM (channel avg+max, then spatial 7×7)
          → CoordAtt (per-row gate × per-column gate)
          → MaxPool2×2 (stages 1–3) ]                               320×14×14
  → GeM pooling, learnable p ∈ [1, 8]                               320
  → Dropout(0.3) → Linear(320 → 39)
```

≈ 2.5 M parameters, ≈ 10 MB — roughly **10× smaller than ResNet50**.

* **Multi-scale dilated conv** — a family's signature lives at byte level,
  function level and section level simultaneously; dilation buys 5×5 and 7×7
  receptive fields at 3×3 parameter cost.
* **CBAM** — which texture channels, and which regions, this family needs.
* **CoordAtt** — the novel part. Pools one axis at a time, so a per-row gate
  survives, and a row *is* a byte-offset band. Channel-only attention (SE, as
  used by PAFE and IMCMK-CNN) pools that position away.
* **GeM** — lets the readout learn where to sit between average and peak texture
  energy instead of us guessing.
* **Byte-aware augmentation** — vertical roll (a section boundary moved), row
  crop (a longer/shorter file), random erasing (a repacked region). **No flips or
  rotations**: they reverse byte order and produce a file that cannot exist.

### Evaluation

Macro-F1 is the headline (imbalanced data, brief §6.2); model selection is on
*validation macro-F1*, never accuracy. Reported per model: accuracy, balanced
accuracy, macro/weighted precision-recall-F1, MCC, Cohen's κ, one-vs-rest
ROC-AUC and average precision, confusion matrix, per-class report, parameter
count, model size, training time and inference time.

### Statistical testing (brief §6.4)

| Evidence | Test | Why |
|---|---|---|
| 5 paired per-fold macro-F1 scores | **Wilcoxon signed-rank** | paired scores, not per-sample outcomes |
| per-sample predictions on the one shared test set | **McNemar** | needs the paired 2×2 table on the same samples |
| 3+ models on the same folds | **Friedman + Nemenyi** | controls the family-wise error rate |
| our number vs a *paper's* number | **no test** | we do not have their per-sample predictions |

With 5 folds the exact two-sided Wilcoxon *p* cannot go below 0.0625, so it can
never reach α = 0.05 regardless of effect size. We report it with that floor
stated, and treat the **McNemar test on the ~2.5 k-sample test set** as where the
statistical power actually is.

### Fair comparison to related work (Pillar A)

Two scopes, set by `CFG.eval_scope`:

* **`"malimg25"` — the fair comparison.** Restricts MaleBin to the 25 original
  Malimg families, i.e. the same dataset and label space as PAFE (2024, F1
  99.27), DRIN (2024, F1 99.05) and SE-AGM (2023, acc 99.43). The only remaining
  difference is our *harder* duplicate-grouped split.
* **`"malebin39"` — indicative only.** Full MaleBin against Alshomrani et al.
  (2025, 94.04 % on a merged 61-class corpus), the closest published analogue to
  a merged multi-source label space. Different dataset, so we never claim a win
  from it.

## 7. Reproducibility

* Single seed `CFG.seed = 42` for splits, initialisation and augmentation.
* The split is derived from image *content* (hashes), so it is identical across
  notebooks and machines without passing files around.
* `code/common/malebin_common.py` is the single source of truth; the notebooks
  embed a verbatim copy. To change the library, edit it and re-run
  `python code/common/make_nb_task1.py` (and `task2`, `task3`, `task3b`).

## 8. Credits

* Malimg dataset — L. Nataraj, S. Karthikeyan, G. Jacob, B. S. Manjunath,
  "Malware images: visualization and automatic classification," *VizSec*, 2011.
* MalwareBazaar-derived images — Kaggle user `walt30`.
* MaleBin compilation — Kaggle user `tashiee` (`tashvin.raj56@gmail.com`).
* Attention modules re-implemented from Hu et al. (CVPR 2018), Woo et al.
  (ECCV 2018) and Hou et al. (CVPR 2021); Grad-CAM from Selvaraju et al.
  (ICCV 2017); LIME from Ribeiro et al. (KDD 2016). Full citations in
  `related_work/`.
