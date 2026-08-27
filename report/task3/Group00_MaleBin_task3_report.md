# Task 3 Report (Final) — Improvement, Comparison & Explainability

**Group00 · CSE475 Summer 2026 · Track 3 (CNN + Attention) · Dataset: MaleBin**

> Fill every `⟨…⟩` from your Kaggle runs of
> `task3_improvement_ablation.ipynb` and `task3_explainability.ipynb`.
> Export to `report/task3/Group00_MaleBin_task3_report.pdf`.

---

## 1. Summary

We classify 39 malware families from 256×256 grayscale byte-plots with
**ByteAttnNet** — a ⟨2.5⟩ M-parameter from-scratch CNN combining multi-scale
dilated convolutions with CBAM and coordinate attention. Everything is evaluated
under a **duplicate-grouped, family-stratified** split, because MaleBin's
polymorphic near-duplicates make a random split measure memorisation.

| | Result |
|---|---|
| Final macro-F1 (held-out test) | ⟨…⟩ |
| Final macro-F1 (5-fold CV) | ⟨…⟩ ± ⟨…⟩ |
| Final accuracy | ⟨…⟩ |
| Best baseline (⟨…⟩) macro-F1, 5-fold CV | ⟨…⟩ ± ⟨…⟩ |
| Wilcoxon (5 folds) vs best baseline | W = ⟨…⟩, *p* = ⟨…⟩ → ⟨…⟩ |
| McNemar (shared test set) vs best baseline | χ² = ⟨…⟩, *p* = ⟨…⟩ → ⟨…⟩ |
| Friedman across 3 models | χ² = ⟨…⟩, *p* = ⟨…⟩ → ⟨…⟩ |
| Pillar-A verdict, scope `malimg25` | ⟨Beat / Match / Below⟩ |
| Pillar-A verdict, scope `malebin39` | ⟨Beat / Match / Below⟩ (indicative only) |
| Parameters vs best baseline | ⟨…⟩× fewer |

## 2. Ablation study — what each design decision is actually worth

Protocol: one variable changed at a time; same split, seed, epoch budget and
early-stopping criterion for every run; ranking read from **held-out test
macro-F1**, not from the value we early-stopped on.

### 2.1 Attention — the central claim

| Variant | macro-F1 | Δ vs no attention | Reading |
|---|---|---|---|
| `attn=none` (control) | ⟨…⟩ | — | |
| `attn=se` (channel only — PAFE / IMCMK-CNN) | ⟨…⟩ | ⟨…⟩ | |
| `attn=spatial` | ⟨…⟩ | ⟨…⟩ | |
| `attn=cbam` (channel + spatial) | ⟨…⟩ | ⟨…⟩ | |
| `attn=coord` (per-row / per-column gates) | ⟨…⟩ | ⟨…⟩ | |
| `attn=cbam+coord` (proposed) | ⟨…⟩ | ⟨…⟩ | |

⟨Insert `task3_attention_ladder.png`.⟩

**Interpret honestly against these cases:**
* `coord > se` → position-aware attention beats channel-only attention on
  byte-plots. This is our contribution, confirmed, and it was *predicted in
  advance* from the Task-1 finding that the row index is the byte offset.
* `coord ≈ se` → position adds nothing here. Report it as a negative result. Do
  not dress it up.
* `cbam+coord > both` → the two are complementary, so the stack is justified.
* `cbam+coord < coord` → the stack is redundant; the honest final model uses
  `coord` alone.

⟨Which case did you land in? State it in one sentence.⟩

### 2.2 Architecture and training recipe

| Variant | macro-F1 | Δ vs proposed | Verdict |
|---|---|---|---|
| `no-multiscale` | ⟨…⟩ | ⟨…⟩ | ⟨multi-scale helped / did not⟩ |
| `pool=gap` (GeM removed) | ⟨…⟩ | ⟨…⟩ | |
| `no-batchnorm` | ⟨…⟩ | ⟨…⟩ | |
| `dropout=0.5` | ⟨…⟩ | ⟨…⟩ | |
| `depth=shallow` | ⟨…⟩ | ⟨…⟩ | |
| `aug=none` | ⟨…⟩ | ⟨…⟩ | |
| `aug=naive` (flips + rotations) | ⟨…⟩ | ⟨…⟩ | ⟨did byte-aware augmentation win?⟩ |
| `no-class-weights` | ⟨…⟩ | ⟨…⟩ | |
| `optimizer=sgd` | ⟨…⟩ | ⟨…⟩ | |
| `scheduler=plateau` | ⟨…⟩ | ⟨…⟩ | |

⟨Insert `task3_ablation.png`.⟩

Bars **left of zero** = removing/changing that component hurt, i.e. our Task-2
choice was right. Bars **right of zero** = the change helped, i.e. our Task-2
reasoning was wrong on that point. Both outcomes are reportable; the second is
more interesting.

### 2.3 The final configuration

Selection rule, applied mechanically to the numbers: within each ablation group,
adopt the best variant **only** if it beat the reference configuration by more
than 0.002 macro-F1 (roughly run-to-run noise). Otherwise the simpler original
choice stands.

Adopted: ⟨…⟩
Rejected (so the original choice stands): ⟨…⟩

Final model: ⟨…⟩ Final training recipe: ⟨…⟩

## 3. Final performance

| Model | macro-F1 | Accuracy | weighted-F1 | macro-recall | AUC (OvR) | MCC | Params | Size (MB) | Inference (ms/img) |
|---|---|---|---|---|---|---|---|---|---|
| Best baseline (⟨…⟩) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| ByteAttnNet v1 (Task 2) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| **ByteAttnNet FINAL** | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

⟨Insert `task3_final_history.png`, `task3_final_cm.png`, `task3_final_roc.png`,
`task3_final_pr.png`; attach the per-class report.⟩

Worst five families by recall: ⟨…⟩. Compare against Task 2's list — did the
improvements land where the baselines were weakest?

## 4. Cross-validation (brief §6.4)

5-fold `StratifiedGroupKFold` on the duplicate-group key, so no near-duplicate
crosses a fold boundary. Within each fold's training part a further grouped
validation slice is carved for early stopping, so the fold's held-out part is
never seen during training or selection.

| Model | macro-F1 | Accuracy | weighted-F1 | macro-recall |
|---|---|---|---|---|
| ByteAttnNet FINAL | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ |
| ⟨best baseline⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ |
| ByteAttnNet no-attention | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ | ⟨…⟩ ± ⟨…⟩ |

⟨Insert `task3_cv.png`.⟩

Per-fold table: ⟨paste `task3_cv_scores.csv`⟩.

## 5. Significance testing

### Which test applies to which evidence, and why

| Evidence we have | Test used | Why this one |
|---|---|---|
| 5 paired per-fold macro-F1 scores | **Wilcoxon signed-rank** | The observations are 5 paired *scores*. McNemar cannot consume fold averages |
| Per-sample predictions from both models on the **one shared test set** | **McNemar** | Needs the paired 2×2 table of who got which sample right, which requires the same samples for both models |
| 3 models on the same folds | **Friedman + Nemenyi** | Repeated pairwise tests inflate the family-wise error rate |
| Our number vs a **paper's** number | **none — no test is possible** | We have no access to their per-sample predictions and their split differs |

### 5.1 Wilcoxon signed-rank, ByteAttnNet FINAL vs ⟨best baseline⟩

| fold | FINAL | baseline | diff |
|---|---|---|---|
| 1 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 2 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 3 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 4 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| 5 | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

W = ⟨…⟩; *p* (two-sided) = ⟨…⟩; *p* (one-sided) = ⟨…⟩; paired *t* = ⟨…⟩,
*p* = ⟨…⟩; Cohen's *d* = ⟨…⟩. **Verdict at α = 0.05: ⟨significant / not
significant⟩.**

**Stated limitation.** With *n* = 5 the exact two-sided Wilcoxon *p* cannot go
below **0.0625**, so this test can *never* reach α = 0.05 regardless of effect
size. We report it because the brief requires it, and we report the paired
*t*-test and effect size alongside so a reader can judge the magnitude. The real
statistical power is in §5.2.

### 5.2 McNemar on the shared held-out test set

| | both correct | only FINAL | only baseline | both wrong |
|---|---|---|---|---|
| ⟨best baseline⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

χ² (Yates-corrected) = ⟨…⟩ on ⟨…⟩ discordant pairs; *p* = ⟨…⟩.
**Verdict at α = 0.05: ⟨REJECT H₀ — the models differ significantly / fail to
reject H₀⟩.** Direction: ⟨…⟩ made more exclusive correct calls.

⟨Repeat for each baseline that has saved per-sample predictions.⟩

### 5.3 Friedman + Nemenyi across 3 models

χ² = ⟨…⟩, *p* = ⟨…⟩. Average ranks (1 = best): ⟨…⟩. Nemenyi critical difference
at α = 0.05: ⟨…⟩. Pairs exceeding the CD: ⟨…⟩.

Note: Nemenyi is conservative at *N* = 5 folds; non-significant pairs are "not
resolved", not "identical".

## 6. Fair comparison to related work (Pillar A)

### Comparison 1 — the fair one (`CFG.eval_scope = "malimg25"`)

MaleBin restricted to the 25 original Malimg families: **the same dataset and the
same label space** as the published work. The only remaining difference is our
*harder* duplicate-grouped split.

| System | Split | Metric | Score |
|---|---|---|---|
| Li et al. 2024 (PAFE) | random | F1 | 99.27 |
| Basak et al. 2024 (DRIN) | random | F1 | 99.05 |
| Panda et al. 2023 (SE-AGM) | random | Accuracy | 99.43 |
| Zhang et al. 2024 (IMCMK-CNN) | random | Accuracy | ⟨read from PDF⟩ |
| **ByteAttnNet FINAL (ours)** | **duplicate-grouped** | macro-F1 | ⟨…⟩ |
| **ByteAttnNet FINAL (ours)** | **duplicate-grouped** | weighted-F1 | ⟨…⟩ |
| **ByteAttnNet FINAL (ours)** | **duplicate-grouped** | Accuracy | ⟨…⟩ |

Pillar-A target: **PAFE 2024, weighted-F1 99.27.** Like-for-like on their metric:
ours ⟨…⟩ vs theirs 99.27 → ⟨±…⟩ points → **⟨Beat / Match / Below⟩**.

### Comparison 2 — indicative only (`CFG.eval_scope = "malebin39"`)

Full MaleBin-39 vs Alshomrani et al. (2025), 94.04 % accuracy on a merged
61-class visual-malware corpus. This is the closest published analogue to a
merged multi-source label space, but it is **not the same dataset**, so we present
it as context and claim nothing from it: ours ⟨…⟩ vs 94.04 → ⟨…⟩.

⟨Insert `task3_related_work_comparison.png`.⟩

### Documented experimental differences

| Dimension | Published work | Ours |
|---|---|---|
| Split | random stratified | duplicate-grouped stratified (worth ⟨…⟩ macro-F1, Task 2 §2) |
| Headline metric | accuracy / weighted-F1 | **macro-F1** (plus accuracy and weighted-F1 for comparability) |
| Classes | 25 (Malimg) or 61 (merged) | 25 (`malimg25` scope) or 39 (`malebin39` scope) |
| Corpus | original Malimg | MaleBin's re-balanced, re-resized copy of Malimg |
| Resize | pixel padding (PAFE) | bilinear, already applied by the dataset uploader |
| Validation | single random split, often val-set numbers | untouched test set + 5-fold grouped CV |

We run **no statistical test against any published number** — we have neither
their per-sample predictions nor their split. Doing so would be the "unfair paper
comparison" the brief names as a common mistake.

## 7. Explainability

### 7.1 The honest limitation, first

Brief §6.2: *"Malware byte-images: they classify well, but Grad-CAM/LIME on them
means little (no real 'regions' to point at). Say this instead of inventing an
explanation."*

We agree. In a photograph a Grad-CAM blob over a dog's face means "the model used
the dog's face", because a face is independently recognisable. In a byte-plot
there is no dog: a bright blob at pixel (137, 82) is a byte value at one offset in
one file, and no human can confirm or refute anything from it. So a heat-map here
is **not** evidence the model is right, and we do not present it as such.

### 7.2 What we did instead — four levels

| Level | Method | What it honestly tells us |
|---|---|---|
| 1 | Grad-CAM / Grad-CAM++ (required) | Where the target logit's gradient is large. A **diagnostic**, not an explanation |
| 2 | Byte-offset profile of the same map | Row *r* holds bytes `[rW, rW+W)`, so the row-mean is importance vs **relative file position** — a checkable claim about PE layout |
| 3 | Row-band **occlusion** | Blank out a band of file offsets and measure the probability drop. **Causal**, so it can confirm or refute level 2 |
| 4 | Coordinate-attention **row gates** | The model's own learned per-row weights — no attribution method, no surrogate |

### 7.3 Case A — a confident CORRECT prediction

File ⟨…⟩; true ⟨…⟩; predicted ⟨…⟩ (*p* = ⟨…⟩).

Top attributed byte-offset bands: ⟨…⟩ % – ⟨…⟩ % of the file (importance ⟨…⟩), …
Most causally important band by occlusion: ⟨…⟩ % – ⟨…⟩ % (costs ⟨…⟩ probability).
Grad-CAM ↔ occlusion agreement: *r* = ⟨…⟩ → ⟨the two agree, so the offset reading
is supported / weak agreement, so Grad-CAM was decorative here⟩.
Corr(deepest CoordAtt row gate, occlusion drop) = ⟨…⟩ → ⟨the mechanism we designed
tracks what actually matters / it does not, so the Task-2 justification is not
confirmed here⟩.

⟨Insert `task3_gradcam_A_correct.png`, `task3_occlusion_A_correct.png`,
`task3_lime_A_correct.png`, `task3_coordatt_gates_A_correct.png`.⟩

### 7.4 Case B — a confident WRONG prediction

File ⟨…⟩; true ⟨…⟩; predicted ⟨…⟩ (*p* = ⟨…⟩); *p*(true) = ⟨…⟩.

⟨Same four readings. Note whether the Grad-CAM maps for the predicted and the
true class differ, and what offset bands each favours — that is a genuine
observation about which parts of the input support which class.⟩

### 7.5 Per-family consistency of the attended offset

⟨Insert `task3_offset_consistency.png`.⟩ Families whose attended offset is stable
(std < 10 % of the file): ⟨…⟩ of ⟨…⟩. A small error bar is a family-level
structural claim; a large one means the attribution is unstable and must not be
interpreted.

### 7.6 LIME, and why its default segmentation is wrong here

| Case | Segmentation | Surrogate R² |
|---|---|---|
| A | row-band grid (ours) | ⟨…⟩ |
| A | quickshift (LIME default) | ⟨…⟩ |
| B | row-band grid (ours) | ⟨…⟩ |
| B | quickshift (LIME default) | ⟨…⟩ |

Quickshift looks for colour-coherent regions — a natural-image assumption. A
byte-plot has none, so quickshift returns blobs corresponding to nothing, and
LIME still ranks them confidently. Our row-band grid makes every segment a
contiguous range of **file offsets**, so a weight on it is a statement about a
region of the binary. The surrogate R² is the honest health check: it is how well
the sparse *linear* model reproduces the CNN locally, and low R² means the picture
explains a fit that does not hold.

### 7.7 What is *not* established

1. A Grad-CAM blob on a byte-plot is not a human-verifiable explanation. We make
   no claim that any highlighted region "is the malicious code".
2. LIME's local surrogate fits poorly here (§7.6); the image looks like an
   explanation but the linear model behind it does not reproduce the CNN.
3. Quickshift segmentation is not meaningful on byte-plots; we include it only to
   demonstrate that.
4. Attribution ≠ causation, and both ≠ correctness. Even a stable, causal,
   family-consistent offset band could be tracking a compiler artefact, a packer
   stub, or the uploader's resize behaviour — all correlated with family in this
   corpus without being anything a defender should rely on.
5. The dataset caps interpretability: 256×256 stands in for files of very
   different true lengths, so "12 % into the file" is approximate and, for files
   far from 64 KB, only ordinal.
6. Two samples are two samples. §7.5 is the only part with enough samples to
   generalise.

## 8. What worked, what did not, and why (Pillar-B Q4)

**What worked.** ⟨List the ablation rows left of zero with their deltas. If
`coord` beat `se`, say so and connect it to the Task-1 finding that predicted it —
a hypothesis stated in advance and then confirmed is the strongest result you can
report. If `aug=byte` beat `aug=naive`, that confirms the byte-plot geometry
argument.⟩

**What did not work.** ⟨List the ablation rows right of zero: changes we did not
predict that helped, and what that says about our reasoning. If `cbam+coord` was
no better than `coord` alone, report the stack as redundant. If attention as a
whole added ~nothing, say so clearly.⟩

**Why we are below the published numbers.** Not a weaker model but a measured
protocol difference: Task 2 §2 quantified what a random split is worth on this
data (⟨+…⟩ macro-F1). Add the dataset's own caps: the Malimg half is outdated
malware, and the uploader's 256×256 resize distorts texture.

**Threats to validity we accept.**
* The duplicate-group key is a *proxy* for provenance, not ground truth. A
  tighter dHash threshold would group more aggressively and lower our score; a
  looser one would raise it. We fixed it at 6/128 and report it.
* Some families have very few distinct binaries, so their per-class recall rests
  on a handful of independent samples and carries error bars a point estimate
  hides.
* 5 folds is the brief's requirement, not a statistically comfortable *n* (§5.1).
* We never recovered the original binaries, so §7.5's offset bands were never
  checked against actual PE section boundaries. That check is what would turn
  "the model attends to 12–19 % of the file" into "the model attends to the import
  table" — the explanation this entire literature claims and nobody, us included,
  has demonstrated on byte-plots.

## 9. Deliverables

| Item | Path |
|---|---|
| Final checkpoint | `models/Group00_MaleBin_best.pth` |
| Label map | `models/Group00_MaleBin_label_map.json` |
| Ablation table | `task3_ablation_ranked.csv` |
| CV scores | `task3_cv_scores.csv`, `task3_cv_mean_std.csv` |
| Significance tests | `task3_significance_tests.json`, `task3_significance_summary.csv` |
| Pillar-A verdict | `task3_pillarA.json` |
| XAI summary | `task3_xai_summary.json` |
