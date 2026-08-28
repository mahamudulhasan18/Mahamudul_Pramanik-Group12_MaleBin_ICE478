# `REAL_RUN` — the CPU run on the real MaleBin dataset

This documents the run that produced `figures/`, `models/` and the three task
reports, plus the untracked `artifacts/` folder of metrics and prediction
arrays. **These numbers come from the real 12,464-image MaleBin dataset**, not a
synthetic stand-in.

They are still not the numbers the README's method section describes, and the
difference matters — read §3 before quoting anything.

---

## 1. How it was run

```bash
python code/common/run_notebooks.py \
    --profile cpu2h \
    --data-root /path/to/malebin/extracted
```

`run_notebooks.py` executes all five notebooks in one working directory, so the
artifacts chain exactly as they would inside a single Kaggle session. Every
budget knob is passed in through environment variables, so **the notebooks
themselves are byte-for-byte the ones that ship to Kaggle** — nothing was edited
to make this run fit.

### Figures are not embedded in the notebooks

The run sets `MALEBIN_NO_INLINE=1`. In that mode `plt.show()` writes the figure
to `figures/`, prints its filename, and closes it instead of embedding a
base64 PNG in the notebook. Plotly's `.show()` does the same for its HTML.

The executed notebooks therefore keep **every table, metric and printed line**
but none of the image payload. The Task-1 notebook went from **12 MB to under
0.5 MB**. Each figure is a real file in `figures/`, named for the section that
produced it.

Leave `MALEBIN_NO_INLINE` unset and the notebooks display plots inline exactly
as before — which is what you want when committing on Kaggle.

---

## 2. Hardware and budget

No GPU was available: `torch.cuda.is_available()` is `False`, 12 CPU cores.
Measured throughput on the real data, 8,698 training images at 64 px:

| model | img/s | min/epoch |
|---|---|---|
| ByteAttnNet | 75.8 | 1.91 |
| SimpleCNN | 99.4 | 1.46 |
| ResNet50 | 31.0 | 4.68 |
| DenseNet121 | 31.4 | 4.62 |
| EfficientNet-B0 | 36.2 | 4.01 |

At the README's full specification (224 px, 25 epochs, 5 folds, 16 ablation
variants, 4 baselines) the chain needs roughly **470 ByteAttnNet epoch-
equivalents plus the baselines — about nine days of CPU**. The `cpu2h` profile
buys a complete, real-data run of all five notebooks inside two hours by
spending resolution and epochs:

| knob | full spec | `cpu2h` |
|---|---|---|
| input size | 224 px | **64 px** |
| cache size | 256 px | 96 px |
| epochs — baselines | 25 | **2** |
| epochs — proposed model | 25 | **5** |
| epochs — Task-3 final | 25 | **4** |
| epochs — per ablation variant | 12 | **2** |
| CV folds × epochs | 5 × 25 | **2 × 2** |
| ablation variants | 16 | **8** (attention + augmentation) |
| baselines | SimpleCNN, ResNet50, DenseNet121, EfficientNet-B0 | SimpleCNN, ResNet50, MobileNetV3-Small |

Everything else is untouched: the full 12,464 images, all 39 families, the same
seed, and the same duplicate-grouped, family-stratified split.

---

## 3. What these numbers can and cannot support

**They can support**, because the data, the split and the leakage control are
the real thing:

* the dataset findings in Task 1, which do not depend on training at all;
* the *relative* ordering of models trained under identical budgets;
* that the pipeline runs end-to-end on real data and the artifacts chain.

**They cannot support** any absolute claim:

* **Absolute F1 / accuracy are floored by the budget, not the architecture.**
  Two to five epochs at 64 px is a fraction of the training these models need.
  Do not compare these figures to a published number.
* **The comparison to related work is not meaningful here.** Those papers train
  to convergence at full resolution.
* **The cross-validation is underpowered by construction.** Two folds cannot
  produce a Wilcoxon *p* below 0.5, so the CV section demonstrates the machinery
  and nothing more. The README already notes that even 5 folds cannot reach
  α = 0.05; at 2 folds there is no test to speak of. The **McNemar test on the
  2,422-sample test set** is the only statistic here with real power.
* **Ablation deltas at 2 epochs measure early-training behaviour**, which is not
  the same thing as final performance. Treat the ordering as a hypothesis.

To produce quotable numbers, run the same notebooks on Kaggle with a GPU at the
default settings (`README.md` §4) — no code changes needed, just leave the
`MALEBIN_*` environment variables unset.

---

## 4. Bugs this run found

Running against real data surfaced four defects that the synthetic
`verification_run/` could not have caught. All four are fixed in
`code/common/malebin_common.py` and the notebook generators.

### 4.1 Two MaleBin families are the same 500 images

`RecordBreaker` and `RedLineStealer` have **identical SHA-1 hash sets** at native
256 px resolution: 500 images each, 499 unique hashes, and every one shared.
They are the same folder under two labels.

`build_dedup_groups()` asserted that no duplicate group spans two families, so
**the notebooks aborted at the leakage-control cell on the real dataset**. The
assertion was the bug, not the grouping: near-duplicate search is per-family, so
the only way to trigger it is an exact pixel match carrying two labels — and
unioning those images is exactly right, because otherwise one copy could train
while its twin was tested. The assertion is now a reported data-quality finding
(Task-1 §F.5, `artifacts/..._task1_F_cross_family_groups.csv`).

This caps what any model can score: two families that share their images cannot
both be recalled, so roughly one class' worth of macro-F1 — about 0.026 — is
unreachable by construction.

### 4.2 A NaN that took out the whole of Task-1 §G

`row_autocorr` guarded `np.corrcoef` with `rows.std() > 0`, which is not
sufficient: if every row is constant except the first or the last, one of the
two shifted halves still has zero variance and `corrcoef` returns NaN. **Four
real MaleBin images do exactly that.** A single NaN propagated through
`StandardScaler` into PCA, t-SNE, UMAP and the kNN check, failing four cells with
an opaque `Input X contains NaN`. Fixed with a `_safe_corr` helper, plus an
explicit sanitation pass over the feature frame that repairs any non-finite value
with the column median and prints what it repaired.

### 4.3 Parallel image decode could fork-bomb on Windows

`load_images()` used `ProcessPoolExecutor`. On Windows "spawn" that re-imports
`__main__`, which either fails inside a Jupyter kernel or recursively re-runs the
calling script. Switched to `ThreadPoolExecutor` — PIL releases the GIL during
decode and resize, so it keeps the speed-up (12,464 images in ~5 s) and works
everywhere.

### 4.4 `groupby(...).apply()` drops the grouping column on pandas 3

The `max_per_class` subsample path lost the `family` column, so every downstream
cell failed with `'DataFrame' object has no attribute 'family'`. Replaced with
`groupby(...).head(n)`. This only affects the smoke-test path, not a full run.

---

## 5. New environment knobs

Added so a run can be re-budgeted without editing a notebook cell. All are
unset by default, so Kaggle behaviour is unchanged.

| variable | effect |
|---|---|
| `MALEBIN_NO_INLINE` | figures to disk, never embedded in the notebook |
| `MALEBIN_IMG_SIZE`, `MALEBIN_CACHE_SIZE` | resolution |
| `MALEBIN_EPOCHS`, `MALEBIN_BATCH`, `MALEBIN_PATIENCE`, `MALEBIN_LR` | training |
| `MALEBIN_FOLDS`, `MALEBIN_CV_EPOCHS` | cross-validation |
| `MALEBIN_ABLATION_EPOCHS`, `MALEBIN_ABLATION_GROUPS` | ablation scope |
| `MALEBIN_RUN_ABLATION`, `MALEBIN_RUN_FINAL`, `MALEBIN_RUN_CV` | section switches |
| `MALEBIN_BASELINES` | comma-separated baseline list |
| `MALEBIN_OUT_DIR`, `MALEBIN_EVAL_SCOPE`, `MALEBIN_MAX_PER_CLASS`, `MALEBIN_WORKERS`, `MALEBIN_AMP` | misc |
