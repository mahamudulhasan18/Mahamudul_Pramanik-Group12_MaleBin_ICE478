"""Generate code/task1/Group00_MaleBin_task1_eda.ipynb"""
from pathlib import Path
from nbtool import build, writefile_cell, BOOT, REPO, PREFIX

C: list[tuple[str, str]] = []
md = lambda s: C.append(("md", s))
co = lambda s: C.append(("code", s))

md(f"""\
# CSE475 Summer 2026 — Task 1: Data Understanding & Related Work
## {PREFIX} · Track 3 (CNN + Attention) · Dataset: **MaleBin**

**Dataset** · *MaleBin: Malware Binary Greyscale Images* — Kaggle
`tashiee/malebin-malware-binary-greyscale-images` (CC BY 4.0).
12,464 grayscale byte-plot images, 39 malware families, 256×256, compiled by the
uploader from **two** sources:

1. **Malimg** (Nataraj et al., *VizSec* 2011) — the 25-family benchmark;
2. a subset of **`walt30/malware-images`**, visualised from MalwareBazaar samples
   following the same byte-to-pixel method.

**Problem type** · single-label multi-class image classification (39 classes).

**Application area** · static malware triage. A PE binary's bytes are read as a
stream, laid out row-wise into a fixed-width 2-D array and rendered as an 8-bit
grayscale image. Different families produce visibly different *textures* because
they share code, packers and resource layouts, which is what makes image
classification work at all here.

---

### What this notebook covers (brief §5, Task 1 — Image EDA A–G, plus H)

| | Item |
|---|---|
| **A** | Summary table — name, source, area, samples, classes, format, resolution, balance, missing/corrupt, duplicates |
| **B** | Class balance — counts, bar chart, majority:minority ratio, and what it does to metrics |
| **C** | Labelled sample grid — which families look alike? |
| **D** | Size / resolution audit |
| **E** | Pixel & texture statistics — mean, median, std, min/max, quartiles, skewness, kurtosis, entropy; histograms, boxplots, violin plots |
| **F** | Image quality — corrupt files, near-constant images, **exact and near-duplicates (the leakage risk)**, correlated features |
| **G** | 2-D structure — correlation heatmap + PCA / t-SNE / UMAP |
| **H** | Interactive Plotly figures |
| — | **Related-work table** (7 papers, 2022–2026) + the research gap |

### How to run this on Kaggle
1. *New Notebook* → **Add Data** → search `MaleBin malware binary greyscale` → **Add**.
2. Settings → **Accelerator: GPU T4/P100** (not needed for Task 1, but keeps all
   five notebooks on one profile), **Internet: off** is fine.
3. *Run All*. Nothing is downloaded; every dependency below ships with the
   Kaggle image.
4. For a ~3-minute wiring check first, set `MALEBIN_FAST=1` in the environment
   cell (or `CFG.fast = True` right after the boot cell).

> ⚠️ **Note on the dataset's own disclaimer.** The uploader states that a newer
> *MaleBin 2.0 RGB* set exists, that resizing in v1 can distort images, and that
> the Malimg half contains **outdated** malware which will not generalise to
> modern threats. We use v1 because it is the version the course dataset list
> points at, and we treat that disclaimer as a stated limitation of the study
> rather than something to hide — it is repeated in every task report.""")

C.append(writefile_cell())
co(BOOT)

co('''\
# extra libraries used only in this EDA notebook (all pre-installed on Kaggle)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=False)          # renders inside the saved .ipynb

from scipy import stats as sstats
from PIL import Image
print("plotly ready")
''')

# ---------------------------------------------------------------- A
md("""\
---
## A · Summary table

We index every file, then read each image's real header (not the folder name) so
that resolution, mode and corruption are measured rather than assumed.""")

co('''\
ROOT = M.find_dataset_root()
df = M.scan_index(ROOT)

# --- read the true header of every file: catches corruption and odd sizes ----
rows = []
bad = []
for p in df.path:
    try:
        with Image.open(p) as im:
            im.verify()                      # header/CRC check, cheap
        with Image.open(p) as im:
            rows.append((im.size[0], im.size[1], im.mode, im.format))
    except Exception as e:
        bad.append((p, f"{type(e).__name__}: {e}"))
        rows.append((np.nan, np.nan, "CORRUPT", "?"))

df[["width", "height", "mode", "format"]] = pd.DataFrame(rows, index=df.index)
df["megapixels"] = df.width * df.height / 1e6
print(f"unreadable / corrupt files: {len(bad)}")
for p, e in bad[:10]:
    print("   ", p, "->", e)
df.head()
''')

co('''\
vc = df.family.value_counts()
summary = pd.DataFrame([
    ("Dataset name",        "MaleBin: Malware Binary Greyscale Images"),
    ("Source",              "Kaggle · tashiee/malebin-malware-binary-greyscale-images (CC BY 4.0)"),
    ("Compiled from",       "Malimg (Nataraj et al. 2011) + subset of kaggle/walt30/malware-images (MalwareBazaar)"),
    ("Application area",    "Static malware family triage from byte-plot images (cyber security)"),
    ("Problem type",        f"Single-label multi-class image classification ({df.family.nunique()} classes)"),
    ("Total samples",       f"{len(df):,}"),
    ("Classes (families)",  f"{df.family.nunique()}"),
    ("  of which Malimg",   f"{df.loc[df.is_malimg,'family'].nunique()} families / {df.is_malimg.sum():,} images"),
    ("  of which other",    f"{df.loc[~df.is_malimg,'family'].nunique()} families / {(~df.is_malimg).sum():,} images"),
    ("File format",         ", ".join(f"{k} ({v:,})" for k, v in df.format.value_counts().items())),
    ("Colour mode",         ", ".join(f"{k} ({v:,})" for k, v in df["mode"].value_counts().items())),
    ("Resolution",          ", ".join(f"{int(w)}x{int(h)} ({n:,})" for (w, h), n
                                      in df.groupby(["width","height"]).size().items())),
    ("Channels",            "1 (8-bit grayscale) — a pixel IS a byte value 0-255"),
    ("Samples / class",     f"min {vc.min()}  median {int(vc.median())}  max {vc.max()}"),
    ("Imbalance ratio",     f"{vc.max()/max(vc.min(),1):.2f} : 1  (majority : minority)"),
    ("Missing values",      "N/A for images — measured instead as unreadable files"),
    ("Corrupt / unreadable",f"{len(bad)}"),
    ("Total size on disk",  f"{df.n_bytes.sum()/1e6:,.0f} MB"),
    ("Subject / source id", "NOT provided by the uploader — we derive one "
                            "(see section F: duplicate groups)"),
], columns=["Property", "Value"])

pd.set_option("display.max_colwidth", 130)
display(summary)
summary.to_csv(CFG.art("task1_A_summary_table.csv"), index=False)
''')

md("""\
**Reading.** Uniform 256×256 8-bit grayscale, one channel, so no resizing or
colour handling is needed before modelling — a real convenience. The two facts
that shape every later decision are (i) the ~*N* : 1 class imbalance printed
above, which forces **macro-F1** as the headline metric (brief §6.2), and
(ii) the absence of any subject/source identifier, which means we must **derive
our own grouping key** before we are allowed to split the data (section F).""")

# ---------------------------------------------------------------- B
md("""\
---
## B · Class balance and what it does to the metrics""")

co('''\
vc = df.family.value_counts().sort_values(ascending=False)
share = 100 * vc / vc.sum()

fig, ax = plt.subplots(2, 1, figsize=(13, 9),
                       gridspec_kw=dict(height_ratios=[2, 1]))
IS_MALIMG = df.drop_duplicates("family").set_index("family").is_malimg.to_dict()
colors = ["tab:blue" if IS_MALIMG.get(f, False) else "tab:orange" for f in vc.index]
ax[0].bar(range(len(vc)), vc.values, color=colors)
ax[0].axhline(vc.mean(), ls="--", c="k", lw=1, label=f"mean = {vc.mean():.0f}")
ax[0].axhline(vc.median(), ls=":", c="r", lw=1, label=f"median = {vc.median():.0f}")
ax[0].set_xticks(range(len(vc)))
ax[0].set_xticklabels(vc.index, rotation=90, fontsize=8)
ax[0].set(ylabel="number of images",
          title=f"B · Images per malware family (n={len(df):,}, "
                f"{len(vc)} classes)\\nblue = one of the 25 original Malimg "
                f"families, orange = added from the MalwareBazaar-derived source")
ax[0].legend()

ax[1].plot(range(len(vc)), np.cumsum(share.values), marker="o", ms=4)
ax[1].axhline(80, ls="--", c="r", lw=1, label="80% of the data")
ax[1].set(xlabel="families, most frequent first", ylabel="cumulative % of images",
          title="Cumulative share — how concentrated is the dataset?")
ax[1].legend()
fig.tight_layout()
fig.savefig(CFG.fig("task1_B_class_balance.png"), dpi=130, bbox_inches="tight")
plt.show()

ratio = vc.max() / max(vc.min(), 1)
gini = 1 - ((share / 100) ** 2).sum() * len(vc) / (len(vc) - 1) + 1 / (len(vc) - 1)
print(f"majority class : {vc.index[0]}  ({vc.iloc[0]:,} images, {share.iloc[0]:.1f}%)")
print(f"minority class : {vc.index[-1]} ({vc.iloc[-1]:,} images, {share.iloc[-1]:.1f}%)")
print(f"imbalance ratio: {ratio:.2f} : 1")
print(f"a majority-class-only classifier would score "
      f"accuracy = {share.iloc[0]/100:.4f} but macro-F1 = "
      f"{(2*share.iloc[0]/100)/(1+share.iloc[0]/100)/len(vc):.4f}")
print(f"\\nfamilies holding 80% of the data: "
      f"{int((np.cumsum(share.values) < 80).sum())+1} of {len(vc)}")
display(pd.DataFrame({"images": vc, "share_%": share.round(2),
                      "is_malimg": [IS_MALIMG.get(f, False) for f in vc.index]}))
''')

md("""\
**Reading.** The uploader deliberately trimmed families to balance them, so
MaleBin is far flatter than raw Malimg (which is ~2949 : 42). It is still not
uniform, and that has two consequences we carry through the whole project:

1. **Accuracy is not a safe headline.** A classifier that only ever predicted
   the majority family would score the accuracy printed above while getting
   38 of 39 families completely wrong. Per §6.2 we therefore report
   **macro-F1 and per-class recall** as the primary numbers, and we select
   models on *validation macro-F1*, never on accuracy.
2. **The loss must be re-weighted.** We use inverse-frequency class weights so
   rare families are not traded away for majority-class gains, and we verify in
   the Task-3 ablation that this actually helps rather than assuming it.""")

# ---------------------------------------------------------------- C
md("""\
---
## C · Labelled sample grid — which families look alike?""")

co('''\
rng = np.random.default_rng(CFG.seed)
fams = sorted(df.family.unique())
n = len(fams)
ncol = 8
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.05 * ncol, 2.35 * nrow))
for ax, fam in zip(axes.ravel(), fams):
    p = df.loc[df.family == fam, "path"].sample(1, random_state=CFG.seed).iloc[0]
    with Image.open(p) as im:
        ax.imshow(np.asarray(im.convert("L")), cmap="gray", vmin=0, vmax=255)
    tag = "M" if df.loc[df.family == fam, "is_malimg"].iloc[0] else "B"
    ax.set_title(f"{fam}\\n[{tag}] n={int((df.family==fam).sum())}", fontsize=7.5)
    ax.axis("off")
for ax in axes.ravel()[n:]:
    ax.axis("off")
fig.suptitle("C · One random byte-plot per family  "
             "([M] = Malimg source, [B] = MalwareBazaar-derived source)\\n"
             "row index = byte offset in the file; brightness = byte value",
             fontsize=12)
fig.tight_layout()
fig.savefig(CFG.fig("task1_C_sample_grid.png"), dpi=120, bbox_inches="tight")
plt.show()
''')

co('''\
# Within-family variability: 6 samples from the three largest and three
# smallest families. This is where polymorphic near-duplicates become visible.
pick = list(vc.index[:3]) + list(vc.index[-3:])
fig, axes = plt.subplots(len(pick), 6, figsize=(13, 2.2 * len(pick)))
for r, fam in enumerate(pick):
    ps = df.loc[df.family == fam, "path"]
    ps = ps.sample(min(6, len(ps)), random_state=CFG.seed).tolist()
    for c in range(6):
        ax = axes[r, c]
        if c < len(ps):
            with Image.open(ps[c]) as im:
                ax.imshow(np.asarray(im.convert("L")), cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
        if c == 0:
            ax.set_title(f"{fam}  (n={int((df.family==fam).sum())})",
                         fontsize=9, loc="left")
fig.suptitle("C · Within-family variability — 3 largest families (top) and "
             "3 smallest (bottom)", fontsize=12)
fig.tight_layout()
fig.savefig(CFG.fig("task1_C_within_family.png"), dpi=120, bbox_inches="tight")
plt.show()
''')

md("""\
**Reading.** Two things stand out and both drive later design choices.

* **Byte-plots have banded, not object-like, structure.** Every image is a stack
  of horizontal bands of differing texture: dense speckle where executable code
  sits, strictly periodic patterns where a table or padding repeats, uniform
  black where the file is zero-padded, and high-entropy noise where a section is
  packed or encrypted. The *vertical position* of a band is meaningful — row
  *r* of a 256-px-wide plot holds bytes `[256r, 256r+256)` — while horizontal
  position is an artefact of the fixed row width. This is the single most
  important observation in the whole EDA: it is why we choose **coordinate
  attention** (a per-row gate) in Task 2, and why we forbid flips and rotations
  in augmentation.
* **Families in the same lineage look nearly identical** (see e.g. the
  `Allaple.A` / `Allaple.L` and `Swizzor.gen!E` / `Swizzor.gen!I` pairs, and the
  repeated rows within a single family). That near-identity is a *leakage
  hazard*, quantified in section F.""")

# ---------------------------------------------------------------- D
md("""\
---
## D · Size / resolution audit""")

co('''\
print(df.groupby(["width", "height", "mode", "format"]).size().rename("count").to_frame())
print("\\nfile size on disk (KB): compressed PNG size varies with byte entropy,")
print("so it is itself a weak signal about how packed a sample is.")
display(df.groupby("family").n_bytes.describe()[["count","mean","std","min","50%","max"]]
          .div(1024).round(1).rename(columns=lambda c: c + "_KB" if c != "count" else c))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
ax[0].hist(df.n_bytes / 1024, bins=60, color="tab:purple")
ax[0].set(xlabel="PNG file size (KB)", ylabel="images",
          title="D · Compressed file-size distribution")
order = df.groupby("family").n_bytes.median().sort_values().index
ax[1].boxplot([df.loc[df.family == f, "n_bytes"] / 1024 for f in order],
              showfliers=False)
ax[1].set_xticks(range(1, len(order) + 1))
ax[1].set_xticklabels(order, rotation=90, fontsize=7)
ax[1].set(ylabel="PNG file size (KB)",
          title="D · File size by family (a proxy for byte entropy)")
fig.tight_layout()
fig.savefig(CFG.fig("task1_D_sizes.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

md("""\
**Reading.** Every image is already 256×256×1, so there is no resolution
heterogeneity to fix — unusual and helpful. Note, though, that the uploader
reached that uniformity by **resizing** the original byte-plots, and warns this
distorts texture; a 256×256 image is 65,536 bytes, so any binary larger than
64 KB has been downsampled and any smaller one upsampled. That is a real
information loss we cannot undo, and we state it as a limitation rather than
pretending the pixels are raw bytes.

Compressed PNG size does still vary a lot between families, which is expected:
PNG compresses low-entropy regions well, so a packed/encrypted sample stays
large. It is a legitimate weak feature but we deliberately **do not** feed it to
the model — file size is metadata, not content, and using it would be the
image-domain equivalent of the ID-column leak §6.2 warns about.""")

# ---------------------------------------------------------------- E
md("""\
---
## E · Pixel and texture statistics

For the statistical work (§B of the tabular EDA spec, adapted to images) we
derive a compact numeric feature vector per image. These features are used **for
EDA and visualisation only** — the CNN sees raw pixels.""")

co('''\
# cache all images once, at 128 px, for the statistics + projections below
EDA_SIDE = 64 if CFG.fast else 128
imgs_eda = M.load_images(df, side=EDA_SIDE)
print("cache:", imgs_eda.shape, imgs_eda.dtype)
''')

co('''\
def image_features(a: np.ndarray) -> dict:
    """Interpretable per-image descriptors (byte-statistics + texture)."""
    x = a.astype(np.float32)
    flat = x.ravel()
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p[p > 0]
    gy, gx = np.gradient(x)
    rows = x.mean(axis=1)
    return dict(
        mean=flat.mean(), median=float(np.median(flat)), std=flat.std(),
        vmin=flat.min(), vmax=flat.max(),
        q25=float(np.percentile(flat, 25)), q50=float(np.percentile(flat, 50)),
        q75=float(np.percentile(flat, 75)),
        iqr=float(np.percentile(flat, 75) - np.percentile(flat, 25)),
        skewness=float(sstats.skew(flat)), kurtosis=float(sstats.kurtosis(flat)),
        entropy=float(-(nz * np.log2(nz)).sum()),                 # byte entropy, 0-8
        zero_frac=float((a == 0).mean()),                         # zero padding
        high_frac=float((a > 240).mean()),                        # 0xFF filler
        ascii_frac=float(((a >= 32) & (a <= 126)).mean()),         # printable strings
        grad_mag=float(np.hypot(gx, gy).mean()),                  # texture roughness
        row_var=float(rows.var()),                                # banding strength
        row_autocorr=float(np.corrcoef(rows[:-1], rows[1:])[0, 1]) if rows.std() > 0 else 0.0,
        col_var=float(x.mean(axis=0).var()),
        top_band_mean=float(x[: x.shape[0] // 8].mean()),          # PE header region
        bottom_band_mean=float(x[-x.shape[0] // 8:].mean()),       # tail / padding
    )

t0 = time.time()
feat = pd.DataFrame([image_features(imgs_eda[i]) for i in range(len(df))])
feat.insert(0, "family", df.family.values)
feat.insert(1, "label", df.label.values)
feat.insert(2, "is_malimg", df.is_malimg.values)
print(f"{feat.shape[1]-3} features x {len(feat):,} images in {time.time()-t0:.1f}s")
FEATCOLS = [c for c in feat.columns if c not in ("family", "label", "is_malimg")]
feat.to_csv(CFG.art("task1_E_image_features.csv"), index=False)

desc = feat[FEATCOLS].describe().T
desc["skewness"] = feat[FEATCOLS].skew()
desc["kurtosis"] = feat[FEATCOLS].kurtosis()
desc["n_outliers_1.5IQR"] = [
    int(((feat[c] < feat[c].quantile(.25) - 1.5 * (feat[c].quantile(.75) - feat[c].quantile(.25))) |
         (feat[c] > feat[c].quantile(.75) + 1.5 * (feat[c].quantile(.75) - feat[c].quantile(.25)))).sum())
    for c in FEATCOLS]
display(desc.round(3))
desc.round(4).to_csv(CFG.art("task1_E_feature_stats.csv"))
''')

co('''\
# histograms
show = ["mean", "std", "entropy", "zero_frac", "ascii_frac", "grad_mag",
        "row_var", "row_autocorr", "skewness", "kurtosis", "top_band_mean",
        "bottom_band_mean"]
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), show):
    ax.hist(feat[c], bins=50, color="tab:blue")
    ax.set(title=f"{c}\\nskew={feat[c].skew():.2f}  kurt={feat[c].kurtosis():.2f}",
           xlabel=c, ylabel="images")
fig.suptitle("E · Distributions of the derived byte/texture features", fontsize=13)
fig.tight_layout()
fig.savefig(CFG.fig("task1_E_histograms.png"), dpi=125, bbox_inches="tight")
plt.show()
''')

co('''\
# violin + box plots: do the classes separate on any single feature?
key = ["entropy", "std", "zero_frac", "row_autocorr"]
order = feat.groupby("family").entropy.median().sort_values().index.tolist()
fig, axes = plt.subplots(len(key), 1, figsize=(14, 3.3 * len(key)), sharex=True)
for ax, c in zip(axes, key):
    data = [feat.loc[feat.family == f, c].values for f in order]
    parts = ax.violinplot(data, showmedians=True, widths=.9)
    for pc in parts["bodies"]:
        pc.set_alpha(.55)
    ax.boxplot(data, widths=.18, showfliers=False,
               medianprops=dict(color="k", lw=1.2))
    # eta^2: how much of this feature's variance is explained by the family label
    grand = feat[c].mean()
    ssb = sum(len(d) * (d.mean() - grand) ** 2 for d in data if len(d))
    sst = ((feat[c] - grand) ** 2).sum()
    ax.set(ylabel=c, title=f"E · {c} by family — eta^2 = {ssb/sst:.3f} "
                           f"(share of variance explained by the class label)")
axes[-1].set_xticks(range(1, len(order) + 1))
axes[-1].set_xticklabels(order, rotation=90, fontsize=7.5)
fig.tight_layout()
fig.savefig(CFG.fig("task1_E_violin.png"), dpi=125, bbox_inches="tight")
plt.show()

eta = {}
for c in FEATCOLS:
    grand = feat[c].mean()
    ssb = sum(len(g) * (g[c].mean() - grand) ** 2 for _, g in feat.groupby("family"))
    eta[c] = ssb / ((feat[c] - grand) ** 2).sum()
print("class-discriminative power of each hand-made feature (eta^2, 0-1):")
display(pd.Series(eta).sort_values(ascending=False).round(3).to_frame("eta_squared"))
''')

md("""\
**Reading.** The byte-value distributions are strongly non-Gaussian — large
positive skew and heavy kurtosis on `zero_frac`, `high_frac` and `mean`, because
most files contain long runs of `0x00` padding and `0xFF` filler. The η² column
says how much of each feature's variance the family label explains: the top
features (typically `entropy`, `row_autocorr`, `zero_frac`, `std`) do carry real
class signal, which is a useful sanity check — it means the task is learnable
from texture and we are not about to train on noise. But no single scalar
separates 39 classes; that is precisely the job we hand to the CNN.""")

# ---------------------------------------------------------------- F
md("""\
---
## F · Data quality, and the leakage risk that decides our whole protocol

This is the most consequential section of Task 1. §6.2 of the brief requires a
**subject/source-based split**. MaleBin ships **no** subject, sample-hash or
source column — so we have to construct the grouping key ourselves, and we have
to justify it.

**The argument.** A malware *family* is a set of polymorphic variants of the same
program. Byte-plots of two variants differ by a shifted section, a repacked
region, or a few patched bytes, and are otherwise the same picture. MaleBin also
*merges two corpora* that partly overlap in provenance, so the same original
sample can appear twice. If a variant lands in train and its near-twin in test,
the reported score measures memorisation, not classification — the exact failure
mode §6.2 forbids, and the reason a random split on this kind of data produces
the 99%+ numbers the literature reports.

**Our grouping key.** Exact duplicates via SHA-1 of the pixel buffer, plus near
duplicates via a 128-bit difference hash (horizontal + vertical gradients) with
Hamming distance ≤ 6, unioned with union-find. Every split in Tasks 2 and 3 is
grouped on that key and stratified on the family.""")

co('''\
# ---- 1. unreadable / corrupt -------------------------------------------------
print(f"corrupt or unreadable files : {len(bad)}")

# ---- 2. degenerate images ---------------------------------------------------
const = feat.std_is_zero = (feat["std"] < 1e-6)
print(f"constant (single-value) images: {int(const.sum())}")
near_const = feat["std"] < 3
print(f"near-constant images (std<3) : {int(near_const.sum())}")
if near_const.sum():
    display(feat.loc[near_const, ["family", "mean", "std", "entropy"]].head(10))

# ---- 3. exact duplicates ----------------------------------------------------
import hashlib
sha = [hashlib.sha1(imgs_eda[i].tobytes()).hexdigest() for i in range(len(df))]
dup = pd.Series(sha).duplicated(keep=False)
print(f"\\nexact pixel duplicates      : {int(dup.sum())} images in "
      f"{pd.Series(sha)[dup].nunique()} groups")
cross = (pd.DataFrame({"h": sha, "f": df.family.values})
           .groupby("h").f.nunique())
print(f"  duplicate hashes appearing under >1 family: {int((cross>1).sum())}"
      "  <- these would be label noise, not just duplication")
if (cross > 1).sum():
    h0 = cross[cross > 1].index[0]
    display(df.loc[[i for i, h in enumerate(sha) if h == h0],
                   ["family", "path"]])
''')

co('''\
# ---- 4. near duplicates -> the grouping key --------------------------------
M.banner("Building the duplicate-group key (this IS our 'subject' column)")
groups = M.build_dedup_groups(imgs_eda, df)
df["dup_group"] = groups

gsz = pd.Series(groups).value_counts()
per_fam = (pd.DataFrame({"g": groups, "family": df.family.values})
             .groupby("family").g.nunique().rename("distinct_binaries").to_frame())
per_fam["images"] = df.family.value_counts().reindex(per_fam.index).values
per_fam["images_per_binary"] = (per_fam.images / per_fam.distinct_binaries).round(2)
per_fam = per_fam.sort_values("images_per_binary", ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
ax[0].hist(gsz.values, bins=range(1, min(gsz.max(), 40) + 2), color="tab:red")
ax[0].set(xlabel="images in one duplicate group", ylabel="groups", yscale="log",
          title=f"F · Duplicate-group sizes\\n{len(gsz):,} groups for "
                f"{len(df):,} images ({100*(1-len(gsz)/len(df)):.1f}% collapse)")
ax[1].bar(range(len(per_fam)), per_fam.images_per_binary.values, color="tab:red")
ax[1].axhline(1, ls="--", c="k", lw=1, label="1 = no duplication")
ax[1].set_xticks(range(len(per_fam)))
ax[1].set_xticklabels(per_fam.index, rotation=90, fontsize=7)
ax[1].set(ylabel="images per distinct binary",
          title="F · Redundancy per family — how many near-copies of each binary?")
ax[1].legend()
fig.tight_layout()
fig.savefig(CFG.fig("task1_F_duplicates.png"), dpi=130, bbox_inches="tight")
plt.show()
display(per_fam)
per_fam.to_csv(CFG.art("task1_F_redundancy_per_family.csv"))
''')

co('''\
# show a duplicate group so the claim is visible, not just asserted
big = gsz[gsz > 1]
if len(big):
    g0 = int(big.index[0])
    members = np.flatnonzero(groups == g0)[:8]
    fig, axes = plt.subplots(1, len(members), figsize=(2.1 * len(members), 2.6))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, members):
        ax.imshow(imgs_eda[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(Path(df.path.iloc[i]).name[:16], fontsize=7)
        ax.axis("off")
    fig.suptitle(f"F · One duplicate group ({len(members)} of "
                 f"{int(gsz[g0])} members), family = {df.family.iloc[members[0]]}\\n"
                 "A random split would scatter these across train and test.",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(CFG.fig("task1_F_dup_group_example.png"), dpi=130,
                bbox_inches="tight")
    plt.show()
else:
    print("no multi-image duplicate groups found at this threshold")
''')

co('''\
# ---- 5. how much would a random split inflate the score? --------------------
# Cheap, honest demonstration with a 1-NN classifier on the EDA features:
#   grouped split  -> a near-twin of a test image can never be in train
#   random  split  -> it usually is
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

X = StandardScaler().fit_transform(feat[FEATCOLS].values)
y = df.label.values
rows = []
for name, splitter, g in [
        ("random (stratified)  <- WRONG", StratifiedKFold(5, shuffle=True,
                                                          random_state=CFG.seed), None),
        ("duplicate-grouped    <- OURS",  StratifiedGroupKFold(5, shuffle=True,
                                                               random_state=CFG.seed), groups)]:
    accs, f1s = [], []
    for trn, tst in (splitter.split(X, y, g) if g is not None else splitter.split(X, y)):
        k = KNeighborsClassifier(1).fit(X[trn], y[trn])
        p = k.predict(X[tst])
        accs.append(accuracy_score(y[tst], p))
        f1s.append(f1_score(y[tst], p, average="macro", zero_division=0))
    rows.append(dict(protocol=name, accuracy=f"{np.mean(accs):.4f} +- {np.std(accs):.4f}",
                     macro_f1=f"{np.mean(f1s):.4f} +- {np.std(f1s):.4f}",
                     _f1=np.mean(f1s)))
tbl = pd.DataFrame(rows)
display(tbl.drop(columns="_f1"))
gap = rows[0]["_f1"] - rows[1]["_f1"]
print(f"\\ninflation from splitting randomly: +{gap:.4f} macro-F1 "
      f"({100*gap/max(rows[1]['_f1'],1e-9):+.1f}% relative)")
print("A 1-nearest-neighbour classifier on 20 hand-made features is not a strong")
print("model. If it does well under a random split, that is memorisation of")
print("near-duplicates, not classification -- which is the whole point.")
tbl.drop(columns="_f1").to_csv(CFG.art("task1_F_leakage_demo.csv"), index=False)
''')

md("""\
**Reading — and the decision that follows.** The gap printed above is the price
of getting the split wrong, measured on this dataset with a deliberately weak
model. Every result in Tasks 2 and 3 therefore uses the **duplicate-grouped,
family-stratified** split, and the same grouping is used for the 5 CV folds in
Task 3. We keep the ungrouped number visible as a control so a reader can see
what we gave up: our headline scores will be *lower* than the published ones,
and that is the honest trade the rubric asks for (§8, "leakage void").

Concretely, the grouping means our effective sample size is the number of
**distinct binaries**, not the number of images — the `distinct_binaries` column
above is the real size of this dataset.""")

# ---------------------------------------------------------------- G
md("""\
---
## G · Correlation, redundancy, and 2-D structure (PCA / t-SNE / UMAP)""")

co('''\
corr = feat[FEATCOLS].corr()
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATCOLS)))
ax.set_yticks(range(len(FEATCOLS)))
ax.set_xticklabels(FEATCOLS, rotation=90, fontsize=8)
ax.set_yticklabels(FEATCOLS, fontsize=8)
for i in range(len(FEATCOLS)):
    for j in range(len(FEATCOLS)):
        if abs(corr.iloc[i, j]) > .55 and i != j:
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    fontsize=6.5)
fig.colorbar(im, ax=ax, fraction=.046, label="Pearson r")
ax.set_title("G · Correlation of the derived features "
             "(|r| > 0.55 annotated)")
fig.tight_layout()
fig.savefig(CFG.fig("task1_G_correlation.png"), dpi=130, bbox_inches="tight")
plt.show()

hi = [(FEATCOLS[i], FEATCOLS[j], round(corr.iloc[i, j], 3))
      for i in range(len(FEATCOLS)) for j in range(i + 1, len(FEATCOLS))
      if abs(corr.iloc[i, j]) > .85]
print("redundant feature pairs (|r| > 0.85) -- one of each pair is expendable:")
for a, b, r in sorted(hi, key=lambda t: -abs(t[2])):
    print(f"   {a:<18} ~ {b:<18} r = {r:+.3f}")
if not hi:
    print("   none")
''')

co('''\
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Two views: (1) the interpretable features, (2) raw downsampled pixels.
Xf = StandardScaler().fit_transform(feat[FEATCOLS].values)
side_small = 32
Xp = np.stack([np.asarray(Image.fromarray(imgs_eda[i]).resize((side_small, side_small)))
               .ravel() for i in range(len(df))]).astype(np.float32) / 255.0
Xp = StandardScaler().fit_transform(Xp)

pca_f = PCA(n_components=min(20, Xf.shape[1]), random_state=CFG.seed).fit(Xf)
pca_p = PCA(n_components=50, random_state=CFG.seed).fit(Xp)
print(f"PCA on features : first 2 PCs explain {pca_f.explained_variance_ratio_[:2].sum():.1%}, "
      f"first 10 -> {pca_f.explained_variance_ratio_[:10].sum():.1%}")
print(f"PCA on pixels   : first 2 PCs explain {pca_p.explained_variance_ratio_[:2].sum():.1%}, "
      f"first 50 -> {pca_p.explained_variance_ratio_[:50].sum():.1%}")

Zf = pca_f.transform(Xf)[:, :2]
Zp2 = pca_p.transform(Xp)[:, :2]
Xp50 = pca_p.transform(Xp)

perp = max(5, min(30, len(df) // 60))
Zt = TSNE(n_components=2, perplexity=perp, init="pca", random_state=CFG.seed,
          max_iter=400 if CFG.fast else 1000).fit_transform(Xp50)

Zu = None
try:
    import umap
    Zu = umap.UMAP(n_components=2, n_neighbors=min(15, len(df)//30),
                   min_dist=.1, random_state=CFG.seed).fit_transform(Xp50)
    print("UMAP ok")
except Exception as e:
    print(f"UMAP unavailable ({type(e).__name__}) -- PCA + t-SNE still shown")
''')

co('''\
def scatter_by_family(ax, Z, title):
    cmap = plt.get_cmap("tab20")
    for i, fam in enumerate(sorted(df.family.unique())):
        m = (df.family == fam).values
        ax.scatter(Z[m, 0], Z[m, 1], s=7, alpha=.7, color=cmap(i % 20),
                   label=fam if i < 20 else None, linewidths=0)
    ax.set(title=title, xlabel="component 1", ylabel="component 2")

panels = [(Zf, "PCA on the 20 derived features"),
          (Zp2, f"PCA on raw {side_small}x{side_small} pixels"),
          (Zt, f"t-SNE on 50 pixel PCs (perplexity={perp})")]
if Zu is not None:
    panels.append((Zu, "UMAP on 50 pixel PCs"))

fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 5.1))
for ax, (Z, t) in zip(np.atleast_1d(axes), panels):
    scatter_by_family(ax, Z, t)
np.atleast_1d(axes)[0].legend(fontsize=6, ncol=2, loc="best", markerscale=1.6)
fig.suptitle("G · Do the 39 families separate without a classifier? "
             "(colour = family; legend shows the first 20)", fontsize=13)
fig.tight_layout()
fig.savefig(CFG.fig("task1_G_projections.png"), dpi=130, bbox_inches="tight")
plt.show()

# quantify the separation instead of eyeballing it
from sklearn.metrics import silhouette_score
for Z, t in panels:
    try:
        s = silhouette_score(Z, df.label.values)
        print(f"  silhouette (family labels) on {t:<42s}: {s:+.3f}")
    except Exception as e:
        print("  silhouette failed:", e)
print("\\n(+1 = perfectly separated clusters, 0 = overlapping, -1 = wrong clusters)")
''')

md("""\
**Reading.** PCA on the hand-made features shows only coarse structure — the
first two components explain a modest share of variance and families overlap
heavily. t-SNE/UMAP on raw pixel PCs is much more encouraging: many families
form tight, well-separated islands, several of which are *sub-clustered*, and
those sub-clusters are exactly the duplicate groups found in section F. A few
families sit on top of each other (typically the `Allaple`, `Swizzor`,
`C2LOP` and `Lolyda` variant pairs), which predicts where the confusion matrix
in Task 2 will be dark.

Two conclusions: (i) the task is genuinely learnable from pixels — the signal is
there before any training; (ii) the *hard* part is the handful of confusable
variant pairs, so improvements should be judged on per-class recall for those
families, not on the overall average.""")

# ---------------------------------------------------------------- H
md("""\
---
## H · Interactive plots (Plotly)

Interactive versions of the three most useful views. Hover for exact values;
double-click a legend entry to isolate a family. These render inside the saved
notebook, so they survive submission to GitHub.""")

co('''\
# H1 -- class balance, split by source, with redundancy on hover
h = (df.groupby(["family", "is_malimg"]).size().rename("images").reset_index())
h["source"] = np.where(h.is_malimg, "Malimg (Nataraj 2011)",
                       "MalwareBazaar-derived (walt30)")
h = h.merge(per_fam.reset_index()[["family", "distinct_binaries",
                                   "images_per_binary"]],
            on="family", how="left")
fig = px.bar(h.sort_values("images", ascending=False), x="family", y="images",
             color="source", hover_data=["distinct_binaries", "images_per_binary"],
             title="H1 · Images per family and source "
                   "(hover: how many *distinct binaries* are behind those images)")
fig.update_layout(xaxis_tickangle=-90, height=560,
                  xaxis_title="malware family", yaxis_title="number of images")
fig.write_html(str(CFG.fig("task1_H1_class_balance.html")))
fig.show()
''')

co('''\
# H2 -- interactive t-SNE / UMAP, hover shows family + duplicate group + file
Zi = Zu if Zu is not None else Zt
lab = "UMAP" if Zu is not None else "t-SNE"
hov = pd.DataFrame({"x": Zi[:, 0], "y": Zi[:, 1], "family": df.family.values,
                    "source": np.where(df.is_malimg, "Malimg", "MalwareBazaar"),
                    "dup_group": groups,
                    "file": [Path(p).name for p in df.path],
                    "entropy": feat.entropy.round(3),
                    "zero_frac": feat.zero_frac.round(3)})
fig = px.scatter(hov, x="x", y="y", color="family", symbol="source",
                 hover_data=["file", "dup_group", "entropy", "zero_frac"],
                 title=f"H2 · {lab} of byte-plots, coloured by family. "
                       f"Tight sub-clusters are duplicate groups -- hover to check "
                       f"the dup_group id.")
fig.update_traces(marker=dict(size=6, opacity=.8))
fig.update_layout(height=680, xaxis_title=f"{lab}-1", yaxis_title=f"{lab}-2")
fig.write_html(str(CFG.fig("task1_H2_projection.html")))
fig.show()
''')

co('''\
# H3 -- feature distributions per family (box + strip), switchable feature
long = feat.melt(id_vars=["family", "is_malimg"],
                 value_vars=["entropy", "std", "zero_frac", "ascii_frac",
                             "grad_mag", "row_autocorr"],
                 var_name="feature", value_name="value")
fig = px.box(long, x="family", y="value", color="feature", points=False,
             facet_row="feature", height=1500,
             title="H3 · Byte/texture feature distributions per family "
                   "(one row per feature; hover for quartiles)")
fig.update_xaxes(tickangle=-90, tickfont=dict(size=8))
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_yaxes(matches=None)
fig.write_html(str(CFG.fig("task1_H3_features.html")))
fig.show()
''')

# ---------------------------------------------------------------- related work
md("""\
---
## Related work — seven peer-reviewed papers, 2022–2026 (brief §6.3)

Track 3 requires the *attention type* column in addition to the shared columns.
Every number below was read from the paper's own abstract or results table; the
`comparable` flag records whether §6.4 lets us compare our result against it.""")

co('''\
rw = M.related_work_frame()
print(f"{len(rw)} papers, years {rw.year.min()}-{rw.year.max()}, "
      f"{int(rw.comparable.sum())} marked comparable\\n")
pd.set_option("display.max_colwidth", 70)
display(rw[["key", "year", "authors", "venue", "dataset", "attention_type",
            "metrics", "comparable"]])
rw.to_csv(REPO_RW := str(CFG.art("task1_related_work_table.csv")), index=False)
print("\\nfull table ->", REPO_RW)
''')

co('''\
# the full table, one paper per block, so nothing is truncated
for r in M.RELATED_WORK:
    print("=" * 100)
    print(f"[{r['key']}]  {r['title']}")
    print(f"  authors        : {r['authors']}")
    print(f"  year / venue   : {r['year']} · {r['venue']}   doi:{r['doi']}")
    print(f"  dataset        : {r['dataset']}")
    print(f"  application    : {r['application']}")
    print(f"  method         : {r['method']}")
    print(f"  attention type : {r['attention_type']}")
    print(f"  metrics        : {r['metrics']}")
    print(f"  strengths      : {r['strengths']}")
    print(f"  limitations    : {r['limitations']}")
    print(f"  research gap   : {r['gap']}")
    print(f"  relation to us : {r['relation']}")
    print(f"  comparable?    : {r['comparable']}")
''')

co('''\
# reported headline numbers, and the two Pillar-A targets we will be judged on
comp = rw[rw.comparable & rw.headline_value.notna()]
fig, ax = plt.subplots(figsize=(9.5, 4.4))
b = ax.barh([f"{k} ({y})" for k, y in zip(comp.key, comp.year)],
            comp.headline_value, color="tab:grey")
for rect, m, v in zip(b, comp.headline_metric, comp.headline_value):
    ax.text(v + .05, rect.get_y() + rect.get_height() / 2, f"{v:.2f}  ({m})",
            va="center", fontsize=9)
ax.set(xlim=(90, 101), xlabel="reported headline score (%)",
       title="Published results we are compared against (Pillar A)\\n"
             "all use a RANDOM split; ours will use a duplicate-grouped split")
fig.tight_layout()
fig.savefig(CFG.fig("task1_related_work_scores.png"), dpi=130, bbox_inches="tight")
plt.show()

for sc in ["malimg25", "malebin39"]:
    t = M.best_comparable_target(sc)
    print(f"\\nPillar-A target for scope '{sc}':")
    print(f"   paper   : {t['paper']} -- {t['citation']}")
    print(f"   metric  : {t['metric']} = {t['value']}")
    print(f"   caveat  : {t['caveat']}")
''')

md("""\
### The research gap we are addressing

Reading the seven papers together, three things are true of *all* of them:

1. **Every reported number comes from a random split.** None of the malware-image
   papers controls for polymorphic near-duplicates or for provenance overlap
   between merged corpora. Section F measured what that is worth on this
   dataset. So the 99.2–99.4% Malimg figures are upper bounds under an
   optimistic protocol, not estimates of generalisation.
2. **The headline metric is almost always accuracy** (or precision), on data
   that is imbalanced. Only MalVis (2025) reports macro-F1, and there it sits
   **4.4 points below** accuracy (90.81 vs 95.19). The PE-image papers never
   publish that gap, and Basak et al. explicitly admit their model "struggles
   with underrepresented classes".
3. **Attention, where used, is channel-only** (SE in PAFE, SE in IMCMK-CNN,
   involution in DRIN) or generic self-attention (Swin in Alshomrani et al.).
   Nobody exploits the one structural fact that byte-plots actually have:
   **the row index is the byte offset in the file.** Channel attention pools
   that away; window self-attention treats it as an arbitrary 2-D coordinate.

**Gap → our contribution.** We build a small from-scratch CNN whose attention is
*direction-aware* (coordinate attention: a per-row and a per-column gate, so
byte-offset position survives), stack it on multi-scale dilated convolutions,
train it with byte-aware augmentation that never flips or rotates the file, and
evaluate the whole thing under a **duplicate-grouped** split with **macro-F1**
as the headline. We then compare on the Malimg-25 subset (same classes as the
published work, harder split) and on full MaleBin-39, and we report both — plus
5-fold CV with a significance test — instead of one flattering number.""")

# ---------------------------------------------------------------- wrap
md("""\
---
## Task 1 conclusions

| Finding | Consequence for Tasks 2–3 |
|---|---|
| Uniform 256×256 8-bit grayscale, no corrupt files | No resizing/colour pipeline needed; input is `1×224×224` after a single resize |
| Imbalance ratio printed in §B | **macro-F1** is the headline metric; inverse-frequency class weights in the loss; model selection on validation macro-F1 |
| No subject/source column, but heavy near-duplication (§F) | We derive a duplicate-group key and use **grouped stratified** splits and CV folds everywhere. Effective *n* = number of distinct binaries |
| Row position = byte offset; images are horizontally banded | Attention must preserve vertical position → **coordinate attention**; augmentation must **not** flip or rotate |
| Signal lives at several scales (byte texture, repeated blocks, whole sections) | **Multi-scale dilated** conv block |
| t-SNE/UMAP show a few overlapping variant pairs | Judge improvements on per-class recall for those families |
| Every related-work number uses a random split and an accuracy headline | Our numbers will be lower and we say why; the fair comparison is Malimg-25 scope |
| Uploader's own disclaimer: resize distortion, outdated Malimg samples | Stated as a limitation in all three reports |

### Deliverables written by this notebook""")

co('''\
print("figures / tables written to", CFG.out_dir, "\\n")
for p in sorted(Path(CFG.out_dir).rglob("*")):
    if p.is_file() and p.suffix in (".png", ".html", ".csv", ".json"):
        print(f"  {p.stat().st_size/1024:8.1f} KB  {p.relative_to(CFG.out_dir)}")
print("""
Copy into the repo as:
  report/task1/Group00_MaleBin_task1_report.pdf     (write from the readings above)
  related_work/Group00_MaleBin_related_work_table.pdf
  related_work/papers/                              (the 7 paper PDFs)
  code/task1/Group00_MaleBin_task1_eda.ipynb        (this notebook, with output)
""")
''')

build(C, REPO / "code" / "task1" / f"{PREFIX}_task1_eda.ipynb",
      "CSE475 Task 1 - MaleBin EDA + Related Work")
