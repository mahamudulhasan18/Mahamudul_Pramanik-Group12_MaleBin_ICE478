"""
malebin_common.py
=================
Shared utilities for the CSE475 (Summer 2026) course project.

Track      : 3 -- CNN + Attention
Dataset    : MaleBin: Malware Binary Greyscale Images
             (kaggle: tashiee/malebin-malware-binary-greyscale-images)
             12,464 grayscale byte-plot images, 39 malware families, 256x256,
             compiled from (a) Malimg (Nataraj et al., 2011) and
                           (b) a MalwareBazaar-derived subset (kaggle: walt30).

This single module is embedded (via %%writefile) at the top of every task
notebook so that each notebook is self-contained and re-runnable on Kaggle
without depending on any other notebook's output.

Design notes that matter for the grading rubric
-----------------------------------------------
* Section 6.2 of the brief requires a *source-based* split, not a random one.
  For byte-plot malware images the analogue of "same subject" is
  "same/near-identical binary": malware families are full of polymorphic
  variants whose byte-plots are near-duplicates.  A random split therefore
  leaks.  `build_dedup_groups()` finds exact duplicates (SHA-1 of raw pixels)
  and near-duplicates (128-bit difference hash, Hamming distance <= T), unions
  them, and every split in this project is a *grouped stratified* split over
  those groups (sklearn StratifiedGroupKFold).  Source-of-origin is also
  recorded and reported.
* Section 6.1 requires the full metric set; `full_metrics()` returns it and
  `macro_f1` is the headline number everywhere (Section 6.2, imbalance rule).
* Section 6.4 requires 5-fold CV (mean +- std) plus a significance test;
  `mcnemar_test`, `wilcoxon_folds` and `friedman_nemenyi` are provided.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

__all__ = [
    "CFG", "set_seed", "on_kaggle", "find_dataset_root", "scan_index",
    "MALIMG_25", "normalize_family", "tag_malimg_subset",
    "build_dedup_groups", "grouped_holdout_split", "grouped_kfold",
    "load_images", "ByteImageDataset", "make_loaders",
    "SEBlock", "SpatialAttention", "CBAM", "CoordAtt", "GeM", "MSConvBlock",
    "ByteAttnNet", "build_baseline", "count_params", "model_size_mb",
    "train_model", "predict", "full_metrics", "metrics_frame",
    "plot_confusion", "plot_roc_ovr", "plot_pr_ovr", "plot_history",
    "mcnemar_test", "wilcoxon_folds", "friedman_nemenyi",
    "save_json", "load_json", "banner", "RELATED_WORK",
]

# =============================================================================
# 0. Global configuration
# =============================================================================

def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


@dataclass
class Config:
    # ---- identity (EDIT THESE TWO for your group) -------------------------
    group: str = "Group00"
    dataset_slug: str = "MaleBin"

    # ---- paths ------------------------------------------------------------
    data_root: str | None = os.environ.get("MALEBIN_DATA_ROOT") or None
    out_dir: str = "/kaggle/working" if Path("/kaggle/working").exists() else "./out"

    # ---- images -----------------------------------------------------------
    img_size: int = 224                   # network input side
    cache_size: int = 256                 # side at which images are cached in RAM
    in_channels: int = 1                  # byte-plots are single-channel

    # ---- split / leakage control -----------------------------------------
    seed: int = 42
    test_frac: float = 0.20               # held-out test set (grouped+stratified)
    val_frac: float = 0.15                # of the remaining train pool
    n_folds: int = 5
    dhash_threshold: int = 6              # Hamming distance (of 128 bits) for near-dup
    group_by_duplicates: bool = True      # <- set False ONLY to demo the leakage effect

    # ---- training ---------------------------------------------------------
    batch_size: int = 64
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 6                     # early stopping on val macro-F1
    num_workers: int = 2
    amp: bool = True
    class_weighted_loss: bool = True

    # ---- evaluation scope -------------------------------------------------
    # "malebin39" -> all 39 families (the real task)
    # "malimg25"  -> only the 25 Malimg families, so results are directly
    #                comparable with the published Malimg numbers (fair Pillar-A)
    eval_scope: str = "malebin39"

    # ---- speed switches ---------------------------------------------------
    fast: bool = bool(int(os.environ.get("MALEBIN_FAST", "0")))
    max_per_class: int | None = None       # subsample for a smoke test

    def __post_init__(self):
        if self.fast:                      # smoke-test / debug profile
            self.img_size = 64
            self.cache_size = 64
            self.epochs = 2
            self.n_folds = 2
            self.batch_size = 32
            self.patience = 2
            self.num_workers = 0
            self.amp = False
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        for sub in ("models", "figures", "artifacts"):
            (Path(self.out_dir) / sub).mkdir(parents=True, exist_ok=True)

    # convenience -----------------------------------------------------------
    @property
    def prefix(self) -> str:
        return f"{self.group}_{self.dataset_slug}"

    def path(self, *parts) -> Path:
        return Path(self.out_dir).joinpath(*parts)

    def fig(self, name: str) -> Path:
        return self.path("figures", f"{self.prefix}_{name}")

    def art(self, name: str) -> Path:
        return self.path("artifacts", f"{self.prefix}_{name}")

    def mdl(self, name: str) -> Path:
        return self.path("models", f"{self.prefix}_{name}")


CFG = Config()


def banner(txt: str, ch: str = "=") -> None:
    line = ch * max(70, len(txt) + 4)
    print(f"\n{line}\n  {txt}\n{line}")


def set_seed(seed: int | None = None) -> None:
    seed = CFG.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# =============================================================================
# 1. Dataset discovery and indexing
# =============================================================================

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# The 25 families of the original Malimg dataset (Nataraj et al., VizSec 2011).
# Used to carve out a subset of MaleBin that is directly comparable with the
# published Malimg numbers -> fair Pillar-A comparison (brief Sec. 8).
MALIMG_25 = [
    "Adialer.C", "Agent.FYI", "Allaple.A", "Allaple.L", "Alueron.gen!J",
    "Autorun.K", "C2LOP.gen!g", "C2LOP.P", "Dialplatform.B", "Dontovo.A",
    "Fakerean", "Instantaccess", "Lolyda.AA1", "Lolyda.AA2", "Lolyda.AA3",
    "Lolyda.AT", "Malex.gen!J", "Obfuscator.AD", "Rbot!gen", "Skintrim.N",
    "Swizzor.gen!E", "Swizzor.gen!I", "VB.AT", "Wintrim.BX", "Yuner.A",
]


def normalize_family(name: str) -> str:
    """Lower-cased, punctuation-stripped family key for robust matching."""
    return "".join(c for c in str(name).lower() if c.isalnum())


_MALIMG_KEYS = {normalize_family(m) for m in MALIMG_25}


def find_dataset_root(explicit: str | None = None) -> Path:
    """
    Locate the directory whose immediate children are the class folders.

    Works for any of these layouts, which is what makes the notebooks
    portable between Kaggle and a local copy:

        <root>/<class>/*.png
        <root>/MaleBin/<class>/*.png
        <root>/train/<class>/*.png          (also picks up val/ and test/)
    """
    cands: list[Path] = []
    if explicit:
        cands.append(Path(explicit))
    if CFG.data_root:
        cands.append(Path(CFG.data_root))
    cands += [
        Path("/kaggle/input/malebin-malware-binary-greyscale-images"),
        Path("./data/malebin"), Path("./malebin"), Path("./data"),
    ]
    if Path("/kaggle/input").exists():
        cands += sorted(Path("/kaggle/input").iterdir())

    for c in cands:
        if not c.exists() or not c.is_dir():
            continue
        root = _descend_to_class_parent(c)
        if root is not None:
            return root
    raise FileNotFoundError(
        "Could not locate the MaleBin image folders.\n"
        "On Kaggle: Add Data -> search 'MaleBin malware binary greyscale' -> Add.\n"
        "Then set CFG.data_root='/kaggle/input/<slug>' and re-run.\n"
        f"Tried: {[str(c) for c in cands]}"
    )


def _immediate_image_count(d: Path, cap: int = 3) -> int:
    n = 0
    try:
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                n += 1
                if n >= cap:
                    break
    except OSError:
        return 0
    return n


def _descend_to_class_parent(d: Path, depth: int = 0) -> Path | None:
    """Return the shallowest directory that has >=2 image-bearing subfolders."""
    if depth > 4:
        return None
    try:
        subs = sorted(p for p in d.iterdir() if p.is_dir())
    except OSError:
        return None
    with_imgs = [s for s in subs if _immediate_image_count(s) > 0]
    if len(with_imgs) >= 2:
        return d
    # only a single container level (e.g. <root>/MaleBin/, or a split wrapper)
    for s in subs:
        r = _descend_to_class_parent(s, depth + 1)
        if r is not None:
            return r
    return None


def scan_index(root: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Walk the dataset and return one row per image.

    Columns
    -------
    path        absolute file path
    family      class folder name (malware family)
    label       integer encoding of `family` (alphabetical, stable)
    n_bytes     file size on disk
    is_malimg   the family is one of the 25 original Malimg families
    """
    root = Path(root) if root is not None else find_dataset_root()
    rows = []
    for fam_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for p in sorted(fam_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                rows.append((str(p), fam_dir.name, p.stat().st_size))
    if not rows:  # images sitting directly under root
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                rows.append((str(p), p.parent.name, p.stat().st_size))
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")

    df = pd.DataFrame(rows, columns=["path", "family", "n_bytes"])
    fams = sorted(df["family"].unique())
    df["label"] = df["family"].map({f: i for i, f in enumerate(fams)}).astype(int)
    df["is_malimg"] = df["family"].map(lambda f: normalize_family(f) in _MALIMG_KEYS)
    df = df.sort_values("path", kind="mergesort").reset_index(drop=True)

    if verbose:
        vc = df.family.value_counts()
        banner(f"Dataset index: {root}")
        print(f"  images            : {len(df):,}")
        print(f"  families (classes): {df['family'].nunique()}")
        print(f"  Malimg families   : {df.loc[df.is_malimg,'family'].nunique()}"
              f" / 25  ({df.is_malimg.sum():,} images)")
        print(f"  other families    : {df.loc[~df.is_malimg,'family'].nunique()}"
              f" ({(~df.is_malimg).sum():,} images)")
        print(f"  per-class counts  : min={vc.min()}  max={vc.max()}  "
              f"imbalance ratio={vc.max()/max(vc.min(),1):.2f}")
    return df


def tag_malimg_subset(df: pd.DataFrame, scope: str | None = None) -> pd.DataFrame:
    """
    Restrict + re-encode labels according to CFG.eval_scope.

    'malebin39' -> everything, labels 0..K-1
    'malimg25'  -> only the Malimg families, labels re-encoded 0..24, so the
                   result is directly comparable with published Malimg numbers.
    """
    scope = scope or CFG.eval_scope
    if scope == "malimg25":
        out = df[df["is_malimg"]].copy()
        if out.empty:
            raise ValueError(
                "eval_scope='malimg25' but no folder name matched the 25 Malimg "
                "families. Print df.family.unique() and adjust MALIMG_25.")
    elif scope == "malebin39":
        out = df.copy()
    else:
        raise ValueError(f"unknown eval_scope {scope!r}")
    fams = sorted(out["family"].unique())
    out["label"] = out["family"].map({f: i for i, f in enumerate(fams)}).astype(int)
    out = out.reset_index(drop=True)
    print(f"[scope={scope}] {len(out):,} images, {len(fams)} classes")
    return out


# =============================================================================
# 2. Image loading / caching
# =============================================================================

def _load_one(args):
    path, side = args
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L")                      # byte-plots are grayscale
        if im.size != (side, side):
            im = im.resize((side, side), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_images(df: pd.DataFrame, side: int | None = None,
                workers: int | None = None, verbose: bool = True) -> np.ndarray:
    """
    Decode every image once into a single uint8 array  (N, side, side).

    12,464 x 256 x 256 uint8 == 817 MB, which fits comfortably in Kaggle RAM and
    removes disk I/O from the training loop entirely (a large speed win over an
    ImageFolder pipeline that re-decodes a PNG on every epoch).
    """
    side = side or CFG.cache_size
    n = len(df)
    out = np.zeros((n, side, side), dtype=np.uint8)
    paths = df["path"].tolist()
    t0 = time.time()

    if workers is None:
        workers = 0 if (CFG.fast or n < 400) else min(4, (os.cpu_count() or 2))

    done = False
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, arr in enumerate(ex.map(_load_one,
                                               ((p, side) for p in paths),
                                               chunksize=64)):
                    out[i] = arr
            done = True
        except Exception as e:                     # sandboxes without fork/spawn
            print(f"  [load_images] parallel decode failed ({e!r}); serial fallback")
    if not done:
        try:
            from tqdm.auto import tqdm
            it = tqdm(paths, desc=f"decode@{side}", unit="img")
        except Exception:
            it = paths
        for i, p in enumerate(it):
            out[i] = _load_one((p, side))

    if verbose:
        print(f"  cached {n:,} images at {side}x{side} "
              f"({out.nbytes/1e6:.0f} MB) in {time.time()-t0:.1f}s")
    return out


# =============================================================================
# 3. LEAKAGE CONTROL  (brief Sec. 6.2 -- graded)
# =============================================================================
#
# Why this is needed
# ------------------
# There is no "subject" column in MaleBin, but the same leakage mechanism is
# present: a malware family consists of polymorphic *variants of one binary*,
# and their byte-plots are near-identical.  If variant A lands in train and its
# near-twin A' lands in test, the model is scored on recognising a file it has
# already memorised -- exactly the failure mode Sec. 6.2 forbids.  MaleBin is
# also a *merge of two sources* (Malimg + a MalwareBazaar-derived set), so the
# same original sample can appear twice.
#
# So we construct the grouping key ourselves:
#   1. exact duplicates : SHA-1 over the raw resized pixel buffer
#   2. near duplicates  : 128-bit difference hash (horizontal + vertical
#                         gradients), Hamming distance <= CFG.dhash_threshold
#   3. union-find over both relations -> `dup_group`
# Every split (hold-out and the 5 CV folds) is then grouped on `dup_group` and
# stratified on `family` via sklearn StratifiedGroupKFold.

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dhash128(imgs: np.ndarray) -> np.ndarray:
    """
    128-bit difference hash per image, packed into (N, 16) uint8:
    64 bits from horizontal gradients + 64 bits from vertical gradients.
    Pure numpy, so there is no `imagehash` dependency to install on Kaggle.
    """
    n, side = imgs.shape[0], imgs.shape[1]
    idx = np.linspace(0, side, 10).astype(int)
    small = np.empty((n, 9, 9), dtype=np.float32)
    for r in range(9):
        r0, r1 = idx[r], max(idx[r] + 1, idx[r + 1])
        for c in range(9):
            c0, c1 = idx[c], max(idx[c] + 1, idx[c + 1])
            small[:, r, c] = imgs[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
    hbits = (small[:, :8, :8] > small[:, :8, 1:9]).reshape(n, 64)
    vbits = (small[:, :8, :8] > small[:, 1:9, :8]).reshape(n, 64)
    return np.packbits(np.concatenate([hbits, vbits], axis=1), axis=1)   # (n,16)


def build_dedup_groups(imgs: np.ndarray, df: pd.DataFrame,
                       threshold: int | None = None,
                       verbose: bool = True) -> np.ndarray:
    """
    Return an integer group id per image so that exact/near duplicates share an
    id.  Near-duplicate search is run *within* each family, so a group can never
    span two families (a cross-family collision would be a labelling issue, not
    a duplicate); we still assert that and report it.
    """
    threshold = CFG.dhash_threshold if threshold is None else threshold
    n = len(df)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if not CFG.group_by_duplicates:
        if verbose:
            print("  [!] CFG.group_by_duplicates=False -> every image is its own "
                  "group (leakage-demonstration mode only)")
        return np.arange(n)

    # ---- 1. exact duplicates ------------------------------------------------
    first: dict[str, int] = {}
    n_exact = 0
    for i in range(n):
        h = hashlib.sha1(imgs[i].tobytes()).hexdigest()
        if h in first:
            union(first[h], i)
            n_exact += 1
        else:
            first[h] = i

    # ---- 2. near duplicates, family by family -------------------------------
    hashes = _dhash128(imgs)
    labels = df["label"].to_numpy()
    n_near = 0
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < 2:
            continue
        H = hashes[idx]
        B = 512
        for s in range(0, len(idx), B):
            blk = H[s:s + B]
            # XOR every pair, then popcount through a 256-entry lookup table
            d = _POPCOUNT[np.bitwise_xor(blk[:, None, :], H[None, :, :])].sum(-1)
            ii, jj = np.nonzero(d <= threshold)
            for a, b in zip(ii, jj):
                gi, gj = int(idx[s + a]), int(idx[b])
                if gi < gj:
                    union(gi, gj)
                    n_near += 1

    raw = np.array([find(i) for i in range(n)])
    _, groups = np.unique(raw, return_inverse=True)          # relabel 0..G-1

    gf = pd.DataFrame({"g": groups, "f": df["family"].to_numpy()})
    cross = int((gf.groupby("g")["f"].nunique() > 1).sum())
    assert cross == 0, f"{cross} duplicate group(s) span more than one family"

    if verbose:
        sizes = pd.Series(groups).value_counts()
        n_groups = int(groups.max()) + 1
        print(f"  exact-duplicate links     : {n_exact:,}")
        print(f"  near-duplicate links      : {n_near:,} "
              f"(dHash Hamming <= {threshold}/128)")
        print(f"  duplicate groups          : {n_groups:,} for {n:,} images "
              f"({100*(1-n_groups/n):.1f}% collapse)")
        print(f"  largest group             : {sizes.max()} images")
        print(f"  groups spanning >1 family : {cross}   (must be 0)")
    return groups


def grouped_holdout_split(df: pd.DataFrame, groups: np.ndarray,
                          test_frac: float | None = None,
                          val_frac: float | None = None,
                          seed: int | None = None,
                          verbose: bool = True):
    """
    Grouped + stratified train / val / test indices.

    StratifiedGroupKFold is applied twice so that (a) no duplicate group is
    split across subsets and (b) every family keeps roughly its global share.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    test_frac = CFG.test_frac if test_frac is None else test_frac
    val_frac = CFG.val_frac if val_frac is None else val_frac
    seed = CFG.seed if seed is None else seed
    y = df["label"].to_numpy()

    k_test = max(2, int(round(1.0 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    pool_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups))

    k_val = max(2, int(round(1.0 / val_frac)))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed + 1)
    sub_tr, sub_va = next(sgkf2.split(np.zeros(len(pool_idx)),
                                      y[pool_idx], groups[pool_idx]))
    train_idx, val_idx = pool_idx[sub_tr], pool_idx[sub_va]

    assert_no_group_leak({"train": train_idx, "val": val_idx, "test": test_idx},
                         groups)
    if verbose:
        print(f"  train {len(train_idx):,} | val {len(val_idx):,} "
              f"| test {len(test_idx):,}")
        print(f"  classes present -> train {len(set(y[train_idx]))}, "
              f"val {len(set(y[val_idx]))}, test {len(set(y[test_idx]))} "
              f"(dataset has {len(set(y))})")
        print("  leakage check: no duplicate group appears in two subsets  [OK]")
    return train_idx, val_idx, test_idx


def grouped_kfold(df: pd.DataFrame, groups: np.ndarray,
                  n_folds: int | None = None, seed: int | None = None):
    """5-fold *stratified grouped* CV folds (brief Sec. 6.4)."""
    from sklearn.model_selection import StratifiedGroupKFold
    n_folds = CFG.n_folds if n_folds is None else n_folds
    seed = CFG.seed if seed is None else seed
    y = df["label"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(sgkf.split(np.zeros(len(y)), y, groups))
    for k, (tr, te) in enumerate(folds):
        assert not (set(groups[tr]) & set(groups[te])), f"group leak in fold {k}"
    return folds


def assert_no_group_leak(subsets: dict, groups: np.ndarray) -> None:
    names = list(subsets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = set(groups[subsets[names[i]]])
            b = set(groups[subsets[names[j]]])
            inter = a & b
            if inter:
                raise AssertionError(
                    f"LEAKAGE: {len(inter)} duplicate group(s) shared between "
                    f"'{names[i]}' and '{names[j]}'")


# =============================================================================
# 4. Torch imports  (torch is pre-installed on Kaggle GPU/CPU images)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 5. Torch dataset + byte-aware augmentation
# =============================================================================

class ByteImageDataset(Dataset):
    """
    Serves cached uint8 byte-plots as normalised float tensors.

    Augmentation policy -- a design decision we defend in the report
    ---------------------------------------------------------------
    A byte-plot is NOT a natural image.  A pixel's (row, col) position is the
    byte's offset in the file: row r covers bytes [r*W, (r+1)*W).  Therefore
      * horizontal / vertical flips are meaningless -- they reverse the byte
        order of a PE file, producing an input that can never occur;
      * rotations are meaningless for the same reason;
      * brightness / contrast jitter rewrites byte *values* -- also invalid.
    What legitimately varies between two variants of the same family is
      * where a section starts (inserted junk, padding)  -> vertical roll
      * the file being longer or shorter                 -> row-crop / scale
      * a region being packed, encrypted or overwritten  -> random erasing
    `aug='byte'` implements exactly those three.  `aug='naive'` is the usual
    flip+rotate recipe and exists only so the Task-3 ablation can measure it.
    """

    def __init__(self, imgs, labels, indices, out_size=None, aug="none",
                 in_channels=None):
        assert aug in ("none", "byte", "naive"), aug
        self.imgs = imgs
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices)
        self.out_size = int(out_size or CFG.img_size)
        self.aug = aug
        self.in_channels = int(in_channels or CFG.in_channels)
        self._epoch = 0

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, e):
        self._epoch = int(e)

    # -- augmentation primitives (numpy, cheap, dependency-free) --------------
    @staticmethod
    def _aug_byte(a, rng):
        s = a.shape[0]
        if rng.random() < 0.7:                      # section boundary shift
            a = np.roll(a, int(rng.integers(-s // 8, s // 8 + 1)), axis=0)
        if rng.random() < 0.5:                      # file longer / shorter
            keep = max(4, int(s * rng.uniform(0.85, 1.0)))
            top = int(rng.integers(0, s - keep + 1))
            a = a[top:top + keep, :]
        if rng.random() < 0.35:                     # packed / zeroed region
            h = int(rng.integers(max(1, a.shape[0] // 16), max(2, a.shape[0] // 5)))
            w = int(rng.integers(max(1, a.shape[1] // 8), max(2, a.shape[1])))
            r0 = int(rng.integers(0, a.shape[0] - h + 1))
            c0 = int(rng.integers(0, a.shape[1] - w + 1))
            fill = int(rng.choice([0, 255, int(rng.integers(0, 256))]))
            a = a.copy()
            a[r0:r0 + h, c0:c0 + w] = fill
        return a

    @staticmethod
    def _aug_naive(a, rng):
        if rng.random() < 0.5:
            a = a[:, ::-1]
        if rng.random() < 0.5:
            a = a[::-1, :]
        k = int(rng.integers(0, 4))
        if k:
            a = np.rot90(a, k)
        return np.ascontiguousarray(a)

    def __getitem__(self, i):
        gi = int(self.indices[i])
        a = self.imgs[gi]
        if self.aug != "none":
            rng = np.random.default_rng(
                (CFG.seed * 1_000_003 + gi * 7919 + self._epoch * 104_729
                 + int(torch.randint(0, 1 << 30, (1,)).item())) % (2 ** 63))
            a = self._aug_byte(a, rng) if self.aug == "byte" else self._aug_naive(a, rng)
        t = torch.from_numpy(np.ascontiguousarray(a)).float().div_(255.0)[None]
        if t.shape[-1] != self.out_size or t.shape[-2] != self.out_size:
            t = F.interpolate(t[None], size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)[0]
        # Per-image standardisation removes the "how bright is this file overall"
        # nuisance factor and keeps the texture, which is what identifies a family.
        t = (t - t.mean()) / (t.std() + 1e-5)
        if self.in_channels == 3:
            t = t.repeat(3, 1, 1)
        return t, int(self.labels[gi])


def make_loaders(imgs, df, train_idx, val_idx, test_idx=None,
                 batch_size=None, aug="byte", in_channels=None, out_size=None):
    bs = int(batch_size or CFG.batch_size)
    y = df["label"].to_numpy()
    kw = dict(out_size=out_size, in_channels=in_channels)
    common = dict(num_workers=CFG.num_workers,
                  pin_memory=torch.cuda.is_available(),
                  persistent_workers=CFG.num_workers > 0)
    tr = DataLoader(ByteImageDataset(imgs, y, train_idx, aug=aug, **kw),
                    batch_size=bs, shuffle=True, drop_last=False, **common)
    va = DataLoader(ByteImageDataset(imgs, y, val_idx, aug="none", **kw),
                    batch_size=bs * 2, shuffle=False, **common)
    te = None
    if test_idx is not None:
        te = DataLoader(ByteImageDataset(imgs, y, test_idx, aug="none", **kw),
                        batch_size=bs * 2, shuffle=False, **common)
    return tr, va, te


# =============================================================================
# 6. Attention building blocks
# =============================================================================
#
# Written from the papers' equations rather than imported, because Pillar B of
# the rubric asks us to explain our own model in our own words. References:
#   SE          Hu, Shen & Sun, CVPR 2018        (channel attention)
#   CBAM        Woo et al., ECCV 2018            (channel + spatial attention)
#   CoordAtt    Hou, Zhou & Feng, CVPR 2021      (direction-aware attention)
#   GeM         Radenovic, Tolias & Chum, TPAMI 2019

class SEBlock(nn.Module):
    """Channel attention -- 'which feature maps matter for this file?'"""

    def __init__(self, ch, r=8):
        super().__init__()
        hid = max(4, ch // r)
        self.fc1 = nn.Conv2d(ch, hid, 1)
        self.fc2 = nn.Conv2d(hid, ch, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * s


class SpatialAttention(nn.Module):
    """
    Spatial attention -- 'which byte offsets matter?'
    Descriptor = [channel-max, channel-mean] -> 7x7 conv -> sigmoid mask.
    """

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        mx = x.max(dim=1, keepdim=True)[0]
        av = x.mean(dim=1, keepdim=True)
        self.last_map = torch.sigmoid(self.conv(torch.cat([mx, av], dim=1)))
        return x * self.last_map


class CBAM(nn.Module):
    """CBAM = channel attention (avg+max squeeze) then spatial attention."""

    def __init__(self, ch, r=8, k=7):
        super().__init__()
        hid = max(4, ch // r)
        self.mlp = nn.Sequential(nn.Conv2d(ch, hid, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(hid, ch, 1))
        self.sa = SpatialAttention(k)

    def forward(self, x):
        c = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        self.last_channel = torch.sigmoid(c)
        return self.sa(x * self.last_channel)


class CoordAtt(nn.Module):
    """
    Coordinate attention (Hou et al., CVPR 2021).

    Why this is the right attention for byte-plots
    ----------------------------------------------
    Pooling happens along one axis at a time, so the module emits one gate per
    ROW and one per COLUMN and never destroys position.  In a byte-plot the row
    index *is* the byte offset: the PE header sits at the top, .text below it,
    .data / .rsrc lower, zero padding at the bottom.  A per-row gate can
    therefore express "attend to the import-table region", a structurally
    meaningful statement that globally-pooled channel attention cannot make.
    `last_h` / `last_w` are kept for the Task-3 explainability notebook.
    """

    def __init__(self, ch, r=16):
        super().__init__()
        hid = max(8, ch // r)
        self.conv1 = nn.Conv2d(ch, hid, 1)
        self.bn1 = nn.BatchNorm2d(hid)
        self.conv_h = nn.Conv2d(hid, ch, 1)
        self.conv_w = nn.Conv2d(hid, ch, 1)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(dim=3, keepdim=True)                       # (n,c,h,1) per row
        xw = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)   # (n,c,w,1) per col
        y = F.hardswish(self.bn1(self.conv1(torch.cat([xh, xw], dim=2))))
        yh, yw = torch.split(y, [h, w], dim=2)
        ah = torch.sigmoid(self.conv_h(yh))                          # (n,c,h,1)
        aw = torch.sigmoid(self.conv_w(yw).permute(0, 1, 3, 2))      # (n,c,1,w)
        self.last_h, self.last_w = ah.detach(), aw.detach()
        return x * ah * aw


def attention_module(kind, ch):
    kind = (kind or "none").lower()
    if kind in ("none", ""):
        return nn.Identity()
    if kind == "se":
        return SEBlock(ch)
    if kind == "spatial":
        return SpatialAttention()
    if kind == "cbam":
        return CBAM(ch)
    if kind == "coord":
        return CoordAtt(ch)
    if kind in ("cbam+coord", "full"):
        return nn.Sequential(CBAM(ch), CoordAtt(ch))
    raise ValueError(f"unknown attention {kind!r}")


class GeM(nn.Module):
    """
    Generalised-mean pooling  (mean(x**p))**(1/p)  with learnable p.
    p=1 is average pooling, p->inf is max pooling.  Byte-plot families are
    distinguished by *texture energy*, so a learnable p lets the head settle
    anywhere between "average texture" and "the single most distinctive block"
    instead of us guessing which one is right.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0, max=8.0)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1) \
                .pow(1.0 / p).flatten(1)


class GapGmp(nn.Module):
    def forward(self, x):
        return torch.cat([F.adaptive_avg_pool2d(x, 1).flatten(1),
                          F.adaptive_max_pool2d(x, 1).flatten(1)], dim=1)


def pooling_module(kind, ch):
    kind = (kind or "gem").lower()
    if kind == "gem":
        return GeM(), ch
    if kind == "gap":
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten()), ch
    if kind == "gap+gmp":
        return GapGmp(), ch * 2
    raise ValueError(f"unknown pool {kind!r}")


class MSConvBlock(nn.Module):
    """
    Multi-scale residual block: three parallel views, concatenated, fused 1x1.
      * 3x3               -> fine byte texture (opcode-level n-grams)
      * 3x3 dilation 2    -> 5x5 receptive field at 3x3 cost (basic-block level)
      * 3x3 dilation 3    -> 7x7 receptive field           (section level)
    A family's signature lives at several scales at once (byte patterns inside a
    function, repeated function bodies, whole padded sections), so one kernel
    size must compromise.  Same motivation as the multi-scale-kernel blocks of
    PAFE and IMCMK-CNN, but the dilated form keeps a plain 3x3 parameter count.
    Set multiscale=False for the ablation.
    """

    def __init__(self, cin, cout, use_bn=True, multiscale=True):
        super().__init__()
        self.multiscale = multiscale
        norm = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        if multiscale:
            br = max(8, cout // 3)
            self.b1 = nn.Sequential(nn.Conv2d(cin, br, 3, padding=1, bias=False),
                                    norm(br), nn.ReLU(inplace=True))
            self.b2 = nn.Sequential(nn.Conv2d(cin, br, 3, padding=2, dilation=2,
                                              bias=False), norm(br), nn.ReLU(inplace=True))
            self.b3 = nn.Sequential(nn.Conv2d(cin, br, 3, padding=3, dilation=3,
                                              bias=False), norm(br), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Conv2d(br * 3, cout, 1, bias=False), norm(cout))
        else:
            self.b1 = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                                    norm(cout), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                                      norm(cout))
        self.short = (nn.Identity() if cin == cout else
                      nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), norm(cout)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = (torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
             if self.multiscale else self.b1(x))
        return self.act(self.fuse(h) + self.short(x))


# =============================================================================
# 7. THE PROPOSED MODEL -- ByteAttnNet
# =============================================================================

class ByteAttnNet(nn.Module):
    """
    ByteAttnNet: a from-scratch CNN with a two-part attention stack, designed
    specifically for malware byte-plot images.

    Forward path
    ------------
        input  (B, 1, S, S)   byte-plot, per-image standardised
          |
        STEM      3x3 conv stride 2 -> BN -> ReLU            S -> S/2
          |
        STAGE i in 1..4:
            MSConvBlock(c_{i-1} -> c_i)     multi-scale texture, residual
            [MSConvBlock(c_i -> c_i)] x (depth_i - 1)
            attention(c_i)                  CBAM then CoordAtt
            MaxPool 2x2  (stages 1..3 only)
          |
        POOL      GeM (learnable p)                      -> (B, c_4)
          |
        HEAD      Dropout -> Linear(c_4 -> n_classes)
    Channels default to (48, 96, 192, 320); depth defaults to (1, 1, 2, 2).

    Why it should beat the baselines
    --------------------------------
    1. The ImageNet-pretrained baselines carry a *natural-image* prior (oriented
       edges, colour-opponent blobs, object parts).  A byte-plot has none of
       those; what it has is stationary texture whose statistics change with
       byte offset.  Fine-tuning has to unlearn the prior first, and it does so
       with 23M+ parameters on ~10k images.
    2. CBAM re-weights *channels* (which texture detectors this family needs)
       and then *space* (which regions carry them).
    3. CoordAtt adds the piece that matters most here: it factorises attention
       into a per-ROW and a per-COLUMN gate, and the row index of a byte-plot is
       literally the byte offset in the file.  The network can therefore learn
       "for family X, the informative bytes sit ~15-25% into the file", which no
       flatten-then-pool channel attention can represent.
    4. Multi-scale dilated branches see opcode-level, function-level and
       section-level structure simultaneously.
    5. GeM lets the readout interpolate between average and max texture energy.
    6. It is ~1-3M parameters, i.e. ~10x smaller than ResNet50, which matters
       for the training-time / model-size columns the brief asks for.

    Every design choice above is switchable, so the Task-3 ablation can measure
    each one instead of us asserting it.
    """

    def __init__(self, n_classes, in_ch=1, channels=(48, 96, 192, 320),
                 depth=(1, 1, 2, 2), attention="cbam+coord", pool="gem",
                 dropout=0.3, use_bn=True, multiscale=True, stem_stride=2):
        super().__init__()
        self.hparams = dict(n_classes=n_classes, in_ch=in_ch,
                            channels=tuple(channels), depth=tuple(depth),
                            attention=attention, pool=pool, dropout=dropout,
                            use_bn=use_bn, multiscale=multiscale,
                            stem_stride=stem_stride)
        norm = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels[0], 3, stride=stem_stride, padding=1, bias=False),
            norm(channels[0]), nn.ReLU(inplace=True))

        stages, cin = [], channels[0]
        for i, (cout, d) in enumerate(zip(channels, depth)):
            blocks = [MSConvBlock(cin, cout, use_bn=use_bn, multiscale=multiscale)]
            blocks += [MSConvBlock(cout, cout, use_bn=use_bn, multiscale=multiscale)
                       for _ in range(max(0, d - 1))]
            blocks.append(attention_module(attention, cout))
            if i < len(channels) - 1:
                blocks.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*blocks))
            cin = cout
        self.stages = nn.ModuleList(stages)
        self.pool, feat = pooling_module(pool, cin)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feat, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def features(self, x):
        """Last conv feature map -- the Grad-CAM target."""
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        return x

    def forward(self, x):
        return self.fc(self.drop(self.pool(self.features(x))))

    @property
    def gradcam_layer(self):
        """Deepest spatial tensor producer -> what Grad-CAM should hook."""
        return self.stages[-1]


# =============================================================================
# 8. Baselines  (brief: 3-4 standard CNNs)
# =============================================================================

BASELINE_NAMES = ["SimpleCNN", "ResNet50", "DenseNet121", "MobileNetV3-Small",
                  "EfficientNet-B0", "VGG16"]


class SimpleCNN(nn.Module):
    """The textbook 4-block VGG-style CNN, trained from scratch, no attention."""

    def __init__(self, n_classes, in_ch=1, width=32, dropout=0.3):
        super().__init__()
        chs = [width, width * 2, width * 4, width * 8]
        layers, c = [], in_ch
        for ch in chs:
            layers += [nn.Conv2d(c, ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                       nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c, n_classes))

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))

    @property
    def gradcam_layer(self):
        return self.body


def build_baseline(name, n_classes, pretrained=True, in_ch=3, dropout=0.3):
    """
    Return (model, in_channels_required).

    Pretrained torchvision backbones expect 3 channels, so `make_loaders` is
    called with in_channels=3 for them and the grayscale plane is repeated.
    This is the standard protocol in the malware-image literature (PAFE,
    IMCEC, Alshomrani et al.) so the comparison stays fair.
    """
    name = name.strip()
    if name == "SimpleCNN":
        return SimpleCNN(n_classes, in_ch=1, dropout=dropout), 1

    import torchvision.models as tvm
    W = "DEFAULT" if pretrained else None

    if name == "ResNet50":
        m = tvm.resnet50(weights=W)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "ResNet18":
        m = tvm.resnet18(weights=W)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif name == "DenseNet121":
        m = tvm.densenet121(weights=W)
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    elif name in ("MobileNetV3-Small", "MobileNetV3"):
        m = tvm.mobilenet_v3_small(weights=W)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "MobileNetV2":
        m = tvm.mobilenet_v2(weights=W)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name in ("EfficientNet-B0", "EfficientNetB0"):
        m = tvm.efficientnet_b0(weights=W)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif name == "VGG16":
        m = tvm.vgg16_bn(weights=W)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    else:
        raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
    return m, 3


def count_params(model, trainable_only=True):
    ps = model.parameters()
    return int(sum(p.numel() for p in ps if (p.requires_grad or not trainable_only)))


def model_size_mb(model):
    """On-disk float32 size of the state dict, in MB."""
    n = sum(p.numel() for p in model.parameters())
    b = sum(buf.numel() for buf in model.buffers())
    return (n + b) * 4 / 1e6


def gradcam_target(model):
    """Best-effort last-conv-stage locator, for any of the models above."""
    if hasattr(model, "gradcam_layer"):
        return model.gradcam_layer
    for attr in ("layer4", "features", "body"):
        if hasattr(model, attr):
            return getattr(model, attr)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("no Conv2d found for Grad-CAM")
    return convs[-1]


# =============================================================================
# 9. Training / inference
# =============================================================================

def class_weights_from(labels, n_classes):
    """
    Inverse-frequency weights, normalised to mean 1.
    Used because the brief (Sec. 6.2) grades macro-F1 and per-class recall:
    an unweighted loss on an imbalanced set optimises the majority classes.
    """
    cnt = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    cnt[cnt == 0] = 1.0
    w = cnt.sum() / (n_classes * cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, loader, device=None, return_logits=False):
    """Return (y_true, y_pred, y_prob) over a loader, in loader order."""
    device = device or DEVICE
    model.eval().to(device)
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type,
                            enabled=(CFG.amp and device.type == "cuda")):
            out = model(xb)
        ps.append(out.float().cpu())
        ys.append(yb)
    logits = torch.cat(ps)
    y_true = torch.cat(ys).numpy()
    prob = torch.softmax(logits, dim=1).numpy()
    y_pred = prob.argmax(1)
    if return_logits:
        return y_true, y_pred, prob, logits.numpy()
    return y_true, y_pred, prob


def train_model(model, train_loader, val_loader, n_classes, *,
                epochs=None, lr=None, weight_decay=None, patience=None,
                label_smoothing=None, class_weighted=None, device=None,
                scheduler="cosine", optimizer="adamw", tag="model",
                verbose=True, save_path=None):
    """
    One generic training loop used by every model in the project, so that the
    comparison between baselines and the proposed model is apples-to-apples
    (same optimiser family, same schedule, same early-stopping criterion, same
    class weighting).  Selection metric is *validation macro-F1*, never accuracy.

    Returns (best_state_dict, history_dataframe, summary_dict).
    """
    from sklearn.metrics import f1_score
    device = device or DEVICE
    epochs = int(epochs or CFG.epochs)
    lr = float(lr if lr is not None else CFG.lr)
    weight_decay = float(weight_decay if weight_decay is not None else CFG.weight_decay)
    patience = int(patience or CFG.patience)
    ls = float(label_smoothing if label_smoothing is not None else CFG.label_smoothing)
    cw_on = CFG.class_weighted_loss if class_weighted is None else class_weighted

    model = model.to(device)
    ytr = np.concatenate([train_loader.dataset.labels[train_loader.dataset.indices]])
    weight = class_weights_from(ytr, n_classes).to(device) if cw_on else None
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)

    if optimizer.lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=weight_decay, nesterov=True)
    else:
        raise ValueError(optimizer)

    steps = max(1, len(train_loader))
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25)
        per_step = True
    elif scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=2)
        per_step = False
    elif scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3),
                                                gamma=0.3)
        per_step = False
    else:
        sched, per_step = None, False

    use_amp = bool(CFG.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, best_ep, bad = -1.0, None, -1, 0
    hist, t_start = [], time.time()

    for ep in range(1, epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(ep)
        model.train()
        run_loss, seen, correct = 0.0, 0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            if sched is not None and per_step:
                sched.step()
            run_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += yb.size(0)

        yv, pv, _ = predict(model, val_loader, device)
        vf1 = f1_score(yv, pv, average="macro", zero_division=0)
        vacc = float((yv == pv).mean())
        if sched is not None and not per_step:
            sched.step(vf1) if scheduler == "plateau" else sched.step()

        hist.append(dict(epoch=ep, train_loss=run_loss / max(1, seen),
                         train_acc=correct / max(1, seen),
                         val_acc=vacc, val_macro_f1=vf1,
                         lr=opt.param_groups[0]["lr"], secs=time.time() - t0))
        if verbose:
            print(f"  [{tag}] ep {ep:>3}/{epochs}  loss {hist[-1]['train_loss']:.4f}"
                  f"  tr_acc {hist[-1]['train_acc']:.4f}"
                  f"  val_acc {vacc:.4f}  val_macroF1 {vf1:.4f}"
                  f"  ({hist[-1]['secs']:.1f}s)")

        if vf1 > best_f1 + 1e-5:
            best_f1, best_ep, bad = vf1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  [{tag}] early stop at epoch {ep} "
                          f"(best epoch {best_ep}, val macro-F1 {best_f1:.4f})")
                break

    train_secs = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = dict(tag=tag, best_epoch=best_ep, best_val_macro_f1=float(best_f1),
                   epochs_run=len(hist), train_seconds=float(train_secs),
                   params=count_params(model), size_mb=model_size_mb(model),
                   device=str(device))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state,
                    "hparams": getattr(model, "hparams", {}),
                    "summary": summary}, save_path)
        if verbose:
            print(f"  [{tag}] checkpoint -> {save_path}")
    return best_state, pd.DataFrame(hist), summary


# =============================================================================
# 10. Metrics  (brief Sec. 6.1 -- the full mandatory set)
# =============================================================================

def full_metrics(y_true, y_pred, y_prob=None, class_names=None, name="model"):
    """
    Everything Sec. 6.1 asks for, in one dict:
      accuracy, balanced accuracy, macro/weighted precision-recall-F1,
      Cohen's kappa, MCC, macro & weighted one-vs-rest ROC-AUC, macro average
      precision, per-class report, confusion matrix.
    macro_f1 is the headline metric for this project (imbalanced data).
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, cohen_kappa_score,
                                 matthews_corrcoef, roc_auc_score,
                                 average_precision_score, classification_report,
                                 confusion_matrix)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(len(class_names)) if class_names is not None else \
        np.unique(np.concatenate([y_true, y_pred]))

    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels)

    out = dict(
        model=name,
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        precision_macro=float(pm), recall_macro=float(rm), f1_macro=float(fm),
        precision_weighted=float(pw), recall_weighted=float(rw), f1_weighted=float(fw),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        n_test=int(len(y_true)), n_classes=int(len(labels)),
    )
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        Y = np.zeros((len(y_true), y_prob.shape[1]), dtype=np.int8)
        Y[np.arange(len(y_true)), y_true] = 1
        keep = Y.sum(0) > 0            # AUC undefined for classes absent from test
        try:
            out["roc_auc_macro_ovr"] = float(
                roc_auc_score(Y[:, keep], y_prob[:, keep], average="macro"))
            out["roc_auc_weighted_ovr"] = float(
                roc_auc_score(Y[:, keep], y_prob[:, keep], average="weighted"))
            out["avg_precision_macro"] = float(
                average_precision_score(Y[:, keep], y_prob[:, keep], average="macro"))
        except Exception as e:
            out["roc_auc_macro_ovr"] = float("nan")
            out["auc_note"] = f"undefined: {e}"

    tn = ([str(c) for c in class_names] if class_names is not None
          else [str(i) for i in labels])
    out["_report_df"] = pd.DataFrame(classification_report(
        y_true, y_pred, labels=labels, target_names=tn,
        output_dict=True, zero_division=0)).T
    out["_confusion"] = confusion_matrix(y_true, y_pred, labels=labels)
    out["_labels"] = labels
    return out


def metrics_frame(list_of_metrics, sort_by="f1_macro"):
    """Tidy comparison table; drops the private '_' keys."""
    rows = [{k: v for k, v in m.items() if not k.startswith("_")}
            for m in list_of_metrics]
    df = pd.DataFrame(rows)
    cols = [c for c in ["model", "accuracy", "balanced_accuracy", "f1_macro",
                        "f1_weighted", "precision_macro", "recall_macro",
                        "precision_weighted", "recall_weighted",
                        "roc_auc_macro_ovr", "avg_precision_macro", "mcc",
                        "cohen_kappa", "params", "size_mb", "train_seconds",
                        "inference_ms_per_image", "n_test", "n_classes"]
            if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)
    return df.reset_index(drop=True)


@torch.no_grad()
def measure_inference_ms(model, loader, device=None, n_batches=10):
    """Mean milliseconds per image at inference (brief asks for inference time)."""
    device = device or DEVICE
    model.eval().to(device)
    it = iter(loader)
    # warm-up
    try:
        xb, _ = next(it)
        model(xb.to(device))
    except StopIteration:
        return float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0, n = time.time(), 0
    for _ in range(n_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        model(xb.to(device))
        n += xb.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / max(1, n)


# =============================================================================
# 11. Plots  (every figure gets a title + axis labels, per Sec. 6.1)
# =============================================================================

def _plt():
    import matplotlib
    if not on_kaggle():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(hist_df, title="Training curves", save=None):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist_df.epoch, hist_df.train_loss, marker="o")
    ax[0].set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(hist_df.epoch, hist_df.train_acc, marker="o", label="train acc")
    ax[1].plot(hist_df.epoch, hist_df.val_acc, marker="s", label="val acc")
    ax[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    ax[1].legend()
    ax[2].plot(hist_df.epoch, hist_df.val_macro_f1, marker="d", color="tab:green")
    b = hist_df.val_macro_f1.idxmax()
    ax[2].axvline(hist_df.epoch[b], ls="--", c="grey",
                  label=f"best ep {int(hist_df.epoch[b])} = {hist_df.val_macro_f1[b]:.4f}")
    ax[2].set(title="Validation macro-F1 (model-selection metric)",
              xlabel="epoch", ylabel="macro-F1")
    ax[2].legend()
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def plot_confusion(cm, class_names, title="Confusion matrix", normalize=True,
                   save=None, figsize=None, annot_threshold=20):
    plt = _plt()
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    k = len(class_names)
    figsize = figsize or (max(7, k * 0.36), max(6, k * 0.32))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    ax.set(xticks=range(k), yticks=range(k),
           xlabel="predicted family", ylabel="true family",
           title=title + (" (row-normalised)" if normalize else " (counts)"))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    if k <= annot_threshold:
        for i in range(k):
            for j in range(k):
                if cm[i, j] > 1e-3:
                    ax.text(j, i, f"{cm[i,j]:.2f}" if normalize else f"{int(cm[i,j])}",
                            ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046,
                 label="recall" if normalize else "count")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def plot_roc_ovr(y_true, y_prob, class_names, title="ROC (one-vs-rest)",
                 save=None, max_curves=40):
    from sklearn.metrics import roc_curve, auc
    plt = _plt()
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    aucs = []
    for i, cn in enumerate(class_names[:max_curves]):
        yi = (y_true == i).astype(int)
        if yi.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yi, y_prob[:, i])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, lw=1.0, alpha=.75)
    # macro-average curve
    grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(grid)
    used = 0
    for i in range(len(class_names)):
        yi = (y_true == i).astype(int)
        if yi.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yi, y_prob[:, i])
        mean_tpr += np.interp(grid, fpr, tpr)
        used += 1
    mean_tpr /= max(1, used)
    ax.plot(grid, mean_tpr, "k-", lw=2.5,
            label=f"macro average (AUC = {np.mean(aucs):.4f})")
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate",
           title=f"{title}\nthin lines = individual families ({used} classes)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig, float(np.mean(aucs))


def plot_pr_ovr(y_true, y_prob, class_names, title="Precision-Recall (one-vs-rest)",
                save=None, max_curves=40):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    plt = _plt()
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    aps = []
    for i, cn in enumerate(class_names[:max_curves]):
        yi = (y_true == i).astype(int)
        if yi.sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(yi, y_prob[:, i])
        aps.append(average_precision_score(yi, y_prob[:, i]))
        ax.plot(rc, pr, lw=1.0, alpha=.75)
    ax.axhline(1.0 / max(1, len(class_names)), ls="--", c="r", lw=1,
               label="chance (uniform prior)")
    ax.set(xlabel="recall", ylabel="precision",
           title=f"{title}\nmacro average precision = {np.mean(aps):.4f}")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig, float(np.mean(aps))


# =============================================================================
# 12. Statistical significance  (brief Sec. 6.4 -- required)
# =============================================================================

def mcnemar_test(y_true, pred_a, pred_b, name_a="A", name_b="B",
                 exact=None, correction=True, verbose=True):
    """
    McNemar's test on ONE shared held-out test set.

    This is the correct test for "does my model beat my baseline" when both
    models produced predictions for the same samples.  It can NOT be used
    against a paper's reported number (we never see their per-sample
    predictions) -- exactly the mistake the brief warns about in Sec. 6.4.

    Contingency table
        n11  both correct        n10  only A correct
        n01  only B correct      n00  both wrong
    H0: n10 and n01 come from the same distribution (models are equivalent).
    """
    from scipy import stats
    y_true = np.asarray(y_true)
    a = np.asarray(pred_a) == y_true
    b = np.asarray(pred_b) == y_true
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    n_disc = n10 + n01
    if exact is None:
        exact = n_disc < 25
    if n_disc == 0:
        stat, p = 0.0, 1.0
        kind = "degenerate (no discordant pairs -> models made identical errors)"
    elif exact:
        stat = float(min(n10, n01))
        p = float(stats.binomtest(min(n10, n01), n_disc, 0.5,
                                  alternative="two-sided").pvalue)
        kind = "exact binomial"
    else:
        c = 1.0 if correction else 0.0
        stat = float((abs(n10 - n01) - c) ** 2 / n_disc)
        p = float(stats.chi2.sf(stat, df=1))
        kind = "chi-square" + (" with Yates correction" if correction else "")
    res = dict(test="McNemar", variant=kind, statistic=stat, p_value=p,
               n11=n11, n10=n10, n01=n01, n00=n00, n_discordant=n_disc,
               model_a=name_a, model_b=name_b,
               acc_a=float(a.mean()), acc_b=float(b.mean()),
               significant_at_0p05=bool(p < 0.05),
               better=(name_a if n10 > n01 else name_b if n01 > n10 else "tie"))
    if verbose:
        banner("McNemar test on the shared held-out test set", "-")
        print(f"  {name_a} accuracy = {a.mean():.4f}   {name_b} accuracy = {b.mean():.4f}")
        print(f"  contingency:  both correct {n11}   only {name_a} {n10}   "
              f"only {name_b} {n01}   both wrong {n00}")
        print(f"  variant     : {kind}")
        print(f"  statistic   = {stat:.4f}")
        print(f"  p-value     = {p:.6g}")
        print(f"  alpha = 0.05 -> "
              f"{'REJECT H0: the two models differ significantly' if p < 0.05 else 'FAIL TO REJECT H0: difference is not significant'}")
        print(f"  direction   : {res['better']} made more exclusive correct calls")
    return res


def wilcoxon_folds(scores_a, scores_b, name_a="A", name_b="B", verbose=True):
    """
    Wilcoxon signed-rank test over paired per-fold scores (5-fold CV).

    This is the right test when what we have is 5 paired fold scores, not
    per-sample predictions.  With n=5 the exact two-sided p-value cannot go
    below 0.0625, so we also report the one-sided p-value and Cohen's d, and we
    say so in the report rather than pretending n=5 gives strong evidence.
    """
    from scipy import stats
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape, "paired scores must have the same length"
    d = a - b
    res = dict(test="Wilcoxon signed-rank", n_folds=int(len(a)),
               model_a=name_a, model_b=name_b,
               mean_a=float(a.mean()), std_a=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
               mean_b=float(b.mean()), std_b=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
               mean_diff=float(d.mean()))
    if np.allclose(d, 0):
        res.update(statistic=0.0, p_value=1.0, p_value_one_sided=1.0,
                   note="all fold differences are zero")
    else:
        w = None
        for kw in ({"method": "exact"}, {"mode": "exact"}, {}):
            try:                       # scipy renamed mode= -> method= in 1.9
                w = stats.wilcoxon(a, b, alternative="two-sided",
                                   zero_method="wilcox", **kw)
                break
            except TypeError:
                continue
        if w is None:
            w = stats.wilcoxon(a, b, alternative="two-sided")
        try:
            w1 = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
            p1 = float(w1.pvalue)
        except Exception:
            p1 = float("nan")
        res.update(statistic=float(w.statistic), p_value=float(w.pvalue),
                   p_value_one_sided=p1)
    # paired t-test as a secondary read, plus effect size
    if len(a) > 1 and not np.allclose(d, 0):
        t = stats.ttest_rel(a, b)
        res["paired_t_statistic"] = float(t.statistic)
        res["paired_t_p_value"] = float(t.pvalue)
        res["cohens_d_paired"] = float(d.mean() / (d.std(ddof=1) + 1e-12))
    res["significant_at_0p05"] = bool(res["p_value"] < 0.05)
    res["min_possible_two_sided_p"] = float(2 / 2 ** len(a)) if len(a) <= 10 else 0.0
    if verbose:
        banner(f"Wilcoxon signed-rank over {len(a)} folds: {name_a} vs {name_b}", "-")
        for i, (x, y) in enumerate(zip(a, b), 1):
            print(f"    fold {i}: {name_a} {x:.4f}   {name_b} {y:.4f}   diff {x-y:+.4f}")
        print(f"  {name_a}: {a.mean():.4f} +- {res['std_a']:.4f}")
        print(f"  {name_b}: {b.mean():.4f} +- {res['std_b']:.4f}")
        print(f"  W statistic = {res['statistic']:.4f}")
        print(f"  p (two-sided) = {res['p_value']:.6g}    "
              f"p (one-sided, A>B) = {res.get('p_value_one_sided', float('nan')):.6g}")
        if "paired_t_p_value" in res:
            print(f"  paired t-test: t = {res['paired_t_statistic']:.4f}, "
                  f"p = {res['paired_t_p_value']:.6g}, "
                  f"Cohen's d = {res['cohens_d_paired']:.3f}")
        print(f"  alpha = 0.05 -> "
              f"{'significant' if res['significant_at_0p05'] else 'NOT significant'}")
        print(f"  note: with n={len(a)} folds the smallest attainable two-sided "
              f"p is {res['min_possible_two_sided_p']:.4f}")
    return res


def friedman_nemenyi(score_matrix, model_names, verbose=True):
    """
    Friedman test + Nemenyi post-hoc for 3+ models over the same folds
    (Demsar, JMLR 2006).  `score_matrix` is (n_folds, n_models).

    Nemenyi critical difference:  CD = q_alpha * sqrt(k(k+1) / (6N))
    """
    from scipy import stats
    S = np.asarray(score_matrix, dtype=float)
    N, k = S.shape
    assert k == len(model_names) and k >= 3, "need >=3 models, one column each"
    chi2, p = stats.friedmanchisquare(*[S[:, j] for j in range(k)])
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, S)  # 1 = best
    avg_rank = ranks.mean(axis=0)

    # studentised range q_0.05 / sqrt(2), Demsar Table 5
    Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
           8: 3.031, 9: 3.102, 10: 3.164}
    q = Q05.get(k, 3.164 + 0.03 * (k - 10))
    CD = q * math.sqrt(k * (k + 1) / (6.0 * N))

    pair = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg_rank[i] - avg_rank[j])
            pair.append(dict(model_a=model_names[i], model_b=model_names[j],
                             rank_diff=float(diff),
                             exceeds_CD=bool(diff > CD),
                             better=model_names[i] if avg_rank[i] < avg_rank[j]
                             else model_names[j]))
    res = dict(test="Friedman + Nemenyi", n_folds=int(N), n_models=int(k),
               chi2_statistic=float(chi2), p_value=float(p),
               significant_at_0p05=bool(p < 0.05),
               average_ranks={m: float(r) for m, r in zip(model_names, avg_rank)},
               critical_difference=float(CD), pairwise=pair)
    if verbose:
        banner(f"Friedman test over {N} folds, {k} models", "-")
        print(f"  chi2 = {chi2:.4f}   p = {p:.6g}   -> "
              f"{'at least one model differs' if p < 0.05 else 'no detectable difference'}")
        print("  average ranks (1 = best):")
        for m, r in sorted(res["average_ranks"].items(), key=lambda t: t[1]):
            print(f"    {r:5.2f}  {m}")
        print(f"  Nemenyi critical difference (alpha=0.05) = {CD:.3f}")
        for pr in pair:
            mark = "SIGNIFICANT" if pr["exceeds_CD"] else "not significant"
            print(f"    {pr['model_a']} vs {pr['model_b']}: "
                  f"|dRank| = {pr['rank_diff']:.2f}  -> {mark}")
        if N < 10:
            print(f"  note: Nemenyi is conservative with only N={N} folds; treat "
                  f"non-significant pairs as 'not resolved', not 'identical'.")
    return res


# =============================================================================
# 13. Explainability  (brief Sec. 6.2 + Track-3 Task 3)
# =============================================================================
#
# HONEST LIMITATION, stated up front because the brief demands it (Sec. 6.2):
#
#   "Malware byte-images: they classify well, but Grad-CAM/LIME on them means
#    little (no real 'regions' to point at). Say this instead of inventing an
#    explanation."
#
# We therefore do three things rather than one:
#   (a) run Grad-CAM and LIME as required, and report them WITHOUT claiming the
#       highlighted blobs are semantically meaningful objects;
#   (b) add a byte-offset reading that IS meaningful: because row r of a WxH
#       byte-plot covers file bytes [r*W, (r+1)*W), a heat-map row profile maps
#       back to a *relative file-offset band*.  "The model relied on bytes at
#       12-19% of the file" is a checkable statement about PE layout, unlike
#       "the model looked at this blob";
#   (c) run a row-band occlusion test, which measures causal importance instead
#       of inferring it from gradients, so we can say whether the Grad-CAM story
#       actually holds.

class GradCAM:
    """
    Minimal Grad-CAM (Selvaraju et al., ICCV 2017), no external dependency.

    cam = ReLU( sum_k  alpha_k * A_k ),   alpha_k = GAP( dY_c / dA_k )
    Grad-CAM++ weights are also available (alpha from second-order terms), which
    behaves better when several disjoint regions support the same class -- the
    common case for byte-plots.
    """

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.layer = target_layer if target_layer is not None else gradcam_target(model)
        self.acts = None
        self.grads = None
        self._h = [self.layer.register_forward_hook(self._fwd)]
        try:
            self._h.append(self.layer.register_full_backward_hook(self._bwd))
        except AttributeError:
            self._h.append(self.layer.register_backward_hook(self._bwd))

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def remove(self):
        for h in self._h:
            h.remove()

    def __call__(self, x, class_idx=None, plus_plus=False):
        """x: (1,C,H,W) tensor. Returns (cam HxW in [0,1], class_idx, prob)."""
        self.model.zero_grad(set_to_none=True)
        was = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            logits = self.model(x)
            prob = torch.softmax(logits, 1)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            logits[0, class_idx].backward()
        finally:
            torch.set_grad_enabled(was)

        A, G = self.acts[0], self.grads[0]                 # (K,h,w)
        if plus_plus:
            g2, g3 = G.pow(2), G.pow(3)
            denom = 2 * g2 + g3 * A.sum(dim=(1, 2), keepdim=True)
            alpha = (g2 / denom.clamp(min=1e-9))
            w = (alpha * F.relu(G)).sum(dim=(1, 2))
        else:
            w = G.mean(dim=(1, 2))
        cam = F.relu((w[:, None, None] * A).sum(0))
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-9)
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                            align_corners=False)[0, 0]
        return cam.cpu().numpy(), class_idx, float(prob[0, class_idx].item())


def cam_to_byte_offsets(cam, img_width_px=256, top_frac=0.25):
    """
    Turn a Grad-CAM map into a statement about *file byte offsets*.

    Row r of a byte-plot with pixel width W holds bytes [r*W, (r+1)*W) of the
    original binary.  So the row-mean of the CAM is an importance profile over
    relative file position, and that IS interpretable (PE header at the top,
    code sections next, resources/padding at the end).

    Returns dict with the row profile, the contiguous top bands, and each band
    expressed as a percentage range of the file.
    """
    cam = np.asarray(cam, dtype=float)
    h = cam.shape[0]
    prof = cam.mean(axis=1)
    prof = (prof - prof.min()) / (np.ptp(prof) + 1e-9)
    thr = np.quantile(prof, 1.0 - top_frac)
    hot = prof >= thr
    bands, start = [], None
    for i, v in enumerate(hot):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, h - 1))
    out_bands = [dict(row_from=int(a), row_to=int(b),
                      file_pct_from=round(100.0 * a / h, 1),
                      file_pct_to=round(100.0 * (b + 1) / h, 1),
                      mean_importance=float(prof[a:b + 1].mean()))
                 for a, b in bands]
    out_bands.sort(key=lambda d: -d["mean_importance"])
    return dict(row_profile=prof, bands=out_bands,
                col_profile=(lambda c: (c - c.min()) / (np.ptp(c) + 1e-9))(cam.mean(axis=0)))


@torch.no_grad()
def occlusion_by_row_band(model, x, class_idx, n_bands=16, fill=0.0, device=None):
    """
    Causal check on the Grad-CAM story: blank out one horizontal band of the
    byte-plot at a time and record how far the target-class probability drops.
    A band = a contiguous slice of file offsets, so the result is directly
    comparable with `cam_to_byte_offsets`.

    Returns (drops array of length n_bands, baseline probability).
    """
    device = device or DEVICE
    model.eval().to(device)
    x = x.to(device)
    base = torch.softmax(model(x), 1)[0, class_idx].item()
    H = x.shape[-2]
    edges = np.linspace(0, H, n_bands + 1).astype(int)
    drops = np.zeros(n_bands)
    for i in range(n_bands):
        xo = x.clone()
        xo[..., edges[i]:edges[i + 1], :] = fill
        p = torch.softmax(model(xo), 1)[0, class_idx].item()
        drops[i] = base - p
    return drops, base


def lime_explain(model, img_uint8, class_names, in_channels=1, out_size=None,
                 top_labels=3, num_samples=400, num_features=8,
                 segments="grid", device=None, seed=None):
    """
    LIME image explanation (Ribeiro et al., KDD 2016).

    Segmentation choice, and why it matters here
    -------------------------------------------
    LIME's default `quickshift` segmentation looks for *colour-coherent
    regions*, which is a natural-image assumption.  A byte-plot has no such
    regions, so quickshift produces essentially arbitrary blobs and the
    explanation is noise dressed up as insight.  We therefore default to a
    regular ROW-BAND grid: every superpixel is a contiguous slice of file
    offsets, which is the only partition of a byte-plot that has a meaning.
    Pass segments='quickshift' to reproduce the naive version for comparison.
    """
    from lime import lime_image
    device = device or DEVICE
    out_size = int(out_size or CFG.img_size)
    seed = CFG.seed if seed is None else seed
    model.eval().to(device)

    a = np.asarray(img_uint8, dtype=np.uint8)
    rgb = np.stack([a] * 3, axis=-1)                # LIME needs HxWx3

    def batch_predict(images):
        xs = []
        for im in images:
            g = im[..., 0].astype(np.float32) / 255.0
            t = torch.from_numpy(g)[None, None]
            if t.shape[-1] != out_size:
                t = F.interpolate(t, size=(out_size, out_size), mode="bilinear",
                                  align_corners=False)
            t = (t - t.mean()) / (t.std() + 1e-5)
            if in_channels == 3:
                t = t.repeat(1, 3, 1, 1)
            xs.append(t[0])
        xb = torch.stack(xs).to(device)
        with torch.no_grad():
            return torch.softmax(model(xb), 1).cpu().numpy()

    if segments == "grid":
        n_rows, n_cols = 16, 4
        H, W = a.shape
        seg = np.zeros((H, W), dtype=int)
        re_ = np.linspace(0, H, n_rows + 1).astype(int)
        ce_ = np.linspace(0, W, n_cols + 1).astype(int)
        k = 0
        for i in range(n_rows):
            for j in range(n_cols):
                seg[re_[i]:re_[i + 1], ce_[j]:ce_[j + 1]] = k
                k += 1
        seg_fn = (lambda image: seg)
    else:
        seg_fn = None

    expl = lime_image.LimeImageExplainer(random_state=seed)
    kw = dict(top_labels=top_labels, hide_color=0, num_samples=int(num_samples),
              batch_size=32)
    if seg_fn is not None:
        kw["segmentation_fn"] = seg_fn
    ex = expl.explain_instance(rgb, batch_predict, **kw)
    return ex


# =============================================================================
# 14. Related-work table (Task 1)  -- five+ peer-reviewed papers, 2022-2026
# =============================================================================
#
# All numbers below were read from the papers' own abstracts / results tables.
# `comparable` marks the rows we are allowed to compare against under Sec. 6.4:
# same dataset family (Malimg), image input, multi-class family classification.
# Rows marked comparable=False differ in dataset or task and are context only.

RELATED_WORK = [
    dict(
        key="PAFE2024",
        title="PAFE: A lightweight visualization-based fast malware classification method",
        authors="S. Li, J. Wang, S. Wang, Y. Song",
        year=2024, venue="Heliyon 10, e35965", doi="10.1016/j.heliyon.2024.e35965",
        dataset="Malimg (25 families, 9,435 images)",
        application="Windows PE malware family classification from grayscale byte-plots",
        method="CNN with FFSE blocks = multi-scale feature fusion + Squeeze-and-Excitation "
               "channel attention; pixel-padding resize instead of interpolation; 256x256 input",
        attention_type="Channel (Squeeze-and-Excitation) inside a multi-scale fusion block",
        metrics="Accuracy 99.25%, Precision 99.29%, Recall 99.25%, F1 99.27%, "
                "inference 10.04 ms, 721,913 params",
        headline_metric="F1", headline_value=99.27,
        strengths="Best published Malimg accuracy/latency trade-off; tiny (0.72M params); "
                  "pixel-padding avoids the texture distortion that bilinear resizing causes; "
                  "reports timing and parameter count, not just accuracy",
        limitations="Random (not source-grouped) split, so polymorphic near-duplicates can "
                    "straddle train/test; no macro-F1 or per-class recall on the rare families; "
                    "authors themselves note generalisation to new variants is unverified; "
                    "no explainability",
        gap="Leakage-controlled evaluation and per-class (macro) reporting are missing; "
            "attention is channel-only, so byte-offset position is discarded",
        relation="Our Pillar-A target. We reproduce the same Malimg 25-family task as a "
                 "subset of MaleBin and compare F1, but under a duplicate-grouped split",
        comparable=True,
    ),
    dict(
        key="DRIN2024",
        title="Attention-Based Malware Detection Model by Visualizing Latent Features "
              "Through Dynamic Residual Kernel Network",
        authors="M. Basak, D.-W. Kim, M.-M. Han, G.-Y. Shin",
        year=2024, venue="Sensors 24(24), 7953", doi="10.3390/s24247953",
        dataset="Custom (25 families, 49,374), Malimg, MaleVis",
        application="Malware family classification from visualised binaries",
        method="Dynamic Residual Involution Network (DRIN): involution kernels that are "
               "spatially specific and channel-agnostic, i.e. attention baked into the kernel",
        attention_type="Involution (spatially specific, channel-agnostic) + residual",
        metrics="Malimg: Acc 99.3%, P 0.992, R 0.989, F1 0.9905 | "
                "MaleVis: Acc 98.9%, F1 0.9892 | Custom: Acc 99.5%, F1 0.9948",
        headline_metric="F1", headline_value=99.05,
        strengths="Attention is intrinsic to the kernel rather than bolted on; validated on "
                  "three datasets; heat-map visualisation attempted",
        limitations="Authors state it still struggles on under-represented classes; heavier "
                    "than lightweight CNNs; sensitive to preprocessing noise; scalability to "
                    "unseen families untested; interpretability admitted to be limited",
        gap="The under-represented-class weakness is exactly what macro-F1 exposes and what "
            "class-weighted training plus a balanced dataset can address",
        relation="Second comparable Malimg baseline; its admitted rare-class weakness "
                 "motivates our macro-F1-driven model selection and class-weighted loss",
        comparable=True,
    ),
    dict(
        key="SEAGM2023",
        title="Transfer Learning for Image-Based Malware Detection for IoT (SE-AGM)",
        authors="P. Panda, C. U. Om Kumar, S. Marappan, M. Suresh, S. Manimurugan, "
                "D. Veesani Nandi",
        year=2023, venue="Sensors 23(6), 3253", doi="10.3390/s23063253",
        dataset="Malimg (25 families)",
        application="IoT malware detection from byte-plot images",
        method="Stacked ensemble of autoencoder + GRU + MLP over 25 CNN-extracted features; "
               "each stage's output feeds the next; data augmentation studied",
        attention_type="None (stacked ensemble, not attention)",
        metrics="Average accuracy 99.43% on Malimg",
        headline_metric="Accuracy", headline_value=99.43,
        strengths="Highest reported Malimg accuracy in our set; very cheap at inference "
                  "because it classifies only 25 encoded features; ablates augmentation",
        limitations="Reports accuracy only -- no macro-F1, no per-class recall, no confusion "
                    "matrix on the rare families, so the number cannot be checked against "
                    "the imbalance rule; feature extractor trained on the same data it later "
                    "encodes; random split",
        gap="An accuracy-only headline on an imbalanced 25-class set is exactly what Sec. 6.2 "
            "warns about; the result is not verifiable per class",
        relation="Highest accuracy number we must acknowledge, but it is accuracy-only, so we "
                 "compare our accuracy to it and explain why macro-F1 is the fairer basis",
        comparable=True,
    ),
    dict(
        key="Hybrid2025",
        title="An Explainable Hybrid CNN-Transformer Architecture for Visual Malware "
              "Classification",
        authors="M. Alshomrani, A. Albeshri, A. A. Alsulami, B. Alturki",
        year=2025, venue="Sensors 25(15), 4581", doi="10.3390/s25154581",
        dataset="Malimg + MaleVis + VirusMNIST combined (61 classes); also Maldeb, "
                "Dumpware-10",
        application="Visual malware classification across merged sources",
        method="ConvNeXt-Tiny (local features) fused with Swin Transformer (global context); "
               "Grad-CAM for interpretability; real-time deployment demo",
        attention_type="Self-attention (shifted-window Swin) + convolutional local features",
        metrics="Combined 61-class validation accuracy 94.04% (ConvNeXt-Tiny alone 92.45%, "
                "Swin alone 90.44%); Maldeb 98%; Dumpware-10 97%",
        headline_metric="Accuracy", headline_value=94.04,
        strengths="The only paper in our set that evaluates a *merged multi-source* label "
                  "space, which is what MaleBin is; uses Grad-CAM and discusses it; shows "
                  "the hybrid beats either half",
        limitations="Accuracy on a validation split rather than an untouched test set; no "
                    "macro-F1 on 61 imbalanced classes; Grad-CAM interpreted without "
                    "acknowledging that byte-plots have no semantic regions; heavy backbones",
        gap="Merged-source label spaces drop ~5 points versus single-source Malimg, and nobody "
            "reports macro-F1 there; also no duplicate control across merged sources",
        relation="Closest analogue to our 39-class MaleBin setting (merged sources, more "
                 "classes). This is the paper our full-MaleBin number is compared against",
        comparable=True,
    ),
    dict(
        key="MalVis2025",
        title="MalVis: A Large-Scale Image-Based Framework and Dataset for Advancing "
              "Android Malware Classification",
        authors="S. J. Makkawy, M. J. De Lucia, K. E. Barner",
        year=2025, venue="arXiv:2505.12106", doi="10.48550/arXiv.2505.12106",
        dataset="MalVis (>1.3M images, 9 malware classes + benign)",
        application="Android malware classification from bytecode visualisations",
        method="Entropy + N-gram enhanced visualisation; MobileNetV2 / DenseNet201 / "
               "ResNet50 / InceptionV3 with eight ensemble strategies; undersampling",
        attention_type="None",
        metrics="Accuracy 95.19%, macro-F1 90.81%, Precision 92.58%, Recall 89.10%, "
                "MCC 87.58%, ROC-AUC 98.06%",
        headline_metric="macro-F1", headline_value=90.81,
        strengths="Reports macro-F1, MCC and ROC-AUC -- the honest metric set for imbalanced "
                  "data; huge scale; explicit imbalance handling",
        limitations="Android bytecode, not Windows PE byte-plots, so not directly comparable; "
                    "undersampling discards data; no attention module",
        gap="Shows how far macro-F1 sits below accuracy on imbalanced visual malware data "
            "(95.19 vs 90.81) -- a gap the PE-image papers never report",
        relation="Context, not a comparison target. It is our evidence that macro-F1 is the "
                 "right headline metric and that accuracy overstates performance",
        comparable=False,
    ),
    dict(
        key="Byteplot2023",
        title="Comparative Analysis of Imbalanced Malware Byteplot Image Classification "
              "using Transfer Learning",
        authors="Jayasudha M, A. Shaik, G. Pendharkar, S. Kumar, Muhesh Kumar B, "
                "S. Balaji",
        year=2023, venue="PEIS 2023, Lecture Notes in Electrical Engineering; "
                         "arXiv:2310.02742",
        doi="10.48550/arXiv.2310.02742",
        dataset="Malimg, a blended dataset, and MaleVis (three imbalance levels)",
        application="Byte-plot malware classification under class imbalance",
        method="Six multi-class transfer-learning models; ResNet50, EfficientNetB0 and "
               "DenseNet169 were the strongest",
        attention_type="None (pure transfer learning)",
        metrics="Max precision 97% (imbalanced), 95% (intermediate), 95% (balanced); "
                "more imbalance -> faster convergence but higher variance across models",
        headline_metric="Precision", headline_value=97.0,
        strengths="Directly studies the imbalance axis; uses a blended (multi-source) dataset "
                  "like MaleBin; documents the convergence/variance trade-off",
        limitations="Precision-only headline; no macro-F1 or per-class recall; no attention "
                    "or custom architecture; no duplicate control across the blend",
        gap="Blended multi-source byte-plot data is under-studied and reported with the wrong "
            "metric",
        relation="Justifies our baseline pool (ResNet50, DenseNet121, EfficientNet-B0) and our "
                 "decision to report variance across folds, not a single number",
        comparable=False,
    ),
    dict(
        key="IMCMK2024",
        title="IMCMK-CNN: A lightweight convolutional neural network with Multi-scale "
              "Kernels for Image-based Malware Classification",
        authors="D. Zhang, Y. Song, Q. Xiang, Y. Wang",
        year=2024, venue="Alexandria Engineering Journal 111, 203-220",
        doi="10.1016/j.aej.2024.10.055",
        dataset="Malimg and other image-based malware sets",
        application="Malware variant classification from byte-plot images",
        method="Multi-scale Kernel (MK) block mixing large and small kernels plus an improved "
               "Squeeze-and-Excitation block; fusion strategy keeps the parameter cost of "
               "small kernels",
        attention_type="Improved Squeeze-and-Excitation channel attention",
        metrics="[FILL FROM PDF -- read the results table of the published version and "
                "replace this string before you submit Task 1]",
        headline_metric="Accuracy", headline_value=float("nan"),
        strengths="Directly motivates multi-scale kernels for byte texture; explicitly targets "
                  "the parameter cost of large kernels",
        limitations="Channel attention only, so byte-offset position is not modelled; "
                    "single-source evaluation",
        gap="Multi-scale + channel attention is established; direction-aware (positional) "
            "attention for byte-plots is not",
        relation="The architectural ancestor of our multi-scale dilated block. We keep its "
                 "multi-scale idea and add the positional attention it lacks",
        comparable=True,
    ),
]


def related_work_frame(only_comparable=False):
    cols = ["key", "title", "authors", "year", "venue", "doi", "dataset",
            "application", "method", "attention_type", "metrics",
            "headline_metric", "headline_value", "strengths", "limitations",
            "gap", "relation", "comparable"]
    rows = [r for r in RELATED_WORK if (r["comparable"] or not only_comparable)]
    return pd.DataFrame(rows)[cols]


def best_comparable_target(scope=None):
    """
    The single number Pillar A is judged against, chosen honestly per scope.

    scope='malimg25'  -> PAFE 2024, F1 99.27 on the very same 25 families.
    scope='malebin39' -> Alshomrani et al. 2025, 94.04% accuracy on a merged
                         61-class multi-source visual-malware label space, the
                         closest published analogue to MaleBin's merged 39.
    Anything else would be an unfair comparison under Sec. 6.4.
    """
    scope = scope or CFG.eval_scope
    if scope == "malimg25":
        r = next(x for x in RELATED_WORK if x["key"] == "PAFE2024")
        return dict(paper=r["key"], citation=f"{r['authors']} ({r['year']}), {r['venue']}",
                    metric="F1 (weighted) on Malimg 25 families", value=99.27,
                    caveat="Their split is random; ours is duplicate-grouped, which is "
                           "strictly harder. Same dataset, same 25 classes, same input type.")
    r = next(x for x in RELATED_WORK if x["key"] == "Hybrid2025")
    return dict(paper=r["key"], citation=f"{r['authors']} ({r['year']}), {r['venue']}",
                metric="Accuracy on a merged 61-class visual-malware set", value=94.04,
                caveat="Different merged corpus (Malimg+MaleVis+VirusMNIST, 61 classes) vs "
                       "MaleBin (39 classes). Not the same dataset, so this is an "
                       "indicative comparison and we label it as such. The strictly fair "
                       "comparison for this project is the malimg25 scope.")
