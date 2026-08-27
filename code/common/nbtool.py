"""
nbtool.py -- tiny notebook builder used by the make_notebooks_*.py generators.

Each task notebook is written as a list of ("md"|"code", source) pairs and
serialised to .ipynb with nbformat.  Cell 2 of every notebook is an automatic
`%%writefile malebin_common.py` cell carrying the whole shared module, so each
notebook is self-contained on Kaggle and does not depend on any other
notebook's output.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
COMMON = HERE / "malebin_common.py"
REPO = HERE.parent.parent            # Group00_MaleBin_CSE475/

PREFIX = "Group00_MaleBin"


def writefile_cell() -> tuple[str, str]:
    """The %%writefile cell that materialises malebin_common.py on Kaggle."""
    src = COMMON.read_text(encoding="utf-8")
    return ("code", "%%writefile malebin_common.py\n" + src)


RUN_NOTE = """\
> ### ⚠️ Before you submit: this notebook must be saved **with its outputs**
>
> The course requires the notebook to show its results. On Kaggle:
>
> 1. **Add Data** → search `MaleBin malware binary greyscale` → **Add**.
> 2. **Settings** → *Accelerator*: **GPU P100 / T4**.
> 3. Set your group number in the boot cell (`CFG.group = "Group00"`).
> 4. **Run All**, then **Save Version → Save & Run All (Commit)**.
> 5. Download that committed version — it contains every table, figure and
>    printed number — and push *that* file to the repo.
>
> For a ~3-minute wiring check first, uncomment `CFG.fast = True` in the boot
> cell. **FAST-mode numbers are meaningless** (64 px, 2 epochs); turn it off
> before producing anything you will quote or submit.
"""

BOOT = f'''\
# ---------------------------------------------------------------- environment
import os, sys, json, time, math, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())            # so `import malebin_common` finds it
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({{"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": .3, "font.size": 10}})

import malebin_common as M
from malebin_common import CFG

# ---- EDIT THESE ------------------------------------------------------------
CFG.group = "Group00"          # <-- your group number
CFG.dataset_slug = "MaleBin"
# CFG.fast = True              # <-- uncomment for a ~3-min wiring check
# CFG.eval_scope = "malimg25"  # <-- uncomment for the strictly fair comparison
# ---------------------------------------------------------------------------
CFG.__post_init__()
M.set_seed(CFG.seed)

print("python      :", sys.version.split()[0])
import torch
print("torch       :", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
print("numpy       :", np.__version__, "| pandas:", pd.__version__)
print("on kaggle   :", M.on_kaggle())
print("output dir  :", CFG.out_dir)
print("FAST mode   :", CFG.fast, "(set MALEBIN_FAST=1 to enable)")
print("img_size    :", CFG.img_size, "| cache", CFG.cache_size,
      "| epochs", CFG.epochs, "| folds", CFG.n_folds)
print("eval scope  :", CFG.eval_scope)
'''

LOAD_SPLIT = '''\
# ---------------------------------------------------------------- data + split
# Every notebook re-derives the SAME split from scratch.  It is deterministic
# (fixed seed + content-based duplicate groups), so notebook 3 evaluates on
# exactly the test set notebook 2 held out, with no artefact passing required.
ROOT = M.find_dataset_root()
df_all = M.scan_index(ROOT)
df = M.tag_malimg_subset(df_all, CFG.eval_scope)

if CFG.max_per_class:                       # smoke-test subsample only
    df = (df.groupby("family", group_keys=False)
            .apply(lambda g: g.head(CFG.max_per_class))
            .reset_index(drop=True))
    print(f"[subsampled] {len(df):,} images")

CLASS_NAMES = sorted(df.family.unique())
N_CLASSES = len(CLASS_NAMES)

imgs = M.load_images(df, side=CFG.cache_size)

M.banner("Leakage control: duplicate grouping (brief Sec. 6.2)")
groups = M.build_dedup_groups(imgs, df)

M.banner("Grouped + stratified hold-out split")
tr_idx, va_idx, te_idx = M.grouped_holdout_split(df, groups)

split_manifest = df[["path", "family", "label"]].copy()
split_manifest["dup_group"] = groups
split_manifest["subset"] = "train"
split_manifest.loc[va_idx, "subset"] = "val"
split_manifest.loc[te_idx, "subset"] = "test"
split_manifest.to_csv(CFG.art("split_manifest.csv"), index=False)
M.save_json({"class_names": CLASS_NAMES,
             "label_to_family": {i: c for i, c in enumerate(CLASS_NAMES)},
             "family_to_label": {c: i for i, c in enumerate(CLASS_NAMES)},
             "eval_scope": CFG.eval_scope, "n_classes": N_CLASSES,
             "n_images": int(len(df)), "seed": CFG.seed,
             "dhash_threshold": CFG.dhash_threshold},
            CFG.mdl("label_map.json"))
print(f"\\nsplit manifest -> {CFG.art('split_manifest.csv')}")
print(f"label map      -> {CFG.mdl('label_map.json')}")
'''


def validate_code(src: str, where: str) -> None:
    """
    Guard rails for code that must run on Kaggle's **Python 3.11**.

    1. `ast.parse` catches ordinary syntax errors at generation time rather than
       three cells into a 40-minute GPU run.
    2. A backslash inside an f-string expression is a SyntaxError before 3.12.
       Generating this file with 3.12+ would silently accept it, so we ban
       backslash-escaped quotes outright -- nothing here legitimately needs one.
    """
    import ast
    for line_no, line in enumerate(src.split("\n"), 1):
        stripped = line.lstrip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue                       # IPython magics are not Python
        if "\\'" in line or '\\"' in line:
            raise SyntaxError(
                f"{where} line {line_no}: backslash-escaped quote. This is a "
                f"SyntaxError inside an f-string on Python 3.11 (Kaggle). "
                f"Rewrite with a temporary variable.\n  {line}")
    body = "\n".join(l for l in src.split("\n")
                     if not l.lstrip().startswith(("%", "!")))
    try:
        ast.parse(body)
    except SyntaxError as e:
        raise SyntaxError(f"{where}: {e}\n  offending line: "
                          f"{(body.split(chr(10))[e.lineno-1] if e.lineno else '?')}") from None


def strip_comments(src: str) -> str:
    """
    Remove `#` comments from a code cell using `tokenize`, so a `#` inside a
    string literal is never touched.  Blank lines left behind are collapsed.
    IPython magic lines (`%`, `!`) are passed through untouched.
    """
    import io
    import tokenize

    head, body = [], src
    lines = src.split("\n")
    while lines and lines[0].lstrip().startswith(("%", "!")):
        head.append(lines.pop(0))
    body = "\n".join(lines)
    if not body.strip():
        return "\n".join(head)

    try:
        toks = list(tokenize.generate_tokens(io.StringIO(body).readline))
    except (tokenize.TokenError, IndentationError):
        return src                                  # leave it alone if unparsable

    drop = {t.start[0] for t in toks
            if t.type == tokenize.COMMENT and not body.split("\n")[t.start[0] - 1]
            [:t.start[1]].strip()}                  # whole-line comments
    out = []
    for i, line in enumerate(body.split("\n"), 1):
        if i in drop:
            continue
        for t in toks:                              # trailing comments
            if t.type == tokenize.COMMENT and t.start[0] == i:
                line = line[:t.start[1]].rstrip()
                break
        out.append(line)

    cleaned, prev_blank = [], True
    for line in out:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = blank
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return "\n".join(head + cleaned)


def build(cells: list[tuple[str, str]], out_path: Path, title: str,
          keep_comments: bool = False) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = []
    if cells and cells[0][0] == "md":
        cells = [cells[0], ("md", RUN_NOTE)] + list(cells[1:])
    for n, (kind, src) in enumerate(cells):
        src = src.rstrip("\n")
        if kind == "code":
            if not keep_comments:
                src = strip_comments(src)
            validate_code(src, f"{out_path.name} cell {n}")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md"
                        else nbf.v4.new_code_cell(src))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "title": title,
        "accelerator": "GPU",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))
    print(f"wrote {out_path}  ({len(nb.cells)} cells)")
    return out_path
