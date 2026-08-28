"""
run_notebooks.py -- execute the five task notebooks end-to-end and write the
executed copies **without** their embedded figures.

Why this exists
---------------
The notebooks save every figure to ``figures/`` with a meaningful name *and*
call ``plt.show()``.  On Kaggle that is what you want: the committed notebook
carries its plots inline.  Locally it is not -- the Task-1 notebook alone came
out at 12 MB of base64 PNG and Plotly bundles.

With ``MALEBIN_NO_INLINE=1`` the shared library turns ``plt.show()`` into
"save, report the filename, close", so the executed notebook keeps every table,
metric and printed line but none of the image payload.  This driver then does a
second, belt-and-braces pass over the finished notebook and drops any image
output that slipped through (``display(fig)``, IPython ``Image``, Plotly's
mime bundle), leaving a short text placeholder that names the file on disk.

Usage
-----
    python run_notebooks.py --profile cpu2h
    python run_notebooks.py --profile cpu2h --only task1 task2a
    python run_notebooks.py --list-profiles

Every knob is passed to the notebooks through environment variables, so the
notebooks themselves stay exactly as they ship to Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                      # Group12_MaleBin_ICE478/

NOTEBOOKS = [
    ("task1",  REPO / "code/task1/Group12_MaleBin_task1_eda.ipynb"),
    ("task2a", REPO / "code/task2/Group12_MaleBin_task2_baselines.ipynb"),
    ("task2b", REPO / "code/task2/Group12_MaleBin_task2_proposed_model.ipynb"),
    ("task3a", REPO / "code/task3/Group12_MaleBin_task3_improvement_ablation.ipynb"),
    ("task3b", REPO / "code/task3/Group12_MaleBin_task3_explainability.ipynb"),
]

# ---------------------------------------------------------------------------
# Budget profiles
# ---------------------------------------------------------------------------
# "cpu2h" is sized from measured throughput on a 12-core CPU with no GPU: the
# whole five-notebook chain, on the real 12,464-image dataset, inside ~2 hours.
# It trades resolution and epochs for wall-clock -- see REAL_RUN.md for exactly
# what that costs and which claims it can and cannot support.
PROFILES: dict[str, dict[str, str]] = {
    # Sized from measured throughput on this 12-core CPU at 64 px, 8,698
    # training images:  ByteAttnNet 1.91 min/epoch, SimpleCNN 1.46,
    # ResNet50 4.68, DenseNet121 4.62, EfficientNet-B0 4.01.
    "cpu2h": {
        "MALEBIN_IMG_SIZE": "64",
        "MALEBIN_CACHE_SIZE": "96",
        "MALEBIN_EPOCHS": "5",
        "MALEBIN_FOLDS": "2",
        "MALEBIN_BATCH": "64",
        "MALEBIN_PATIENCE": "3",
        "MALEBIN_WORKERS": "0",
        "MALEBIN_AMP": "0",
        "MALEBIN_BASELINES": "SimpleCNN,ResNet50,MobileNetV3-Small",
        "MALEBIN_ABLATION_EPOCHS": "2",
        "MALEBIN_ABLATION_GROUPS": "attention,augmentation",
        "MALEBIN_CV_EPOCHS": "2",
        "MALEBIN_RUN_ABLATION": "1",
        "MALEBIN_RUN_FINAL": "1",
        "MALEBIN_RUN_CV": "1",
    },
    "cpu6h": {
        "MALEBIN_IMG_SIZE": "96",
        "MALEBIN_CACHE_SIZE": "128",
        "MALEBIN_EPOCHS": "10",
        "MALEBIN_FOLDS": "3",
        "MALEBIN_BATCH": "64",
        "MALEBIN_PATIENCE": "4",
        "MALEBIN_WORKERS": "0",
        "MALEBIN_AMP": "0",
        "MALEBIN_BASELINES": "SimpleCNN,ResNet50,DenseNet121,EfficientNet-B0",
        "MALEBIN_ABLATION_EPOCHS": "4",
        "MALEBIN_ABLATION_GROUPS": "attention,augmentation,pooling,multiscale",
    },
    # Second pass, after cpu2h showed ByteAttnNet was still climbing at epoch 5
    # (val macro-F1 0.020 -> 0.145 -> 0.326 -> 0.354 -> 0.365) while the OneCycle
    # schedule had already annealed the LR to ~0.  Two consequences, both fixed
    # here: the head-to-head was measuring an undertrained network, and a
    # 2-epoch ablation would have compared variants sitting at ~0.145 macro-F1,
    # i.e. noise.  This profile spends the budget on epochs instead of breadth:
    # 15 epochs for the proposed model, 4 per ablation variant, attention only,
    # and no cross-validation (2 folds cannot produce a usable p-value anyway).
    "fair": {
        "MALEBIN_IMG_SIZE": "64",
        "MALEBIN_CACHE_SIZE": "96",
        "MALEBIN_EPOCHS": "15",
        "MALEBIN_BATCH": "64",
        "MALEBIN_PATIENCE": "6",
        "MALEBIN_WORKERS": "0",
        "MALEBIN_AMP": "0",
        "MALEBIN_ABLATION_EPOCHS": "4",
        "MALEBIN_ABLATION_GROUPS": "attention",
        "MALEBIN_RUN_ABLATION": "1",
        "MALEBIN_RUN_FINAL": "1",
        "MALEBIN_RUN_CV": "0",
        "MALEBIN_FOLDS": "2",
    },
    # Third pass.  The 4-epoch attention ablation ranked the proposed stack
    # LAST (cbam+coord 0.3355) and plain SE first (0.4950), while attn=coord had
    # the best *validation* score of the six -- a spread that looks like early
    # training dynamics rather than a settled ordering.  Task 2b had already
    # shown this architecture needs ~10 epochs before it separates from noise.
    # So: 12 epochs per variant, patience 5, and test evaluation at each
    # variant's best checkpoint (train_model already restores best_state).
    # The 4-epoch table is kept as task3_ablation_4epoch.csv for the appendix.
    "ablation12": {
        "MALEBIN_IMG_SIZE": "64",
        "MALEBIN_CACHE_SIZE": "96",
        "MALEBIN_EPOCHS": "15",                 # final model, matches Task 2b
        "MALEBIN_BATCH": "64",
        "MALEBIN_PATIENCE": "5",
        "MALEBIN_WORKERS": "0",
        "MALEBIN_AMP": "0",
        "MALEBIN_ABLATION_EPOCHS": "12",
        "MALEBIN_ABLATION_GROUPS": "attention",
        "MALEBIN_RUN_ABLATION": "1",
        "MALEBIN_RUN_FINAL": "1",
        "MALEBIN_RUN_CV": "0",
        "MALEBIN_FOLDS": "2",
    },
    "full": {                                   # the README spec; GPU territory
        "MALEBIN_IMG_SIZE": "224",
        "MALEBIN_CACHE_SIZE": "256",
        "MALEBIN_EPOCHS": "25",
        "MALEBIN_FOLDS": "5",
    },
}

# Per-notebook overrides layered on top of the profile.  Epoch counts are not
# one number across the chain: the baselines are reference points and get the
# fewest, the proposed model and the ablation carry the actual claims.
PER_NOTEBOOK: dict[str, dict[str, dict[str, str]]] = {
    "cpu2h": {
        "task2a": {"MALEBIN_EPOCHS": "2"},      # 3 baselines x 2 ep ~ 19 min
        "task2b": {"MALEBIN_EPOCHS": "5"},      # the proposed model  ~ 12 min
        "task3a": {"MALEBIN_EPOCHS": "4"},      # final + 8 ablations + CV ~ 67 min
    },
    "fair": {
        "task2b": {"MALEBIN_EPOCHS": "15"},     # proper head-to-head  ~ 32 min
        "task3a": {"MALEBIN_EPOCHS": "6"},      # 6 ablations @4ep + final ~ 60 min
    },
}


# ---------------------------------------------------------------------------
# Output stripping
# ---------------------------------------------------------------------------
IMAGE_KEYS = ("image/png", "image/jpeg", "image/svg+xml", "image/gif")
PLOTLY_KEYS = ("application/vnd.plotly.v1+json",)


def strip_visual_outputs(nb) -> tuple[int, float]:
    """
    Remove image / Plotly payloads from an executed notebook in place.

    Text, tables (``text/plain`` and the small ``text/html`` DataFrame repr) and
    every printed number are kept -- only the heavy rendered visuals go, and
    each one leaves behind a line saying where the file actually is.
    """
    removed = 0
    saved_bytes = 0
    for cell in nb.cells:
        for out in cell.get("outputs", []) or []:
            data = out.get("data")
            if not data:
                continue
            hits = [k for k in list(data) if k in IMAGE_KEYS or k in PLOTLY_KEYS]
            # Plotly also ships a self-contained <script> blob in text/html that
            # carries the whole plotly.js bundle; a DataFrame's text/html repr is
            # small, so size is what separates them.
            for k in list(data):
                if k == "text/html" and len(str(data[k])) > 200_000:
                    hits.append(k)
            if not hits:
                continue
            for k in hits:
                saved_bytes += len(str(data[k]))
                del data[k]
                removed += 1
            data.setdefault(
                "text/plain",
                "<figure written to figures/ -- not embedded; "
                "see the [figure] line above for the filename>")
    return removed, saved_bytes / 1e6


def run_one(tag: str, src: Path, workdir: Path, outdir: Path,
            timeout: int, kernel: str) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(src, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name=kernel,
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=True,          # keep going; we report failures ourselves
    )
    t0 = time.time()
    print(f"\n=== {tag}: {src.name} ===", flush=True)
    try:
        client.execute()
        err = None
    except CellExecutionError as e:             # only if allow_errors is off
        err = str(e)
    dt = time.time() - t0

    # Which cells raised?
    failures = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                failures.append({
                    "cell": i,
                    "ename": out.get("ename"),
                    "evalue": (out.get("evalue") or "")[:400],
                })

    removed, mb = strip_visual_outputs(nb)
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / src.name
    nbformat.write(nb, str(dest))
    size_mb = dest.stat().st_size / 1e6

    status = "OK" if not failures else f"{len(failures)} ERROR CELL(S)"
    print(f"--- {tag}: {status} in {dt/60:.1f} min | "
          f"stripped {removed} visuals ({mb:.1f} MB) | notebook {size_mb:.2f} MB",
          flush=True)
    for f in failures:
        print(f"      cell {f['cell']}: {f['ename']}: {f['evalue'][:200]}",
              flush=True)

    return {"tag": tag, "notebook": src.name, "seconds": round(dt, 1),
            "failures": failures, "visuals_stripped": removed,
            "payload_mb_removed": round(mb, 2),
            "out_notebook_mb": round(size_mb, 2),
            "harness_error": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="cpu2h", choices=sorted(PROFILES))
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of tags, e.g. --only task1 task2a")
    ap.add_argument("--data-root", default=os.environ.get("MALEBIN_DATA_ROOT"))
    ap.add_argument("--out-dir", default=str(REPO),
                    help="where figures/ models/ artifacts/ are written")
    ap.add_argument("--workdir", default=str(REPO / "run"))
    ap.add_argument("--executed", default=str(REPO / "executed_notebooks"))
    ap.add_argument("--timeout", type=int, default=60 * 90)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--list-profiles", action="store_true")
    args = ap.parse_args()

    if args.list_profiles:
        print(json.dumps(PROFILES, indent=2))
        return 0

    if not args.data_root:
        print("ERROR: set --data-root or MALEBIN_DATA_ROOT", file=sys.stderr)
        return 2

    env = PROFILES[args.profile].copy()
    env["MALEBIN_DATA_ROOT"] = str(Path(args.data_root).resolve())
    env["MALEBIN_OUT_DIR"] = str(Path(args.out_dir).resolve())
    env["MALEBIN_NO_INLINE"] = "1"          # <- the whole point
    env["MALEBIN_FAST"] = "0"
    env.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
    os.environ.update(env)

    # The installed kernelspec launches a bare "python", which resolves through
    # PATH -- on Windows that can easily be the Store stub rather than the
    # interpreter running this script.  Put our own Scripts/ dir first so the
    # kernel is guaranteed to be the same environment (and the same torch).
    os.environ["PATH"] = (str(Path(sys.executable).parent) + os.pathsep
                          + os.environ.get("PATH", ""))

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # the notebooks materialise their own copy here via %%writefile
    shutil.copy2(HERE / "malebin_common.py", workdir / "malebin_common.py")

    todo = [(t, p) for t, p in NOTEBOOKS if args.only is None or t in args.only]
    print(f"profile   : {args.profile}")
    print(f"data root : {env['MALEBIN_DATA_ROOT']}")
    print(f"out dir   : {env['MALEBIN_OUT_DIR']}")
    print(f"workdir   : {workdir}")
    print(f"notebooks : {[t for t, _ in todo]}")
    print("settings  : " + ", ".join(f"{k.replace('MALEBIN_','').lower()}={v}"
                                     for k, v in sorted(PROFILES[args.profile].items())))

    results, t0 = [], time.time()
    per_nb = PER_NOTEBOOK.get(args.profile, {})
    for tag, src in todo:
        extra = per_nb.get(tag, {})
        for k, v in extra.items():              # per-notebook budget
            os.environ[k] = v
        if extra:
            print(f"    [{tag}] overrides: "
                  + ", ".join(f"{k}={v}" for k, v in extra.items()))
        results.append(run_one(tag, src, workdir, Path(args.executed),
                               args.timeout, args.kernel))
        for k in extra:                          # restore the profile default
            if k in env:
                os.environ[k] = env[k]

    total = time.time() - t0
    summary = {"profile": args.profile, "settings": PROFILES[args.profile],
               "total_minutes": round(total / 60, 1), "runs": results}
    Path(args.executed).mkdir(parents=True, exist_ok=True)
    (Path(args.executed) / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    n_fail = sum(len(r["failures"]) for r in results)
    print(f"\n{'=' * 70}\nTOTAL {total/60:.1f} min | "
          f"{len(results)} notebooks | {n_fail} error cells")
    for r in results:
        print(f"  {r['tag']:<7s} {r['seconds']/60:6.1f} min  "
              f"{r['out_notebook_mb']:6.2f} MB  "
              f"{'OK' if not r['failures'] else 'FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
