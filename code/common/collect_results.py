"""
collect_results.py -- turn the run's artifacts into report-ready tables.

Reads whatever the notebook chain has produced so far and writes RESULTS.md
plus a couple of tidied CSVs.  Safe to run mid-run: every section is optional
and simply reports "not produced yet" if its artifact is missing, so you can
call it as soon as Task 2 lands without waiting for Task 3.

    python code/common/collect_results.py --out-dir . --run-log run_real.log
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PREFIX = "Group12_MaleBin"

# Columns the report actually needs, in the order a marker reads them.
HEADLINE = [
    ("model", "Model"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced acc."),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "F1 (macro)"),
    ("f1_weighted", "F1 (weighted)"),
    ("mcc", "MCC"),
    ("cohen_kappa", "Cohen kappa"),
    ("roc_auc_macro_ovr", "ROC-AUC (OvR)"),
    ("params", "Params"),
    ("size_mb", "Size (MB)"),
    ("train_seconds", "Train (s)"),
    ("inference_ms_per_image", "Infer (ms/img)"),
]


def _fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in ("params",):
            out[c] = out[c].map(lambda v: f"{int(v):,}" if pd.notna(v) else "-")
        elif pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    return out


def md_table(df: pd.DataFrame) -> str:
    """Render a GitHub markdown table without depending on `tabulate`."""
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in rec]
            for rec in df.itertuples(index=False, name=None)]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |"
    rule = "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"
    body = ["| " + " | ".join(r[i].ljust(w[i]) for i in range(len(cols))) + " |"
            for r in rows]
    return "\n".join([head, rule, *body])


def section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--run-log", default=None)
    args = ap.parse_args()

    root = Path(args.out_dir).resolve()
    art = root / "artifacts"
    figs = root / "figures"

    def A(name: str) -> Path:
        return art / f"{PREFIX}_{name}"

    parts: list[str] = []
    parts.append(f"""# Results — MaleBin / ByteAttnNet

> **Budget-constrained run.** Produced on the real 12,464-image MaleBin dataset
> (39 families, duplicate-grouped family-stratified split, seed 42) but on a
> **CPU-only machine at 64 px with 2–5 training epochs**, not the 224 px /
> 25-epoch specification. Relative ordering under an identical budget is
> meaningful; **absolute values are floored by the training budget and must not
> be compared with published numbers.** See `REAL_RUN.md` for the full budget
> table and caveats.
""")

    # ---------------------------------------------------------------- split
    man = A("split_manifest.csv")
    if man.exists():
        m = pd.read_csv(man)
        counts = m.subset.value_counts()
        body = (f"- images: **{len(m):,}** · families: **{m.family.nunique()}** · "
                f"duplicate groups: **{m.dup_group.nunique():,}** "
                f"({100 * (1 - m.dup_group.nunique() / len(m)):.1f}% collapse)\n"
                f"- train **{counts.get('train', 0):,}** · "
                f"val **{counts.get('val', 0):,}** · "
                f"test **{counts.get('test', 0):,}**\n\n"
                "Split is grouped on duplicate-group id and stratified on family, "
                "so no image can appear in training with a near-identical twin in "
                "test.")
        parts.append(section("1 · Dataset and split", body))

    # ------------------------------------------------- headline comparison
    allm = A("task2_all_models_comparison.csv")
    base = A("task2_baseline_comparison.csv")
    src = allm if allm.exists() else (base if base.exists() else None)
    if src is not None:
        df = pd.read_csv(src)
        cols = [c for c, _ in HEADLINE if c in df.columns]
        t = _fmt(df[cols].sort_values("f1_macro", ascending=False))
        t = t.rename(columns=dict(HEADLINE))
        which = ("baselines **and** ByteAttnNet" if src is allm
                 else "baselines only (ByteAttnNet not finished yet)")
        body = (f"Held-out test set, {which}. Sorted by macro-F1, which is the "
                f"headline metric for this imbalanced 39-class problem.\n\n"
                + md_table(t))
        if src is allm and "model" in df.columns:
            prop = df[df.model.str.contains("ByteAttn", case=False, na=False)]
            bl = df[~df.model.str.contains("ByteAttn", case=False, na=False)]
            if len(prop) and len(bl):
                best_bl = bl.loc[bl.f1_macro.idxmax()]
                p = prop.iloc[0]
                d = p.f1_macro - best_bl.f1_macro
                body += (f"\n\n**ByteAttnNet vs the best baseline "
                         f"({best_bl.model}):** macro-F1 "
                         f"{p.f1_macro:.4f} vs {best_bl.f1_macro:.4f} "
                         f"(**{d:+.4f}**), with "
                         f"{int(p.params):,} vs {int(best_bl.params):,} "
                         f"parameters "
                         f"({best_bl.params / max(1, p.params):.1f}x smaller).")
        t.to_csv(root / "RESULTS_model_comparison.csv", index=False)
        parts.append(section("2 · Model comparison (the headline table)", body))
    else:
        parts.append(section("2 · Model comparison",
                             "_Not produced yet — Task 2 still running._"))

    # ------------------------------------------------------ per-class recall
    pc = A("task2_perclass_recall_all.csv")
    if pc.exists():
        d = pd.read_csv(pc)
        parts.append(section(
            "3 · Per-class performance",
            "Full per-class precision / recall / F1 per model:\n\n"
            + "\n".join(f"- `artifacts/{p.name}`"
                        for p in sorted(art.glob(f"{PREFIX}_task2_perclass_*.csv")))
            + f"\n\nCombined per-class recall table: {d.shape[0]} rows x "
              f"{d.shape[1]} columns (`artifacts/{pc.name}`)."))

    # ---------------------------------------------------- confusion matrices
    cms = sorted(figs.glob(f"{PREFIX}_task2_cm_*.png")) + \
        sorted(figs.glob(f"{PREFIX}_task2_proposed_cm.png")) + \
        sorted(figs.glob(f"{PREFIX}_task3_final_cm.png"))
    if cms:
        parts.append(section(
            "4 · Confusion matrices",
            "Row-normalised (cell = recall of that true family):\n\n"
            + "\n".join(f"- `figures/{p.name}`" for p in cms)))

    # ---------------------------------------------------------- ablation
    ab = A("task3_ablation_ranked.csv")
    if not ab.exists():
        ab = A("task3_ablation.csv")
    if ab.exists():
        d = pd.read_csv(ab)
        keep = [c for c in ["variant", "group", "test_macro_f1",
                            "delta_vs_proposed", "test_accuracy",
                            "test_macro_recall", "params", "train_seconds", "note"]
                if c in d.columns]
        d = d[keep].sort_values("test_macro_f1", ascending=False)
        parts.append(section(
            "5 · Ablation",
            "Each row is the same network with exactly one thing changed, all "
            "trained under the same reduced budget.\n\n" + md_table(_fmt(d))))

    # ---------------------------------------------------------------- CV
    cv = A("task3_cv_mean_std.csv")
    if cv.exists():
        parts.append(section(
            "6 · Cross-validation",
            "**Underpowered by budget** — folds were reduced, so treat this as "
            "the machinery working rather than as evidence.\n\n"
            + md_table(_fmt(pd.read_csv(cv)))))

    # ------------------------------------------------------- significance
    sig = A("task3_significance_summary.csv")
    if sig.exists():
        parts.append(section(
            "7 · Statistical significance",
            "The McNemar test on the shared held-out test set is the only test "
            "here with real statistical power.\n\n"
            + md_table(_fmt(pd.read_csv(sig)))))

    # ---------------------------------------------------------- figures
    if figs.exists():
        allf = sorted(figs.glob("*"))
        listing = "\n".join(
            f"- `figures/{p.name}` ({p.stat().st_size / 1024:.0f} KB)"
            for p in allf)
        parts.append(section(
            f"8 · Figures ({len(allf)} files)",
            "Every figure is a file on disk — none are embedded in the "
            "notebooks.\n\n" + listing))

    # ---------------------------------------------------------- run log
    if args.run_log and Path(args.run_log).exists():
        lines = [l.rstrip() for l in
                 Path(args.run_log).read_text(errors="replace").splitlines()
                 if l.startswith(("--- task", "TOTAL", "  task"))]
        if lines:
            parts.append(section("9 · Execution record",
                                 "```\n" + "\n".join(lines) + "\n```"))

    out = root / "RESULTS.md"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
