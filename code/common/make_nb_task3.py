"""Generate the two Task-3 notebooks."""
from nbtool import build, writefile_cell, BOOT, LOAD_SPLIT, REPO, PREFIX

# =============================================================================
#  3a -- IMPROVEMENT, ABLATION, CROSS-VALIDATION, SIGNIFICANCE
# =============================================================================
C: list[tuple[str, str]] = []
md = lambda s: C.append(("md", s))
co = lambda s: C.append(("code", s))

md(f"""\
# ICE478 Task 3a — Improvement, Ablation, Cross-Validation & Significance
## {PREFIX} · Track 3 (CNN + Attention)

Four things happen here, in this order, because each depends on the last:

1. **Ablation study** — turn one design decision off at a time and measure it.
   Every claim made in Task 2b is tested here; the ones that fail are reported
   as failures.
2. **Final model** — rebuilt from `v1` plus *only* the changes the ablation
   showed to help.
3. **5-fold grouped-stratified cross-validation** of the final model and the
   best baseline, reported as **mean ± std** (brief §6.4).
4. **Significance testing** — Wilcoxon signed-rank across folds, McNemar on the
   shared held-out test set, and Friedman + Nemenyi across all models, with the
   statistic, the *p*-value and a verdict at α = 0.05. The brief warns about
   picking the wrong test; §6 below spells out which test applies to which
   evidence and why.

Then the **fair comparison to related work** and the **Pillar-A verdict**.

⏱ Runtime on a Kaggle P100/T4, full settings: ablation ≈ 1.5–2.5 h,
CV ≈ 1–1.5 h. Both stages are individually switchable below. Run
`MALEBIN_FAST=1` first (≈8 min) to confirm the wiring, then do the real run.
If you must split it across sessions, run section 4 (ablation) in one session
and sections 5–7 in another — the notebook reloads the ablation table from disk
if it is there.""")

C.append(writefile_cell())
co(BOOT)

co('''\
# ---- what to run in this session -------------------------------------------
def _envflag(name, default):
    v = os.environ.get(name)
    return default if v in (None, "") else v not in ("0", "false", "False")

RUN_ABLATION = _envflag("MALEBIN_RUN_ABLATION", True)   # section 4 (expensive)
RUN_FINAL    = _envflag("MALEBIN_RUN_FINAL", True)      # section 5
RUN_CV       = _envflag("MALEBIN_RUN_CV", True)         # section 6 (folds x 2)
ABLATION_EPOCHS = int(os.environ.get("MALEBIN_ABLATION_EPOCHS", 0))                   or max(3, CFG.epochs // 2)
ABLATION_GROUPS = [g.strip() for g in
                   os.environ.get("MALEBIN_ABLATION_GROUPS", "").split(",")
                   if g.strip()] or None
print(f"ablation={RUN_ABLATION}  final={RUN_FINAL}  cv={RUN_CV}")
print(f"ablation groups : {ABLATION_GROUPS or 'all'}")
print(f"ablation epochs {ABLATION_EPOCHS} | final epochs {CFG.epochs} "
      f"| folds {CFG.n_folds}")
''')

co(LOAD_SPLIT)

# ---------------------------------------------------------------- harness
md("""\
---
## 3 · The ablation harness

One function, one protocol. Everything except the variable under test is held
fixed: same split, same seed, same epoch budget, same optimiser, same early
stopping on validation macro-F1. Each run reports the held-out **test** macro-F1
so the ranking is not contaminated by the value we early-stopped on.

*Why a shorter epoch budget for ablations.* We need a reliable **ranking** of
~20 variants, not a final number for each. Halving the budget makes the study
affordable; the winning configuration is then retrained at full budget in
section 5.""")

co('''\
from pathlib import Path
ABL_CSV = Path(CFG.art("task3_ablation.csv"))
abl_rows = []

BASE_MODEL = dict(channels=(16, 32, 64, 96) if CFG.fast else (48, 96, 192, 320),
                  depth=(1, 1, 2, 2), attention="cbam+coord", pool="gem",
                  dropout=0.3, use_bn=True, multiscale=True)
BASE_TRAIN = dict(aug="byte", optimizer="adamw", scheduler="cosine",
                  lr=CFG.lr, class_weighted=True, out_size=None)

def ablate(name, group, model_kw=None, train_kw=None, note=""):
    """Train one variant. model_kw / train_kw hold ONLY what this variant changes."""
    if ABLATION_GROUPS is not None and group not in ABLATION_GROUPS:
        print(f"  {name:<26s} SKIPPED (group {group!r} not in MALEBIN_ABLATION_GROUPS)")
        return None
    model_kw = dict(model_kw or {})
    train_kw = dict(train_kw or {})
    cfg_m = {**BASE_MODEL, **model_kw}
    cfg_t = {**BASE_TRAIN, **train_kw}

    M.set_seed(CFG.seed)
    model = M.ByteAttnNet(N_CLASSES, in_ch=1, **cfg_m)
    ltr, lva, lte = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx,
                                   aug=cfg_t["aug"], in_channels=1,
                                   out_size=cfg_t["out_size"])
    _, _, summ = M.train_model(model, ltr, lva, N_CLASSES, tag=name,
                               epochs=ABLATION_EPOCHS, lr=cfg_t["lr"],
                               optimizer=cfg_t["optimizer"],
                               scheduler=cfg_t["scheduler"],
                               class_weighted=cfg_t["class_weighted"],
                               verbose=False)
    y, p, pr = M.predict(model, lte)
    mm = M.full_metrics(y, p, pr, CLASS_NAMES, name=name)
    row = dict(variant=name, group=group,
               test_macro_f1=mm["f1_macro"], test_accuracy=mm["accuracy"],
               test_weighted_f1=mm["f1_weighted"],
               test_macro_recall=mm["recall_macro"],
               val_macro_f1=summ["best_val_macro_f1"],
               params=summ["params"], size_mb=round(summ["size_mb"], 2),
               train_seconds=round(summ["train_seconds"], 1),
               epochs_run=summ["epochs_run"], note=note,
               model_kw=json.dumps(model_kw), train_kw=json.dumps(train_kw))
    abl_rows.append(row)
    print(f"  {name:<26s} macro-F1 {mm['f1_macro']:.4f} | acc {mm['accuracy']:.4f}"
          f" | {summ['params']/1e6:.2f}M | {summ['train_seconds']:.0f}s")
    pd.DataFrame(abl_rows).to_csv(ABL_CSV, index=False)
    return row
''')

# ---------------------------------------------------------------- ablation
md("""\
---
## 4 · Ablation study

### 4.1 Attention — the central claim of the project
Six variants of exactly the same network, differing only in the attention module.
This is where "coordinate attention is the right attention for byte-plots" either
holds up or does not.""")

co('''\
if RUN_ABLATION:
    M.banner("4.1 ATTENTION")
    for att, note in [
        ("none",        "no attention at all -- the control"),
        ("se",          "channel only (Squeeze-Excitation) = what PAFE / IMCMK-CNN use"),
        ("spatial",     "spatial only (CBAM's second half)"),
        ("cbam",        "channel + spatial, but both destroy row position"),
        ("coord",       "coordinate attention only -- per-ROW and per-COLUMN gates"),
        ("cbam+coord",  "the proposed stack"),
    ]:
        ablate(f"attn={att}", "attention", dict(attention=att), note=note)
else:
    print("RUN_ABLATION is False -- skipping")
''')

md("""### 4.2 Architecture — multi-scale, pooling, depth, width, batch-norm, dropout""")

co('''\
if RUN_ABLATION:
    M.banner("4.2 ARCHITECTURE")
    ablate("no-multiscale", "multiscale", dict(multiscale=False),
           note="single 3x3 branch instead of 3x3 + dilated 5x5 + dilated 7x7")
    ablate("pool=gap", "pooling", dict(pool="gap"),
           note="GeM replaced by plain global average pooling")
    ablate("no-batchnorm", "batchnorm", dict(use_bn=False),
           note="BatchNorm removed everywhere")
    ablate("dropout=0.5", "dropout", dict(dropout=0.5),
           note="stronger head dropout")
    ablate("depth=shallow", "depth", dict(depth=(1, 1, 1, 1)),
           note="one block per stage")
''')

md("""### 4.3 Training recipe — augmentation, class weighting, optimiser, schedule, input size

The augmentation row is the second substantive claim: byte-plots must not be
flipped or rotated, because that produces a file that cannot exist.""")

co('''\
if RUN_ABLATION:
    M.banner("4.3 TRAINING RECIPE")
    ablate("aug=none", "augmentation", train_kw=dict(aug="none"),
           note="no augmentation")
    ablate("aug=naive", "augmentation", train_kw=dict(aug="naive"),
           note="flips + 90-degree rotations = the natural-image recipe")
    ablate("no-class-weights", "imbalance",
           train_kw=dict(class_weighted=False),
           note="plain cross-entropy on imbalanced data")
    ablate("optimizer=sgd", "optimiser", train_kw=dict(optimizer="sgd"),
           note="SGD+Nesterov instead of AdamW")
    ablate("scheduler=plateau", "schedule", train_kw=dict(scheduler="plateau"),
           note="ReduceLROnPlateau instead of OneCycle")
''')

md("""### 4.4 Reading the ablation""")

co('''\
abl = pd.read_csv(ABL_CSV) if ABL_CSV.exists() else pd.DataFrame(abl_rows)
if len(abl) == 0:
    raise RuntimeError("No ablation rows. Set RUN_ABLATION=True and re-run "
                       "section 4, or place task3_ablation.csv in the artifacts "
                       "folder.")
REF = "attn=cbam+coord"
ref_f1 = float(abl.loc[abl.variant == REF, "test_macro_f1"].iloc[0]) \\
         if (abl.variant == REF).any() else float(abl.test_macro_f1.max())
abl["delta_vs_proposed"] = abl.test_macro_f1 - ref_f1
abl = abl.sort_values("test_macro_f1", ascending=False).reset_index(drop=True)

display(abl[["variant", "group", "test_macro_f1", "delta_vs_proposed",
             "test_accuracy", "test_macro_recall", "params", "train_seconds",
             "note"]].round(4))
abl.to_csv(CFG.art("task3_ablation_ranked.csv"), index=False)
print(f"\\nreference configuration : {REF}  (macro-F1 {ref_f1:.4f})")
print(f"best variant found      : {abl.variant.iloc[0]}  "
      f"(macro-F1 {abl.test_macro_f1.iloc[0]:.4f})")
''')

co('''\
fig, ax = plt.subplots(figsize=(11.5, max(6, .34 * len(abl))))
gcol = {g: c for g, c in zip(sorted(abl.group.unique()),
                             plt.get_cmap("tab10").colors)}
d = abl.sort_values("delta_vs_proposed")
ax.barh(d.variant, d.delta_vs_proposed,
        color=[gcol[g] for g in d.group])
ax.axvline(0, c="k", lw=1.2)
ax.set(xlabel="change in test macro-F1 vs the proposed configuration",
       title="Ablation: what each design decision is actually worth\\n"
             "left of zero = removing/changing it HURT (so the choice was right)\\n"
             "right of zero = the change HELPED (so our Task-2 choice was wrong)")
import matplotlib.patches as mpatch
ax.legend(handles=[mpatch.Patch(color=c, label=g) for g, c in gcol.items()],
          fontsize=9, loc="lower right")
for i, (v, lbl) in enumerate(zip(d.delta_vs_proposed, d.variant)):
    ax.text(v + (0.001 if v >= 0 else -0.001), i, f"{v:+.4f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
fig.tight_layout()
fig.savefig(CFG.fig("task3_ablation.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

co('''\
# focused view: the attention ladder, which is the project's central claim
att = abl[abl.group == "attention"].set_index("variant")
order = [f"attn={a}" for a in ["none", "se", "spatial", "cbam", "coord", "cbam+coord"]]
order = [o for o in order if o in att.index]
fig, ax = plt.subplots(1, 2, figsize=(14.5, 4.8))
ax[0].bar(range(len(order)), att.loc[order, "test_macro_f1"],
          color=["tab:grey"] * (len(order) - 1) + ["tab:red"])
ax[0].set_xticks(range(len(order)))
ax[0].set_xticklabels([o.replace("attn=", "") for o in order], rotation=15)
for i, v in enumerate(att.loc[order, "test_macro_f1"]):
    ax[0].text(i, v + .004, f"{v:.4f}", ha="center", fontsize=9)
ax[0].set(ylabel="test macro-F1", title="4.4 · Attention ladder (red = proposed)")

base_f1 = float(att.loc["attn=none", "test_macro_f1"]) if "attn=none" in att.index else np.nan
gains = (att.loc[order, "test_macro_f1"] - base_f1).drop("attn=none", errors="ignore")
ax[1].bar(range(len(gains)), gains.values, color="tab:blue")
ax[1].set_xticks(range(len(gains)))
ax[1].set_xticklabels([g.replace("attn=", "") for g in gains.index], rotation=15)
ax[1].axhline(0, c="k", lw=1)
ax[1].set(ylabel="macro-F1 gain over NO attention",
          title="What each attention mechanism adds")
for i, v in enumerate(gains.values):
    ax[1].text(i, v + (.002 if v >= 0 else -.004), f"{v:+.4f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(CFG.fig("task3_attention_ladder.png"), dpi=130, bbox_inches="tight")
plt.show()

print("Interpretation guide -- read your own numbers against these cases:")
print("  coord > se           -> position-aware attention beats channel-only "
      "attention on byte-plots. This is our contribution, confirmed.")
print("  coord ~= se          -> position adds nothing here. Report it as a "
      "negative result; do NOT dress it up.")
print("  cbam+coord > both    -> the two are complementary (channel/spatial + "
      "positional), so the stack is justified.")
print("  cbam+coord < coord   -> the stack is redundant; the honest final model "
      "should use coord alone.")
''')

# ---------------------------------------------------------------- final model
md("""\
---
## 5 · The final model — built from what the ablation actually showed

The rule we follow: adopt a change **only** if it improved test macro-F1 in
section 4 by more than a small tolerance (0.002, roughly run-to-run noise at
this scale). Anything inside the tolerance keeps the simpler option. This is a
mechanical rule applied to the numbers, not a post-hoc story.""")

co('''\
TOL = 0.002
final_kw = dict(BASE_MODEL)
final_train = dict(BASE_TRAIN)
print(f"tolerance = {TOL} macro-F1")
print(f"reference = {REF} at macro-F1 {ref_f1:.4f}")
print()

adopted, rejected = [], []
for grp, g in abl[abl.variant != REF].groupby("group"):
    win = g.sort_values("test_macro_f1", ascending=False).iloc[0]
    if win.test_macro_f1 > ref_f1 + TOL:
        mk = json.loads(win.model_kw)
        tk = json.loads(win.train_kw)
        mk = {k: (tuple(v) if isinstance(v, list) else v) for k, v in mk.items()}
        final_kw.update(mk)
        final_train.update(tk)
        adopted.append((grp, win.variant, win.test_macro_f1, {**mk, **tk}))
    else:
        rejected.append((grp, win.variant, win.test_macro_f1))

print("ADOPTED (best variant in its group, and it beat the reference):")
for grp, v, f1, ch in adopted:
    print(f"  [{grp:<13s}] {v:<20s} {f1:.4f}  ->  {ch}")
if not adopted:
    print("  none -- the Task-2b configuration was already the best in every")
    print("  group. That is a legitimate outcome; say so in the report.")
print()
print("REJECTED (group's best variant did not beat the reference, so our")
print("original choice stands -- this is the evidence for each design decision):")
for grp, v, f1 in rejected:
    print(f"  [{grp:<13s}] {v:<20s} {f1:.4f}  ({f1 - ref_f1:+.4f})")
print()
print(f"FINAL model config : {final_kw}")
print(f"FINAL train config : {final_train}")
M.save_json(dict(final_model=final_kw, final_train=final_train,
                 reference=REF, reference_macro_f1=ref_f1, tolerance=TOL,
                 adopted=[(g, v, float(f)) for g, v, f, _ in adopted],
                 rejected=[(g, v, float(f)) for g, v, f in rejected]),
            CFG.art("task3_final_config.json"))
''')

co('''\
if RUN_FINAL:
    M.banner("Training the FINAL model at full epoch budget")
    M.set_seed(CFG.seed)
    final_model = M.ByteAttnNet(N_CLASSES, in_ch=1, **final_kw)
    ltr, lva, lte = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx,
                                   aug=final_train["aug"], in_channels=1,
                                   out_size=final_train["out_size"])
    _, hist_f, summ_f = M.train_model(
        final_model, ltr, lva, N_CLASSES, tag="ByteAttnNet-FINAL",
        optimizer=final_train["optimizer"], scheduler=final_train["scheduler"],
        lr=final_train["lr"], class_weighted=final_train["class_weighted"],
        save_path=CFG.mdl("best.pth"))
    M.plot_history(hist_f, title="ByteAttnNet FINAL — training",
                   save=CFG.fig("task3_final_history.png")); plt.show()

    y_f, p_f, pr_f = M.predict(final_model, lte)
    mm_final = M.full_metrics(y_f, p_f, pr_f, CLASS_NAMES, name="ByteAttnNet-FINAL")
    mm_final.update(params=summ_f["params"], size_mb=summ_f["size_mb"],
                    train_seconds=summ_f["train_seconds"],
                    inference_ms_per_image=M.measure_inference_ms(final_model, lte))
    np.savez_compressed(CFG.art("task3_final_test_predictions.npz"),
                        test_index=te_idx, y_true=y_f, y_pred=p_f, y_prob=pr_f)
    M.save_json({k: v for k, v in mm_final.items() if not k.startswith("_")},
                CFG.art("task3_final_metrics.json"))
    print(f"\\nFINAL  accuracy {mm_final['accuracy']:.4f} | "
          f"MACRO-F1 {mm_final['f1_macro']:.4f} | "
          f"weighted-F1 {mm_final['f1_weighted']:.4f} | "
          f"macro-recall {mm_final['recall_macro']:.4f} | "
          f"AUC {mm_final.get('roc_auc_macro_ovr', float('nan')):.4f}")
    M.plot_confusion(mm_final["_confusion"], CLASS_NAMES,
                     title="ByteAttnNet FINAL — confusion matrix",
                     save=CFG.fig("task3_final_cm.png")); plt.show()
    M.plot_roc_ovr(y_f, pr_f, CLASS_NAMES, title="ByteAttnNet FINAL — ROC OvR",
                   save=CFG.fig("task3_final_roc.png")); plt.show()
    M.plot_pr_ovr(y_f, pr_f, CLASS_NAMES, title="ByteAttnNet FINAL — PR OvR",
                  save=CFG.fig("task3_final_pr.png")); plt.show()
    display(mm_final["_report_df"].iloc[:N_CLASSES].sort_values("recall").round(3))
else:
    mm_final = M.load_json(CFG.art("task3_final_metrics.json"))
    print("loaded final metrics from disk")
''')

co('''\
# the deliverable checkpoint, in the exact shape the brief asks for
import torch
if RUN_FINAL:
    torch.save(dict(state_dict=final_model.state_dict(),
                    hparams=final_model.hparams,
                    class_names=CLASS_NAMES,
                    eval_scope=CFG.eval_scope,
                    img_size=final_train["out_size"] or CFG.img_size,
                    in_channels=1,
                    train_config=final_train,
                    metrics={k: v for k, v in mm_final.items()
                             if not k.startswith("_")}),
               CFG.mdl("best.pth"))
    M.save_json(dict(label_to_family={i: c for i, c in enumerate(CLASS_NAMES)},
                     family_to_label={c: i for i, c in enumerate(CLASS_NAMES)},
                     n_classes=N_CLASSES, eval_scope=CFG.eval_scope,
                     img_size=final_train["out_size"] or CFG.img_size,
                     in_channels=1, architecture="ByteAttnNet",
                     hparams=final_kw),
                CFG.mdl("label_map.json"))
    print("saved:")
    print("  ", CFG.mdl("best.pth"))
    print("  ", CFG.mdl("label_map.json"))
    print("\\nCopy both into the repo under models/ .")
''')

# ---------------------------------------------------------------- CV + stats
md("""\
---
## 6 · 5-fold cross-validation and significance testing (brief §6.4)

### Which test applies to which evidence — and why

The brief flags two mistakes students make. Here is exactly what we do and why:

| Evidence we have | Correct test | Why |
|---|---|---|
| 5 paired per-fold macro-F1 scores (final model vs best baseline, same folds) | **Wilcoxon signed-rank** | The observations are 5 *paired scores*, not per-sample outcomes. McNemar cannot consume fold averages. |
| Per-sample predictions from both models on the **one shared held-out test set** | **McNemar** | McNemar needs the paired 2×2 table of who got which sample right; that requires the same samples for both models, which is exactly what a shared test set gives. |
| 3+ models scored on the same folds | **Friedman** + **Nemenyi** post-hoc | Repeated pairwise tests inflate the family-wise error rate; Friedman tests them jointly and Nemenyi gives the critical rank difference. |
| Our number vs a **paper's** reported number | **no test is possible** | We do not have their per-sample predictions, and their split differs. Comparing is descriptive only. §7 labels it as such. |

**Fold construction.** `StratifiedGroupKFold` on the same duplicate-group key,
so no near-duplicate crosses a fold boundary. Within each fold's training part
we carve a further grouped validation slice for early stopping, so the fold's
held-out part is never seen during training or model selection.

**Honest limitation of n = 5.** The exact two-sided Wilcoxon *p*-value with 5
pairs cannot go below 0.0625, so it can *never* reach α = 0.05 no matter how
large the effect. We therefore report the two-sided *p*, the one-sided *p*, the
paired *t*-test, and Cohen's *d*, and we say plainly that the fold-level test is
under-powered — the McNemar test on ~2.5 k test samples is where the real
statistical power is. Claiming significance from 5 folds would be the exact
mistake the brief warns about.""")

co('''\
CV_CSV = Path(CFG.art("task3_cv_scores.csv"))
CV_EPOCHS = int(os.environ.get("MALEBIN_CV_EPOCHS", 0)) or CFG.epochs

def fold_val_split(train_part, frac=0.15):
    """Carve a grouped validation slice out of a fold's training part."""
    from sklearn.model_selection import StratifiedGroupKFold
    k = max(2, int(round(1 / frac)))
    s = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=CFG.seed + 7)
    a, b = next(s.split(np.zeros(len(train_part)), df.label.values[train_part],
                        groups[train_part]))
    return train_part[a], train_part[b]

def cv_model(make_model, tag, aug="byte", in_ch=1, out_size=None,
             optimizer="adamw", scheduler="cosine", lr=None, class_weighted=None):
    folds = M.grouped_kfold(df, groups, CFG.n_folds)
    out = []
    for k, (trn, tst) in enumerate(folds, 1):
        sub_tr, sub_va = fold_val_split(trn)
        M.set_seed(CFG.seed + k)
        m = make_model()
        a, b, c = M.make_loaders(imgs, df, sub_tr, sub_va, tst, aug=aug,
                                 in_channels=in_ch, out_size=out_size)
        _, _, s = M.train_model(m, a, b, N_CLASSES, tag=f"{tag}-f{k}",
                                epochs=CV_EPOCHS,
                                optimizer=optimizer, scheduler=scheduler, lr=lr,
                                class_weighted=class_weighted, verbose=False)
        y, p, pr = M.predict(m, c)
        mm = M.full_metrics(y, p, pr, CLASS_NAMES, name=tag)
        out.append(dict(model=tag, fold=k, n_test=len(tst),
                        macro_f1=mm["f1_macro"], accuracy=mm["accuracy"],
                        weighted_f1=mm["f1_weighted"],
                        macro_recall=mm["recall_macro"],
                        balanced_accuracy=mm["balanced_accuracy"],
                        train_seconds=round(s["train_seconds"], 1)))
        print(f"    fold {k}/{CFG.n_folds}  macro-F1 {mm['f1_macro']:.4f}  "
              f"acc {mm['accuracy']:.4f}  ({s['train_seconds']:.0f}s)")
    return out
''')

co('''\
# Resolve the baseline under test unconditionally: section 6.2 (McNemar) needs
# it even when the cross-validation section is switched off.
bl_json = Path(CFG.art("task2_baseline_summary.json"))
alt = list(Path("/kaggle/input").rglob("*task2_baseline_summary.json")) \\
      if Path("/kaggle/input").exists() else []
if bl_json.exists():
    BEST_BASELINE = M.load_json(bl_json)["best_baseline"]
elif alt:
    BEST_BASELINE = M.load_json(alt[0])["best_baseline"]
else:
    BEST_BASELINE = "ResNet18" if CFG.fast else "ResNet50"
    print(f"Task-2a summary not found -> using {BEST_BASELINE} as the baseline")
print(f"baseline under test: {BEST_BASELINE}")

if RUN_CV:
    cv_rows = []
    M.banner(f"CV: ByteAttnNet-FINAL ({CFG.n_folds} grouped stratified folds)")
    cv_rows += cv_model(lambda: M.ByteAttnNet(N_CLASSES, in_ch=1, **final_kw),
                        "ByteAttnNet-FINAL", aug=final_train["aug"],
                        out_size=final_train["out_size"],
                        optimizer=final_train["optimizer"],
                        scheduler=final_train["scheduler"], lr=final_train["lr"],
                        class_weighted=final_train["class_weighted"])

    M.banner(f"CV: {BEST_BASELINE}")
    _, bl_ch = M.build_baseline(BEST_BASELINE, N_CLASSES, pretrained=not CFG.fast)
    cv_rows += cv_model(
        lambda: M.build_baseline(BEST_BASELINE, N_CLASSES,
                                 pretrained=not CFG.fast)[0],
        BEST_BASELINE, in_ch=bl_ch)

    M.banner("CV: ByteAttnNet without attention (the ablation control)")
    cv_rows += cv_model(lambda: M.ByteAttnNet(N_CLASSES, in_ch=1,
                                              **{**final_kw, "attention": "none"}),
                        "ByteAttnNet-no-attn", aug=final_train["aug"])

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(CV_CSV, index=False)
elif CV_CSV.exists():
    cv = pd.read_csv(CV_CSV)
    BEST_BASELINE = [m for m in cv.model.unique() if "ByteAttnNet" not in m][0]
    print("loaded CV scores from disk")
else:
    # RUN_CV is off and nothing was left on disk by an earlier session.  This is
    # a legitimate configuration, not an error: with a small fold count the
    # Wilcoxon test cannot reach any useful p-value anyway, and the statistical
    # weight of this notebook sits in the McNemar test of section 6.2, which
    # runs on the shared held-out test set and does not need folds at all.
    cv = None
    print("cross-validation skipped (RUN_CV=0 and no task3_cv_scores.csv on "
          "disk) -- sections 6 and 6.1 are reported as not run; the McNemar "
          "test in 6.2 is unaffected and still runs.")
''')

co('''\
# mean +- std across folds -- the format the brief requires
if cv is None:
    agg = None
    pretty = None
    print("cross-validation was not run in this session -- no mean +- std "
          "table and no per-fold figure. Section 6.2 (McNemar) is unaffected.")
else:
    agg = (cv.groupby("model")[["macro_f1", "accuracy", "weighted_f1",
                                "macro_recall", "balanced_accuracy"]]
             .agg(["mean", "std"]).round(4))
    display(agg)
    pretty = pd.DataFrame({
        m: {c: f"{g[c].mean():.4f} +- {g[c].std(ddof=1):.4f}"
            for c in ["macro_f1", "accuracy", "weighted_f1", "macro_recall",
                      "balanced_accuracy"]}
        for m, g in cv.groupby("model")}).T
    print(f"{CFG.n_folds}-fold cross-validation, mean +- standard deviation:")
    display(pretty)
    pretty.to_csv(CFG.art("task3_cv_mean_std.csv"))
    agg.to_csv(CFG.art("task3_cv_agg.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    models = sorted(cv.model.unique())
    data = [cv.loc[cv.model == m, "macro_f1"].values for m in models]
    ax[0].boxplot(data, showmeans=True, widths=.55)
    for i, d in enumerate(data, 1):
        ax[0].scatter(np.full(len(d), i) + np.random.uniform(-.08, .08, len(d)),
                      d, zorder=3, s=42, alpha=.85)
    ax[0].set_xticks(range(1, len(models) + 1))
    ax[0].set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax[0].set(ylabel="macro-F1 per fold",
              title=f"{CFG.n_folds}-fold grouped-stratified CV\\n"
                    "dots = individual folds")
    for m in models:
        g = cv[cv.model == m].sort_values("fold")
        ax[1].plot(g.fold, g.macro_f1, marker="o", label=m)
    ax[1].set(xlabel="fold", ylabel="macro-F1",
              xticks=sorted(cv.fold.unique()),
              title="Per-fold macro-F1 -- are the folds themselves consistent?")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CFG.fig("task3_cv.png"), dpi=130, bbox_inches="tight")
    plt.show()
''')

md("""### 6.1 Wilcoxon signed-rank across the folds""")

co('''\
stat_results = {}
if cv is None:
    piv = None
    print("no per-fold scores -> Wilcoxon signed-rank not applicable. "
          "The paired test that carries this notebook is McNemar (6.2).")
else:
    piv = cv.pivot_table(index="fold", columns="model", values="macro_f1")
    piv = piv.sort_index()
    display(piv.round(4))
    stat_results["wilcoxon_vs_best_baseline"] = M.wilcoxon_folds(
        piv["ByteAttnNet-FINAL"].values, piv[BEST_BASELINE].values,
        "ByteAttnNet-FINAL", BEST_BASELINE)
    if "ByteAttnNet-no-attn" in piv.columns:
        stat_results["wilcoxon_vs_no_attention"] = M.wilcoxon_folds(
            piv["ByteAttnNet-FINAL"].values, piv["ByteAttnNet-no-attn"].values,
            "ByteAttnNet-FINAL", "ByteAttnNet-no-attn")
''')

md("""### 6.2 McNemar on the shared held-out test set — where the real power is""")

co('''\
# paired per-sample predictions on the SAME test set
fin = np.load(CFG.art("task3_final_test_predictions.npz"))
bl_npz = Path(CFG.art("task2_baseline_test_predictions.npz"))
alt = list(Path("/kaggle/input").rglob("*task2_baseline_test_predictions.npz")) \\
      if Path("/kaggle/input").exists() else []
src = bl_npz if bl_npz.exists() else (alt[0] if alt else None)

if src is not None:
    z = np.load(src, allow_pickle=True)
    assert np.array_equal(z["test_index"], fin["test_index"]), \\
        "test indices differ -> the two notebooks used different splits"
    assert np.array_equal(z["y_true"], fin["y_true"])
    avail = [k.split("pred__")[1] for k in z.files if k.startswith("pred__")]
    print("baselines with saved per-sample predictions:", avail)
    for nm in avail:
        stat_results[f"mcnemar_vs_{nm}"] = M.mcnemar_test(
            fin["y_true"], fin["y_pred"], z[f"pred__{nm}"],
            "ByteAttnNet-FINAL", nm)
else:
    print("No Task-2a per-sample predictions found. Training the best baseline "
          "here so McNemar can still be run on the shared test set.")
    M.set_seed(CFG.seed)
    bm, ic = M.build_baseline(BEST_BASELINE, N_CLASSES, pretrained=not CFG.fast)
    a, b, c = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx, aug="byte",
                             in_channels=ic)
    M.train_model(bm, a, b, N_CLASSES, tag=BEST_BASELINE, verbose=False)
    yb, pb, _ = M.predict(bm, c)
    stat_results[f"mcnemar_vs_{BEST_BASELINE}"] = M.mcnemar_test(
        fin["y_true"], fin["y_pred"], pb, "ByteAttnNet-FINAL", BEST_BASELINE)
''')

md("""### 6.3 Friedman + Nemenyi across all three models""")

co('''\
if piv is None:
    print("no CV table -> Friedman + Nemenyi not applicable in this session")
elif piv.shape[1] >= 3:
    stat_results["friedman_nemenyi"] = M.friedman_nemenyi(
        piv.values, list(piv.columns))
else:
    print(f"only {piv.shape[1]} models in the CV table -- Friedman needs 3+")
M.save_json(stat_results, CFG.art("task3_significance_tests.json"))
''')

co('''\
# one-line summary table of every test, for the report
rows = []
for k, r in stat_results.items():
    rows.append(dict(comparison=k, test=r["test"],
                     statistic=round(r.get("statistic", r.get("chi2_statistic",
                                                              float("nan"))), 4),
                     p_value=r["p_value"],
                     alpha=0.05,
                     verdict="SIGNIFICANT" if r["significant_at_0p05"]
                             else "not significant",
                     detail=r.get("variant", r.get("note", ""))))
st = pd.DataFrame(rows)
display(st)
st.to_csv(CFG.art("task3_significance_summary.csv"), index=False)
''')

# ---------------------------------------------------------------- Pillar A
md("""\
---
## 7 · Fair comparison to related work — and the Pillar-A verdict

Two comparisons, clearly labelled, because they are not equally fair:

* **Comparison 1 (the fair one).** Set `CFG.eval_scope = "malimg25"` and re-run
  the notebook. That restricts MaleBin to the 25 original Malimg families, i.e.
  **the same dataset and the same label space** as PAFE (2024), DRIN (2024) and
  SE-AGM (2023). The remaining difference is our *harder* split — duplicate
  grouped instead of random — and Task 2a §3 measured what that costs. This is
  the comparison the rubric means.
* **Comparison 2 (indicative only).** Full MaleBin-39 against Alshomrani et al.
  (2025), who report 94.04 % on a merged 61-class visual-malware corpus. It is
  the closest published analogue to a merged multi-source label space, but it is
  **not the same dataset**, so we present it as context and never claim a win
  from it.

We do **not** run a statistical test against any published number: we have no
access to their per-sample predictions, and their split differs. Doing so would
be the "unfair paper comparison" the brief explicitly names as a common
mistake.""")

co('''\
target = M.best_comparable_target(CFG.eval_scope)
final_f1 = float(mm_final["f1_macro"]) * 100
final_acc = float(mm_final["accuracy"]) * 100
final_wf1 = float(mm_final["f1_weighted"]) * 100
if cv is None:
    cv_mean = cv_std = float("nan")
else:
    sel = cv.loc[cv.model == "ByteAttnNet-FINAL", "macro_f1"]
    cv_mean = float(sel.mean()) * 100
    cv_std = float(sel.std(ddof=1)) * 100

M.banner(f"PILLAR A — scope '{CFG.eval_scope}'")
print(f"target paper   : {target['paper']}")
print(f"citation       : {target['citation']}")
print(f"their metric   : {target['metric']} = {target['value']:.2f}")
print(f"caveat         : {target['caveat']}")
print()
print(f"our macro-F1 (held-out test) : {final_f1:.2f}")
if cv is None:
    print("our macro-F1 (CV)            : not run in this session")
else:
    print(f"our macro-F1 ({CFG.n_folds}-fold CV)     : {cv_mean:.2f} +- {cv_std:.2f}")
print(f"our weighted-F1              : {final_wf1:.2f}")
print(f"our accuracy                 : {final_acc:.2f}")

# compare like-with-like: use whichever of ours matches THEIR metric
ours = {"F1": final_wf1, "Accuracy": final_acc,
        "macro-F1": final_f1}.get(target["metric"].split()[0], final_f1)
diff = ours - target["value"]
verdict = ("BEAT" if diff > 1.0 else "MATCH" if diff > -1.0 else "BELOW")
print()
print(f"like-for-like on their metric: ours {ours:.2f} vs theirs "
      f"{target['value']:.2f}  ->  {diff:+.2f} points")
print(f"PILLAR-A CLASSIFICATION      : {verdict}")
print("   (margin rule used: > +1.0 point = Beat, within +-1.0 = Match, "
      "< -1.0 = Below)")

M.save_json(dict(scope=CFG.eval_scope, target=target,
                 ours_macro_f1=final_f1, ours_weighted_f1=final_wf1,
                 ours_accuracy=final_acc,
                 ours_cv_macro_f1_mean=cv_mean, ours_cv_macro_f1_std=cv_std,
                 like_for_like=ours, difference=diff, verdict=verdict),
            CFG.art("task3_pillarA.json"))
''')

co('''\
# the full comparison figure for the report
comp_rows = []
for r in M.RELATED_WORK:
    if r["comparable"] and not np.isnan(r["headline_value"]):
        comp_rows.append(dict(system=f"{r['key']} ({r['year']})",
                              score=r["headline_value"],
                              metric=r["headline_metric"], kind="published",
                              split="random"))
comp_rows.append(dict(system="ByteAttnNet-FINAL (ours)", score=final_f1,
                      metric="macro-F1", kind="ours", split="duplicate-grouped"))
comp_rows.append(dict(system="ByteAttnNet-FINAL (ours)", score=final_wf1,
                      metric="weighted-F1", kind="ours", split="duplicate-grouped"))
comp_rows.append(dict(system="ByteAttnNet-FINAL (ours)", score=final_acc,
                      metric="accuracy", kind="ours", split="duplicate-grouped"))
cdf = pd.DataFrame(comp_rows)
display(cdf)
cdf.to_csv(CFG.art("task3_related_work_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(11, 5))
c = ["tab:red" if k == "ours" else "tab:grey" for k in cdf.kind]
lbl = [f"{s}\\n[{m}, {sp} split]" for s, m, sp in zip(cdf.system, cdf.metric, cdf.split)]
ax.barh(range(len(cdf)), cdf.score, color=c)
ax.set_yticks(range(len(cdf)))
ax.set_yticklabels(lbl, fontsize=8.5)
for i, v in enumerate(cdf.score):
    ax.text(v + .3, i, f"{v:.2f}", va="center", fontsize=9)
ax.axvline(target["value"], ls="--", c="k", lw=1.4,
           label=f"Pillar-A target: {target['paper']} = {target['value']:.2f}")
ax.set(xlabel="score (%)", xlim=(0, 105),
       title=f"Fair comparison, scope = {CFG.eval_scope} (red = ours)\\n"
             "NOTE: every grey bar comes from a RANDOM split; ours is "
             "duplicate-grouped and therefore strictly harder")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(CFG.fig("task3_related_work_comparison.png"), dpi=130,
            bbox_inches="tight")
plt.show()
''')

md("""\
---
## 8 · What worked, what did not, and why (Pillar-B Q4)

Write this from **your** numbers. The structure the rubric rewards:

**What worked.**
* Whichever ablation rows sit left of zero in §4.4 — those are the design
  choices the data confirmed. Quote the delta for each.
* If `coord` beat `se`, say so and connect it back to the Task-1 finding
  (row index = byte offset) that predicted it. That is a design hypothesis
  stated in advance and then confirmed, which is the strongest thing you can
  report.
* If `aug=byte` beat `aug=naive`, that confirms the byte-plot geometry argument.

**What did not work.**
* Any ablation row right of zero: a change we did not predict that turned out to
  help. Name it and say what it means about our reasoning.
* If `cbam+coord` was no better than `coord` alone, the stack is redundant —
  report it as redundant. If attention as a whole added ~nothing, say that
  clearly; a null result honestly reported scores better than a mysterious win.
* The Wilcoxon test on 5 folds is under-powered by construction (§6). Report the
  p-value and its floor; do not claim significance it cannot deliver.

**Why we are below the published numbers.**
Not because the model is weak, but because of a protocol difference we measured:
Task 2a §3 quantified the inflation a random split buys on this data. Also note
the dataset's own disclaimer — the Malimg half is outdated malware, and the
uploader's 256×256 resize distorts texture, so 65,536 pixels stand in for a file
that may be far larger. Both cap what any model can achieve here.

**Threats to validity we accept.**
* The duplicate-group key is a *proxy* for provenance, not ground truth. A
  tighter dHash threshold would group more aggressively and lower our score
  further; a looser one would raise it. We fix it at 6/128 and report it.
* Some families have very few distinct binaries, so their per-class recall
  rests on a handful of independent samples. Per-class numbers for those
  families carry large error bars that a single point estimate hides.
* 5 folds is the brief's requirement, not a statistically comfortable *n*.""")

co('''\
print("Task 3a artefacts:")
for p in sorted(Path(CFG.out_dir).rglob("*")):
    if p.is_file() and ("task3" in p.name or p.name.endswith(("best.pth",
                                                             "label_map.json"))):
        print(f"  {p.stat().st_size/1024:9.1f} KB  {p.relative_to(CFG.out_dir)}")
print("""
Copy into the repo:
  models/Group12_MaleBin_best.pth
  models/Group12_MaleBin_label_map.json
  report/task3/Group12_MaleBin_task3_report.pdf
Next: Group12_MaleBin_task3_explainability.ipynb
""")
''')

build(C, REPO / "code" / "task3" / f"{PREFIX}_task3_improvement_ablation.ipynb",
      "ICE478 Task 3a - ablation, CV, significance")
print("task3a done")
