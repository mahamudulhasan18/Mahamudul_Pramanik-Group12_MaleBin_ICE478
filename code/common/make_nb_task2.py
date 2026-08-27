"""Generate the two Task-2 notebooks."""
from nbtool import build, writefile_cell, BOOT, LOAD_SPLIT, REPO, PREFIX

# =============================================================================
#  2a -- BASELINES
# =============================================================================
C: list[tuple[str, str]] = []
md = lambda s: C.append(("md", s))
co = lambda s: C.append(("code", s))

md(f"""\
# CSE475 Task 2a — Baseline CNNs on MaleBin
## {PREFIX} · Track 3 (CNN + Attention)

**Goal.** Establish an honest, leakage-free performance floor with four standard
CNNs, using the full mandatory metric set (brief §6.1), so that Task 2b's
proposed model has something real to beat and Task 3 has a named
*baseline-to-beat* for the significance test.

### Baselines (from the brief's §5 Track-3 pool)
| Model | Why it is in the pool |
|---|---|
| **SimpleCNN** | 4-block VGG-style CNN trained *from scratch*, no attention, no pretraining. The genuine like-for-like control: it isolates the effect of our architecture choices rather than of ImageNet weights. |
| **ResNet50** | The default backbone in almost every malware-image paper (PAFE, IMCEC, Jayasudha et al. all report it). ImageNet-pretrained. |
| **DenseNet121** | Dense connectivity reuses low-level texture features, which is a plausible advantage on byte-plots. Reported as a top performer by Jayasudha et al. (2023). |
| **EfficientNet-B0** | Modern compound-scaled baseline at a comparable parameter budget to our proposed model, so "we won because we are bigger" is not available as an explanation. |

`MobileNetV3-Small` and `VGG16` are also wired up in `build_baseline()` if you
want a fifth or sixth.

### Preprocessing standards applied (brief §6.2)
1. **Split by source, not at random.** MaleBin has no subject column, so we
   derive one: exact + near-duplicate groups (SHA-1 of pixels ∪ 128-bit dHash
   with Hamming ≤ 6, union-find). All splits are `StratifiedGroupKFold` on that
   key. Section 3 below *measures* what this costs us.
2. **No ID-like features.** The model sees pixels only. File size, path, folder
   name and PNG compression ratio are never inputs — they are the image-domain
   equivalent of the packet-ID leak the brief warns about.
3. **Imbalance handled** by inverse-frequency class weights in the loss, and
   judged by **macro-F1 + per-class recall**, never raw accuracy.
4. **Identical protocol for every model**: same split, same input size, same
   optimiser family, same schedule, same early-stopping criterion
   (validation macro-F1), same augmentation. Anything else would make the
   comparison meaningless.
5. **Normalisation**: per-image standardisation (subtract that image's mean,
   divide by its std). Grayscale is repeated to 3 channels only for the
   ImageNet-pretrained backbones, which is the standard protocol in this
   literature.

⏱ **Runtime**: ~40–70 min on a Kaggle P100/T4 for all four baselines at 224 px.
Set `MALEBIN_FAST=1` first for a ~5-minute wiring check.""")

C.append(writefile_cell())
co(BOOT)

co('''\
# ---- what we train here ----------------------------------------------------
BASELINES = ["SimpleCNN", "ResNet50", "DenseNet121", "EfficientNet-B0"]
if CFG.fast:
    BASELINES = ["SimpleCNN", "ResNet18"]      # cheap stand-ins for the smoke run
print("baselines:", BASELINES)
print("epochs   :", CFG.epochs, "| img_size:", CFG.img_size,
      "| batch:", CFG.batch_size)
''')

co(LOAD_SPLIT)

# ---------------------------------------------------------------- leakage cost
md("""\
---
## 3 · What the honest split costs us — measured, not asserted

Before training anything we quantify the leakage effect with a real CNN, because
this single number explains why our headline scores will sit below the published
99%+ figures. Same model, same hyper-parameters, same number of epochs; the only
difference is whether the split respects duplicate groups.

This is the control the brief's §8 "leakage void" rule is really about: it lets a
reader see that our lower number is the *honest* one, not a weaker model.""")

co('''\
M.banner("Leakage control experiment: grouped split vs random split")
from sklearn.model_selection import StratifiedKFold

def quick_train_eval(train_i, val_i, test_i, tag, epochs=None):
    m = M.ByteAttnNet(N_CLASSES, attention="cbam+coord")
    ltr, lva, lte = M.make_loaders(imgs, df, train_i, val_i, test_i, aug="byte")
    _, _, s = M.train_model(m, ltr, lva, N_CLASSES, tag=tag,
                            epochs=epochs or max(3, CFG.epochs // 3), verbose=False)
    y, p, pr = M.predict(m, lte)
    mm = M.full_metrics(y, p, pr, CLASS_NAMES, name=tag)
    return mm, s

# (a) our protocol -- already computed above as tr_idx / va_idx / te_idx
mm_grp, s_grp = quick_train_eval(tr_idx, va_idx, te_idx, "grouped split (OURS)")

# (b) the protocol every related-work paper uses
skf = StratifiedKFold(5, shuffle=True, random_state=CFG.seed)
pool_r, test_r = next(skf.split(np.zeros(len(df)), df.label.values))
skf2 = StratifiedKFold(6, shuffle=True, random_state=CFG.seed + 1)
sub_tr, sub_va = next(skf2.split(np.zeros(len(pool_r)), df.label.values[pool_r]))
mm_rnd, s_rnd = quick_train_eval(pool_r[sub_tr], pool_r[sub_va], test_r,
                                "random split (LEAKY)")

leak = pd.DataFrame([
    dict(protocol="random stratified (what the papers do)",
         accuracy=mm_rnd["accuracy"], macro_f1=mm_rnd["f1_macro"],
         weighted_f1=mm_rnd["f1_weighted"]),
    dict(protocol="duplicate-grouped stratified (ours)",
         accuracy=mm_grp["accuracy"], macro_f1=mm_grp["f1_macro"],
         weighted_f1=mm_grp["f1_weighted"]),
])
leak["macro_f1_inflation"] = leak.macro_f1 - leak.macro_f1.iloc[-1]
display(leak.round(4))
leak.round(5).to_csv(CFG.art("task2_leakage_control.csv"), index=False)
f1_rnd, f1_grp = mm_rnd["f1_macro"], mm_grp["f1_macro"]
gap = f1_rnd - f1_grp
print()
print(f"Random splitting changes macro-F1 by {gap:+.4f} on the SAME model"
      f" ({f1_rnd:.4f} random vs {f1_grp:.4f} duplicate-grouped).")
if f1_grp > 0.20:
    print(f"   relative inflation: {100 * gap / f1_grp:+.1f}%")
else:
    print("   (relative % suppressed: the grouped score is too low for the "
          "ratio to mean anything -- this is what a short/FAST run looks like)")
if CFG.fast:
    print()
    print("!! CFG.fast is ON: this section is a WIRING CHECK, not a result."
          " Both models are barely trained, so even the SIGN of the gap is"
          " noise. Re-run with CFG.fast=False before quoting any of it.")
print()
print("Every number from here on uses the duplicate-grouped split.")
''')

# ---------------------------------------------------------------- baselines
md("""\
---
## 4 · Train the baselines

One shared loop, one shared protocol. `in_channels` is 1 for `SimpleCNN` and 3
for the ImageNet backbones (grayscale repeated), which is the only thing that
differs between them.""")

co('''\
results, histories, summaries, test_preds = [], {}, {}, {}

for name in BASELINES:
    M.banner(f"BASELINE: {name}")
    M.set_seed(CFG.seed)                       # identical init conditions
    model, in_ch = M.build_baseline(name, N_CLASSES, pretrained=not CFG.fast)
    print(f"  input channels {in_ch} | params {M.count_params(model):,} "
          f"| {M.model_size_mb(model):.1f} MB")

    ltr, lva, lte = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx,
                                   aug="byte", in_channels=in_ch)
    _, hist, summ = M.train_model(model, ltr, lva, N_CLASSES, tag=name,
                                  save_path=CFG.mdl(f"baseline_{name}.pth"))

    y, p, pr = M.predict(model, lte)
    mm = M.full_metrics(y, p, pr, CLASS_NAMES, name=name)
    mm.update(params=summ["params"], size_mb=summ["size_mb"],
              train_seconds=summ["train_seconds"], best_epoch=summ["best_epoch"],
              epochs_run=summ["epochs_run"],
              inference_ms_per_image=M.measure_inference_ms(model, lte),
              in_channels=in_ch, pretrained=not CFG.fast)
    results.append(mm)
    histories[name] = hist
    summaries[name] = summ
    test_preds[name] = dict(y_true=y, y_pred=p, y_prob=pr)

    print(f"\\n  >> {name}: accuracy {mm['accuracy']:.4f} | "
          f"MACRO-F1 {mm['f1_macro']:.4f} | weighted-F1 {mm['f1_weighted']:.4f} | "
          f"macro-recall {mm['recall_macro']:.4f} | "
          f"AUC {mm.get('roc_auc_macro_ovr', float('nan')):.4f}")
    hist.to_csv(CFG.art(f"task2_history_{name}.csv"), index=False)
''')

co('''\
# persist per-sample test predictions -- required for the McNemar test in Task 3
np.savez_compressed(
    CFG.art("task2_baseline_test_predictions.npz"),
    test_index=te_idx, y_true=test_preds[BASELINES[0]]["y_true"],
    **{f"pred__{k}": v["y_pred"] for k, v in test_preds.items()},
    **{f"prob__{k}": v["y_prob"] for k, v in test_preds.items()})
print("saved ->", CFG.art("task2_baseline_test_predictions.npz"))
''')

# ---------------------------------------------------------------- comparison
md("""\
---
## 5 · Comparison table — the full mandatory metric set (§6.1)""")

co('''\
tbl = M.metrics_frame(results, sort_by="f1_macro")
show = ["model", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "roc_auc_macro_ovr",
        "avg_precision_macro", "mcc", "params", "size_mb", "train_seconds",
        "inference_ms_per_image", "best_epoch"]
display(tbl[[c for c in show if c in tbl.columns]].round(4))
tbl.to_csv(CFG.art("task2_baseline_comparison.csv"), index=False)

BEST_BASELINE = tbl.model.iloc[0]
BEST_BASELINE_F1 = float(tbl.f1_macro.iloc[0])
M.save_json(dict(best_baseline=BEST_BASELINE, best_baseline_macro_f1=BEST_BASELINE_F1,
                 table=tbl.to_dict(orient="records"), baselines=BASELINES,
                 eval_scope=CFG.eval_scope, n_classes=N_CLASSES,
                 n_test=int(len(te_idx))),
            CFG.art("task2_baseline_summary.json"))
print(f"\\n*** BASELINE TO BEAT: {BEST_BASELINE} "
      f"with macro-F1 = {BEST_BASELINE_F1:.4f} ***")
print("This is the model Task 3's Wilcoxon / McNemar tests are run against.")
''')

co('''\
# accuracy vs macro-F1: the gap is the imbalance story
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
x = np.arange(len(tbl))
ax[0].bar(x - .2, tbl.accuracy, .4, label="accuracy")
ax[0].bar(x + .2, tbl.f1_macro, .4, label="macro-F1")
for i, (a, f) in enumerate(zip(tbl.accuracy, tbl.f1_macro)):
    ax[0].text(i, max(a, f) + .01, f"{a-f:+.3f}", ha="center", fontsize=8)
ax[0].set_xticks(x); ax[0].set_xticklabels(tbl.model, rotation=20, ha="right")
ax[0].set(ylabel="score", ylim=(0, 1.08),
          title="Accuracy vs macro-F1\\n(label = accuracy - macro-F1, i.e. the "
                "imbalance premium)")
ax[0].legend()

ax[1].scatter(tbl.params / 1e6, tbl.f1_macro, s=90, c="tab:red")
for _, r in tbl.iterrows():
    ax[1].annotate(r.model, (r.params / 1e6, r.f1_macro), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
ax[1].set(xscale="log", xlabel="trainable parameters (millions, log)",
          ylabel="macro-F1", title="Cost vs benefit — is bigger better?")

ax[2].scatter(tbl.train_seconds / 60, tbl.f1_macro, s=90, c="tab:green")
for _, r in tbl.iterrows():
    ax[2].annotate(r.model, (r.train_seconds / 60, r.f1_macro), fontsize=8,
                   xytext=(4, 4), textcoords="offset points")
ax[2].set(xlabel="training time (minutes)", ylabel="macro-F1",
          title="Training cost vs benefit")
fig.tight_layout()
fig.savefig(CFG.fig("task2_baseline_comparison.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

co('''\
# training curves, all baselines together
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
for name, h in histories.items():
    ax[0].plot(h.epoch, h.train_loss, marker="o", ms=3, label=name)
    ax[1].plot(h.epoch, h.val_macro_f1, marker="s", ms=3, label=name)
ax[0].set(xlabel="epoch", ylabel="training cross-entropy", title="Training loss")
ax[1].set(xlabel="epoch", ylabel="validation macro-F1",
          title="Validation macro-F1 (the model-selection metric)")
for a in ax:
    a.legend(fontsize=9)
fig.tight_layout()
fig.savefig(CFG.fig("task2_baseline_curves.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

md("""\
---
## 6 · Per-model diagnostics: confusion matrix, ROC, PR, per-class report""")

co('''\
for mm in results:
    name = mm["model"]
    M.banner(f"{name}")
    M.plot_confusion(mm["_confusion"], CLASS_NAMES,
                     title=f"{name} — confusion matrix on the held-out test set",
                     save=CFG.fig(f"task2_cm_{name}.png"))
    plt.show()

    f, auc = M.plot_roc_ovr(test_preds[name]["y_true"],
                            test_preds[name]["y_prob"],
                            CLASS_NAMES, title=f"{name} — ROC one-vs-rest",
                            save=CFG.fig(f"task2_roc_{name}.png"))
    plt.show()
    f2, ap = M.plot_pr_ovr(test_preds[name]["y_true"], test_preds[name]["y_prob"],
                           CLASS_NAMES, title=f"{name} — precision-recall one-vs-rest",
                           save=CFG.fig(f"task2_pr_{name}.png"))
    plt.show()

    rep = mm["_report_df"].iloc[:N_CLASSES].copy()
    rep["support"] = rep["support"].astype(int)
    print(f"\\nper-class report — {name}   (sorted by recall, worst first)")
    display(rep.sort_values("recall").round(3))
    rep.round(4).to_csv(CFG.art(f"task2_perclass_{name}.csv"))
    worst = rep.sort_values("recall").head(5)
    print(f"  five hardest families for {name}: "
          + ", ".join(f"{i} (recall {r.recall:.2f})" for i, r in worst.iterrows()))
''')

co('''\
# which families does EVERY baseline get wrong? that is the real target for Task 3
rec = pd.DataFrame({mm["model"]: mm["_report_df"].iloc[:N_CLASSES]["recall"]
                    for mm in results})
rec["mean_recall"] = rec.mean(axis=1)
rec["support"] = results[0]["_report_df"].iloc[:N_CLASSES]["support"].astype(int)
hard = rec.sort_values("mean_recall")
display(hard.round(3))
hard.round(4).to_csv(CFG.art("task2_perclass_recall_all.csv"))

fig, ax = plt.subplots(figsize=(13, 5))
w = .8 / len(results)
for i, mm in enumerate(results):
    ax.bar(np.arange(len(hard)) + i * w, hard[mm["model"]], w, label=mm["model"])
ax.set_xticks(np.arange(len(hard)) + .4)
ax.set_xticklabels(hard.index, rotation=90, fontsize=7.5)
ax.axhline(.5, ls="--", c="r", lw=1, label="50% recall")
ax.set(ylabel="per-class recall on the test set",
       title="Per-class recall by baseline, hardest families first.\\n"
             "Families low for ALL baselines are what the proposed model must fix.")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(CFG.fig("task2_perclass_recall.png"), dpi=130, bbox_inches="tight")
plt.show()

print("families where every baseline stays under 0.60 recall:")
allbad = hard[(hard[[mm["model"] for mm in results]] < .60).all(axis=1)]
print("  " + (", ".join(allbad.index) if len(allbad) else "none"))
''')

md("""\
---
## 7 · Task 2a findings

* **Baseline to beat** — printed in section 5. Note *which* model won: if
  `SimpleCNN` (from scratch, 1-channel) is competitive with a 23 M-parameter
  ImageNet ResNet50, that is direct evidence that the natural-image prior does
  not transfer well to byte-plots — which is the core premise of the model we
  build in Task 2b, and it is now measured rather than assumed.
* **Accuracy overstates every model** by the gap plotted in section 5. That gap
  is the imbalance premium, and it is why macro-F1 is our headline.
* **The failures are concentrated**, not spread out: a handful of families that
  every baseline confuses (typically variant pairs inside one lineage). Task 3's
  improvements are judged on *those* families' recall, not on the average.
* **Bigger is not better here.** The parameter/macro-F1 scatter shows no
  monotone relationship, which is the opening our ~2.5 M-parameter proposed
  model exploits.
* **The honest split costs real points** (section 3). We keep that number
  visible in every report so the comparison to published work is transparent.""")

co('''\
print("Task 2a artefacts:")
for p in sorted(__import__("pathlib").Path(CFG.out_dir).rglob("*task2*")):
    print(f"  {p.stat().st_size/1024:9.1f} KB  {p.name}")
print("\\nNext: Group00_MaleBin_task2_proposed_model.ipynb")
''')

build(C, REPO / "code" / "task2" / f"{PREFIX}_task2_baselines.ipynb",
      "CSE475 Task 2a - MaleBin baselines")


# =============================================================================
#  2b -- PROPOSED MODEL
# =============================================================================
C = []
md = lambda s: C.append(("md", s))
co = lambda s: C.append(("code", s))

md(f"""\
# CSE475 Task 2b — Proposed Model: **ByteAttnNet**
## {PREFIX} · Track 3 (CNN + Attention)

A from-scratch CNN with a two-part attention stack, designed around one
observation from the Task-1 EDA:

> In a byte-plot, **the row index is the byte offset in the file.** Row *r* of a
> 256-px-wide image holds bytes `[256r, 256r+256)`. Vertical position is
> therefore *semantically meaningful* — PE header near the top, `.text` below it,
> `.data`/`.rsrc` lower, zero padding at the end — while horizontal position is
> an artefact of the chosen row width.

Every design choice below follows from that, and every one of them is a
switchable flag so the Task-3 ablation can *measure* it instead of us asserting
it. This notebook answers the four Pillar-B questions (brief §6.5, §8) in
sections 2–5.

⏱ ~15–25 min on a Kaggle P100/T4. `MALEBIN_FAST=1` for a wiring check.""")

C.append(writefile_cell())
co(BOOT)
co(LOAD_SPLIT)

# ------------------------------------------------------- architecture diagram
md("""\
---
## 1 · Architecture diagram

`input → stem → 4 × [multi-scale conv block + CBAM + CoordAtt] → GeM → dropout → linear`""")

co('''\
import matplotlib.patches as mp

def draw_architecture(save=None):
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=8.4, ec="black"):
        ax.add_patch(mp.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                                       fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, txt="", ls="-"):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333", ls=ls))
        if txt:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.0, txt, ha="center",
                    fontsize=7.4, color="#444")

    C_IN, C_STEM = "#dbeafe", "#bfdbfe"
    C_MS, C_ATT, C_POOL, C_HEAD = "#fde68a", "#fca5a5", "#a7f3d0", "#c7d2fe"

    box(1.5, 24, 10, 8, "INPUT\\nbyte-plot\\n1 x 224 x 224\\nper-image\\nstandardised", C_IN, 8)
    arrow(11.5, 28, 15, 28)
    box(15, 24, 10, 8, "STEM\\nConv 3x3 s2\\n-> BN -> ReLU\\n48 x 112 x 112", C_STEM, 8)

    xs = [28.5, 46, 63.5, 81]
    dims = ["48->48\\n112x112", "48->96\\n56x56", "96->192\\n28x28", "192->320\\n14x14"]
    deps = ["x1 block", "x1 block", "x2 blocks", "x2 blocks"]
    for i, (x, d, dp) in enumerate(zip(xs, dims, deps)):
        box(x, 38, 15, 11,
            f"STAGE {i+1}  ({dp})\\nMULTI-SCALE CONV\\n"
            "3x3  |  3x3 d=2  |  3x3 d=3\\nconcat -> 1x1 fuse -> +skip\\n" + d,
            C_MS, 7.7)
        box(x, 25.5, 15, 10.5,
            "ATTENTION\\nCBAM: channel(avg+max)\\n  then spatial 7x7\\n"
            "CoordAtt: per-ROW gate\\n  x per-COLUMN gate", C_ATT, 7.7)
        arrow(x + 7.5, 38, x + 7.5, 36.2)
        if i < 3:
            arrow(x + 15, 30.7, x + 17, 30.7, "MaxPool 2x2")
        if i == 0:
            arrow(25, 29, 28.5, 40)

    arrow(96, 30.7, 96, 22)
    box(74, 12, 22, 8.5,
        "GeM POOL   (mean(x^p))^(1/p)\\np learnable, clamped to [1, 8]\\n"
        "320 x 14 x 14  ->  320", C_POOL, 8.2)
    arrow(74, 16.2, 66, 16.2)
    box(44, 12, 21, 8.5,
        "HEAD\\nDropout(0.3)\\nLinear(320 -> n_classes)", C_HEAD, 8.4)
    arrow(44, 16.2, 36, 16.2)
    box(16, 12, 27, 8.5, "OUTPUT\\nlogits over 39 malware families\\n"
                         "softmax -> family probability", C_IN, 8.4)

    ax.text(50, 53.6, "ByteAttnNet — CNN + Attention for malware byte-plots",
            ha="center", fontsize=15, weight="bold")
    ax.text(50, 51.0,
            "yellow = multi-scale texture   |   red = attention   |   "
            "green = pooling   |   blue = I/O + head",
            ha="center", fontsize=9, color="#444")
    ax.text(50, 6.2,
            "Why attention here: CBAM answers 'which texture channels and which "
            "regions matter'.  CoordAtt answers 'which BYTE OFFSETS matter' -- it "
            "pools one axis at a time,\\nso a per-row gate survives, and a row IS a "
            "byte-offset band. Channel-only attention (SE, as in PAFE/IMCMK-CNN) "
            "pools that position away.",
            ha="center", fontsize=8.6, color="#111",
            bbox=dict(fc="#fff7ed", ec="#fdba74"))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return fig

draw_architecture(CFG.fig("task2_architecture.png"))
plt.show()
''')

# ------------------------------------------------------- Pillar B answers
md("""\
---
## 2 · Pillar-B Q1 — How the model works, step by step, in our own words

**Input.** A 256×256 grayscale byte-plot, bilinearly resized to 224×224 and
standardised *per image* (subtract that image's own mean, divide by its own std).
Per-image rather than dataset-wide statistics, because "how bright is this file
overall" is the average byte value, which is a nuisance factor: it changes when a
file is padded or packed, without changing which family it is. What survives
standardisation is the *texture*, and texture is the signal.

**Stem.** One 3×3 convolution with stride 2 → BatchNorm → ReLU, producing 48
channels at 112×112. Stride 2 immediately, because byte-plots have no fine
detail worth preserving at full resolution — the uploader already resized them,
so single-pixel structure is partly a resampling artefact. This halves the cost
of every later layer.

**Four stages.** Each stage is `multi-scale conv block(s) → attention → pool`.

*The multi-scale conv block* runs three convolutions on the same input in
parallel and concatenates them:

| branch | kernel | dilation | effective view | what it sees |
|---|---|---|---|---|
| 1 | 3×3 | 1 | 3×3 | fine byte texture — opcode-level n-grams |
| 2 | 3×3 | 2 | 5×5 | basic-block / function-level repetition |
| 3 | 3×3 | 3 | 7×7 | section-level layout |

The three are concatenated and fused by a 1×1 convolution, then added to a
shortcut (identity, or a 1×1 projection when the channel count changes) and
passed through ReLU. Using *dilation* instead of literally larger kernels is
what keeps the cost at 3×3 level: a real 7×7 convolution has 5.4× the parameters
of a 3×3 one, and PAFE/IMCMK-CNN both had to design fusion tricks to afford
theirs.

*The attention block* is CBAM followed by coordinate attention:

1. **CBAM channel gate.** Squeeze each feature map to one number twice — once by
   average pooling, once by max pooling — push both through a shared bottleneck
   MLP, add, sigmoid. Result: one weight per channel, i.e. *which texture
   detectors does this file need?* Average and max are both used because average
   describes how widespread a texture is and max describes whether it occurs at
   all anywhere; for a small packed section only the max fires.
2. **CBAM spatial gate.** Collapse channels to two maps (per-pixel max and
   per-pixel mean), concatenate, run one 7×7 convolution, sigmoid. Result: one
   weight per location, i.e. *which regions matter?*
3. **Coordinate attention.** Pool along **one axis at a time**: average over
   columns → a vector of length *H* (one value per row); average over rows → a
   vector of length *W*. Concatenate, one shared 1×1 conv + BN + hardswish, split
   back, and produce a per-row gate `a_h` and a per-column gate `a_w`; multiply
   the feature map by both. Because pooling never collapses both axes at once,
   *positional information survives* — and a row is a byte-offset band.

*Pooling between stages* is 2×2 max pooling after stages 1–3, so the spatial map
goes 112 → 56 → 28 → 14 while channels go 48 → 96 → 192 → 320.

**Readout.** GeM pooling: raise the feature map to the power *p*, average, take
the *p*-th root, with *p* a learnable scalar clamped to [1, 8]. *p* = 1 is
average pooling, *p* → ∞ is max pooling, so the network chooses where to sit
between "average texture energy over the whole file" and "the single most
distinctive block", instead of us guessing.

**Head.** Dropout(0.3) → one linear layer to 39 logits. No hidden FC layer:
with ~2.5 M convolutional parameters already, a wide FC head is where a
from-scratch model on ~10 k images starts memorising.

**Loss and selection.** Cross-entropy with inverse-frequency class weights and
label smoothing 0.05; AdamW; OneCycle LR; early stopping on **validation
macro-F1**. Selection on macro-F1 rather than accuracy or loss is deliberate —
it is the metric §6.2 says we are judged on, so it is the metric we optimise.""")

md("""\
---
## 3 · Pillar-B Q2 — Why each design choice is there

| Choice | Justification (traced to a Task-1 finding or a paper) |
|---|---|
| **Trained from scratch, 1 input channel** | Task-1 §C: byte-plots contain no oriented edges, no objects, no colour opponency. An ImageNet prior is not just useless but actively wrong, and fine-tuning must unlearn it. Replicating grayscale to 3 channels to satisfy a pretrained stem also triples the stem's FLOPs for zero information. |
| **Multi-scale dilated conv** | Task-1 §C/§E: signal exists at byte level (`entropy`, `grad_mag`), block level (`row_autocorr`), and section level (`row_var`, `top_band_mean`). One kernel size must compromise. PAFE (Heliyon 2024) and IMCMK-CNN (Alex. Eng. J. 2024) both report gains from multi-scale kernels; dilation gets the same receptive fields at 3×3 cost. |
| **CBAM channel attention** | Different families are identified by different textures. A per-channel gate lets one network specialise per family without extra capacity. This is the mechanism PAFE's FFSE and IMCMK-CNN's improved-SE blocks use, so we keep it. |
| **CBAM spatial attention** | Task-1 §C: the informative region is a *band*, and which band it is varies by family; the rest of the image is padding. A spatial gate suppresses padding. |
| **Coordinate attention — the novel part** | This is the gap identified in Task 1. Row index = byte offset. SE and CBAM's channel branch both global-pool, destroying that. Swin (Alshomrani et al. 2025) keeps 2-D position but treats it as a generic image coordinate and costs far more. CoordAtt (Hou et al., CVPR 2021) factorises attention into a per-row and per-column gate for the price of two 1×1 convolutions, so the network can literally learn "for family *X*, look 15–25 % into the file". |
| **No flips, no rotations** | Task-1 §C: flipping reverses byte order; rotating transposes offsets into row-width. Both produce inputs that cannot exist. Task 3 ablates this against the naive recipe. |
| **Byte-aware augmentation instead** | What *does* vary between variants of one binary: where a section starts (→ vertical roll ±12 %), total file length (→ row crop 85–100 %), a region being repacked (→ random erasing). Each transform corresponds to a real polymorphic operation. |
| **GeM pooling** | Byte-plot classes differ in texture *energy*; whether average or peak energy matters is an empirical question, so we make it a learnable parameter. |
| **Inverse-frequency class weights** | Task-1 §B imbalance + §6.2's macro-F1 rule. Basak et al. (2024) explicitly report their model "struggles with underrepresented classes"; weighting is the cheapest direct fix. |
| **~2.5 M parameters** | Effective *n* is the number of *distinct binaries* (Task-1 §F), which is far below the image count. A 23 M-parameter ResNet50 on that little data is asking for memorisation. |""")

md("""\
---
## 4 · Pillar-B Q3 — Why this should beat the baselines

Four concrete, falsifiable claims. Task 3 tests each one.

1. **The pretrained baselines carry the wrong prior.** ResNet50/DenseNet121/
   EfficientNet-B0 start from ImageNet filters — Gabor-like edge detectors and
   colour blobs. A byte-plot has neither. They must spend capacity unlearning,
   with 10–25 M parameters on an effective sample size of a few thousand
   distinct binaries. *Test:* if `SimpleCNN` (from scratch) is already
   competitive in Task 2a, the prior is confirmed unhelpful.
2. **Nobody else models byte offset, and byte offset is real information.**
   PE section layout is family-characteristic. CoordAtt is the only module in the
   comparison that can represent "attend to this offset band". *Test:* the
   ablation `attention = coord` vs `se` vs `none` isolates exactly this, and the
   Task-3 row-band occlusion test checks whether the learned row gates
   correspond to genuinely causal offsets.
3. **Multi-scale beats single-scale on texture.** *Test:* ablation
   `multiscale = False` at matched parameter count.
4. **Byte-aware augmentation beats the natural-image recipe.** *Test:* ablation
   `aug = 'byte'` vs `'naive'` vs `'none'`.

**Where we expect to lose, stated in advance.** Against the *published* Malimg
numbers (99.2–99.4 %) we expect to come in lower, because every one of those
papers uses a random split and we do not — Task 2a §3 measured that gap
directly. Our claim is only that we beat *our own* baselines under an identical
honest protocol, and that is what the significance tests in Task 3 examine.""")

# ------------------------------------------------------- train
md("""\
---
## 5 · Train ByteAttnNet (first version)""")

co('''\
M.set_seed(CFG.seed)
PROPOSED_CFG = dict(channels=(48, 96, 192, 320), depth=(1, 1, 2, 2),
                    attention="cbam+coord", pool="gem", dropout=0.3,
                    use_bn=True, multiscale=True, stem_stride=2)
if CFG.fast:
    PROPOSED_CFG["channels"] = (16, 32, 64, 96)

model = M.ByteAttnNet(N_CLASSES, in_ch=1, **PROPOSED_CFG)
print(model.hparams)
print(f"\\ntrainable parameters : {M.count_params(model):,}")
print(f"state-dict size      : {M.model_size_mb(model):.2f} MB")

# layer-by-layer shape trace -- the diagram, verified
import torch
x = torch.randn(1, 1, CFG.img_size, CFG.img_size)
with torch.no_grad():
    h = model.stem(x); print(f"\\nstem   -> {tuple(h.shape)}")
    for i, s in enumerate(model.stages):
        h = s(h); print(f"stage{i+1} -> {tuple(h.shape)}")
    v = model.pool(h); print(f"pool   -> {tuple(v.shape)}")
    print(f"logits -> {tuple(model.fc(v).shape)}")
''')

co('''\
ltr, lva, lte = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx,
                               aug="byte", in_channels=1)
_, hist, summ = M.train_model(model, ltr, lva, N_CLASSES, tag="ByteAttnNet-v1",
                              save_path=CFG.mdl("proposed_v1.pth"))
M.plot_history(hist, title="ByteAttnNet v1 — training",
               save=CFG.fig("task2_proposed_history.png"))
plt.show()
hist.to_csv(CFG.art("task2_history_ByteAttnNet-v1.csv"), index=False)

# what did GeM's learnable exponent settle on?
if hasattr(model.pool, "p"):
    p = float(model.pool.p.detach().clamp(1, 8))
    print(f"\\nlearned GeM exponent p = {p:.3f}  "
          f"({'≈ average pooling' if p < 1.5 else 'between average and max pooling' if p < 5 else 'close to max pooling'})")
    print("This is the network telling us whether family identity lives in the")
    print("average texture energy (low p) or in a few peak blocks (high p).")
''')

co('''\
y, p_, pr = M.predict(model, lte)
mm_prop = M.full_metrics(y, p_, pr, CLASS_NAMES, name="ByteAttnNet-v1")
mm_prop.update(params=summ["params"], size_mb=summ["size_mb"],
               train_seconds=summ["train_seconds"], best_epoch=summ["best_epoch"],
               inference_ms_per_image=M.measure_inference_ms(model, lte))
np.savez_compressed(CFG.art("task2_proposed_test_predictions.npz"),
                    test_index=te_idx, y_true=y, y_pred=p_, y_prob=pr)
M.save_json({k: v for k, v in mm_prop.items() if not k.startswith("_")},
            CFG.art("task2_proposed_metrics.json"))

print(f"accuracy      {mm_prop['accuracy']:.4f}")
print(f"MACRO-F1      {mm_prop['f1_macro']:.4f}   <- headline")
print(f"weighted-F1   {mm_prop['f1_weighted']:.4f}")
print(f"macro-recall  {mm_prop['recall_macro']:.4f}")
print(f"macro AUC     {mm_prop.get('roc_auc_macro_ovr', float('nan')):.4f}")
print(f"MCC           {mm_prop['mcc']:.4f}")

M.plot_confusion(mm_prop["_confusion"], CLASS_NAMES,
                 title="ByteAttnNet v1 — confusion matrix (held-out test set)",
                 save=CFG.fig("task2_proposed_cm.png")); plt.show()
M.plot_roc_ovr(y, pr, CLASS_NAMES, title="ByteAttnNet v1 — ROC one-vs-rest",
               save=CFG.fig("task2_proposed_roc.png")); plt.show()
M.plot_pr_ovr(y, pr, CLASS_NAMES, title="ByteAttnNet v1 — precision-recall",
              save=CFG.fig("task2_proposed_pr.png")); plt.show()
display(mm_prop["_report_df"].iloc[:N_CLASSES].sort_values("recall").round(3))
''')

# ------------------------------------------------------- comparison
md("""\
---
## 6 · First results — does attention help?

We compare against the Task-2a baselines. If `task2_baseline_comparison.csv` is
not on disk (because you are running this notebook standalone), the cell below
retrains a compact baseline set so the comparison is still real rather than
missing.""")

co('''\
from pathlib import Path
bl_csv = Path(CFG.art("task2_baseline_comparison.csv"))
alt = list(Path("/kaggle/input").rglob("*task2_baseline_comparison.csv")) \\
      if Path("/kaggle/input").exists() else []
if bl_csv.exists():
    base_tbl = pd.read_csv(bl_csv)
    print("loaded baselines from", bl_csv)
elif alt:
    base_tbl = pd.read_csv(alt[0]); print("loaded baselines from", alt[0])
else:
    print("No Task-2a table found -> retraining a compact baseline set here.\\n"
          "(On Kaggle you can instead add notebook 2a's output as a dataset.)")
    rows = []
    for nm in (["SimpleCNN", "ResNet18"] if CFG.fast
               else ["SimpleCNN", "ResNet50", "DenseNet121"]):
        M.set_seed(CFG.seed)
        bm, ic = M.build_baseline(nm, N_CLASSES, pretrained=not CFG.fast)
        a, b, c = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx,
                                 aug="byte", in_channels=ic)
        _, _, s = M.train_model(bm, a, b, N_CLASSES, tag=nm, verbose=False)
        yy, pp, prr = M.predict(bm, c)
        r = M.full_metrics(yy, pp, prr, CLASS_NAMES, name=nm)
        r.update(params=s["params"], size_mb=s["size_mb"],
                 train_seconds=s["train_seconds"])
        rows.append(r)
        np.savez_compressed(CFG.art(f"task2_fallback_preds_{nm}.npz"),
                            y_true=yy, y_pred=pp, y_prob=prr, test_index=te_idx)
        print(f"  {nm}: macro-F1 {r['f1_macro']:.4f}")
    base_tbl = M.metrics_frame(rows)
    base_tbl.to_csv(CFG.art("task2_baseline_comparison.csv"), index=False)
''')

co('''\
comp = pd.concat([base_tbl,
                  M.metrics_frame([mm_prop])], ignore_index=True)
comp = comp.sort_values("f1_macro", ascending=False).reset_index(drop=True)
cols = [c for c in ["model", "accuracy", "f1_macro", "f1_weighted",
                    "precision_macro", "recall_macro", "roc_auc_macro_ovr",
                    "mcc", "params", "size_mb", "train_seconds",
                    "inference_ms_per_image"] if c in comp.columns]
display(comp[cols].round(4))
comp.to_csv(CFG.art("task2_all_models_comparison.csv"), index=False)

best_base = comp[comp.model != "ByteAttnNet-v1"].iloc[0]
delta = mm_prop["f1_macro"] - float(best_base.f1_macro)
print(f"\\nbest baseline      : {best_base.model}  macro-F1 {best_base.f1_macro:.4f} "
      f"({best_base.params/1e6:.1f} M params)")
print(f"ByteAttnNet v1     : macro-F1 {mm_prop['f1_macro']:.4f} "
      f"({mm_prop['params']/1e6:.1f} M params)")
print(f"difference         : {delta:+.4f} macro-F1  "
      f"({100*delta/max(float(best_base.f1_macro),1e-9):+.1f}% relative)")
print(f"parameter ratio    : {float(best_base.params)/mm_prop['params']:.1f}x "
      f"fewer parameters than the best baseline")
print("\\nNOTE: this is ONE split. Whether the difference is real is decided in")
print("Task 3 by 5-fold CV + Wilcoxon signed-rank + McNemar, not here.")
''')

co('''\
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5))
c = ["tab:red" if m == "ByteAttnNet-v1" else "tab:grey" for m in comp.model]
ax[0].barh(comp.model[::-1], comp.f1_macro[::-1], color=c[::-1])
for i, v in enumerate(comp.f1_macro[::-1]):
    ax[0].text(v + .004, i, f"{v:.4f}", va="center", fontsize=9)
ax[0].set(xlabel="macro-F1 on the held-out test set",
          title="First results — macro-F1 (red = proposed)",
          xlim=(0, min(1.05, comp.f1_macro.max() * 1.18)))
ax[1].scatter(comp.params / 1e6, comp.f1_macro, s=110, c=c)
for _, r in comp.iterrows():
    ax[1].annotate(r.model, (r.params / 1e6, r.f1_macro), fontsize=8.5,
                   xytext=(5, 5), textcoords="offset points")
ax[1].set(xscale="log", xlabel="parameters (millions, log scale)",
          ylabel="macro-F1", title="Accuracy per parameter — the efficiency claim")
fig.tight_layout()
fig.savefig(CFG.fig("task2_first_results.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

co('''\
# per-class: where exactly did the proposed model gain or lose?
prop_rec = mm_prop["_report_df"].iloc[:N_CLASSES]["recall"]
bl_files = sorted(Path(CFG.out_dir).rglob("*task2_perclass_*.csv"))
if bl_files:
    bl = {f.stem.split("task2_perclass_")[-1]: pd.read_csv(f, index_col=0)["recall"]
          for f in bl_files if "recall_all" not in f.stem}
    base_rec = pd.DataFrame(bl).mean(axis=1)
    d = (prop_rec - base_rec).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(range(len(d)), d.values,
           color=["tab:red" if v < 0 else "tab:green" for v in d.values])
    ax.axhline(0, c="k", lw=1)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d.index, rotation=90, fontsize=7.5)
    ax.set(ylabel="recall(proposed) - mean recall(baselines)",
           title="Per-class recall change. Green = ByteAttnNet fixed a family the "
                 "baselines struggled with; red = it made one worse.")
    fig.tight_layout()
    fig.savefig(CFG.fig("task2_perclass_delta.png"), dpi=130, bbox_inches="tight")
    plt.show()
    print("biggest gains :", ", ".join(f"{i} ({v:+.2f})" for i, v in d.tail(5)[::-1].items()))
    print("biggest losses:", ", ".join(f"{i} ({v:+.2f})" for i, v in d.head(5).items()))
else:
    print("run notebook 2a first for the per-class comparison")
''')

md("""\
---
## 7 · Pillar-B Q4 — Honest reading of these first results

Fill this in from the numbers your run produced. State plainly:

* **Did it beat the best baseline on macro-F1, and by how much?** Report the
  delta and the parameter ratio. One split is *not* evidence — say so, and point
  forward to Task 3's 5-fold CV and Wilcoxon test.
* **Which families improved and which got worse?** The per-class delta plot
  above is the answer. If a rare family got worse, say it and say why you think
  so (usually: class weighting traded majority precision for minority recall).
* **What the learned GeM exponent tells you.** A low *p* means family identity is
  in the average texture; a high *p* means it is in a few distinctive blocks.
  Either is interesting; report what you actually got.
* **What has NOT been tested yet.** At this point every attention claim is still
  unproven — CBAM and CoordAtt are switched on together. Task 3's ablation is
  what turns "we designed it this way because…" into evidence, and it is
  entirely possible that CoordAtt contributes nothing. Say that up front; the
  rubric rewards an honest negative result over a mysterious win.
* **The published-number gap.** We are below the 99.2–99.4 % Malimg figures. The
  reason is measured, not guessed: Task 2a §3 shows what a random split is worth
  on this data.""")

co('''\
M.save_json(dict(hparams=PROPOSED_CFG,
                 metrics={k: v for k, v in mm_prop.items() if not k.startswith("_")},
                 summary=summ, class_names=CLASS_NAMES,
                 eval_scope=CFG.eval_scope,
                 gem_p=float(model.pool.p.detach()) if hasattr(model.pool, "p") else None),
            CFG.art("task2_proposed_summary.json"))
print("Task 2b artefacts:")
for p in sorted(Path(CFG.out_dir).rglob("*")):
    if p.is_file() and "task2" in p.name:
        print(f"  {p.stat().st_size/1024:9.1f} KB  {p.name}")
print("\\nNext: Group00_MaleBin_task3_improvement_ablation.ipynb")
''')

build(C, REPO / "code" / "task2" / f"{PREFIX}_task2_proposed_model.ipynb",
      "CSE475 Task 2b - ByteAttnNet proposed model")
