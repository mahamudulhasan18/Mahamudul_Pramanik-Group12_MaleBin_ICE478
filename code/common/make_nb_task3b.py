"""Generate code/task3/Group00_MaleBin_task3_explainability.ipynb"""
from nbtool import build, writefile_cell, BOOT, LOAD_SPLIT, REPO, PREFIX

C: list[tuple[str, str]] = []
md = lambda s: C.append(("md", s))
co = lambda s: C.append(("code", s))

md(f"""\
# CSE475 Task 3b — Explainability (Grad-CAM + LIME)
## {PREFIX} · Track 3 (CNN + Attention)

### Read this before the figures — the honest limitation, stated first

The brief (§6.2) is explicit:

> **Malware byte-images:** they classify well, but Grad-CAM/LIME on them means
> little (no real "regions" to point at). Say this instead of inventing an
> explanation.

We agree, and we are not going to pretend otherwise. In a photograph, a Grad-CAM
blob over a dog's face means *"the model used the dog's face"*, because a face is
a thing a human can independently recognise. In a byte-plot there is no dog.
A bright blob at pixel (137, 82) is a byte value at one offset in one file; a
human cannot look at it and confirm or refute anything. So a Grad-CAM heat-map
here is **not** evidence that the model is right, and we do not present it as
such.

**What we do instead — three levels, weakest to strongest:**

| § | Method | What it can honestly tell us |
|---|---|---|
| 3 | **Grad-CAM / Grad-CAM++** (required) | Which parts of the input the gradient says the logit was sensitive to. Presented as a *diagnostic*, not an explanation. |
| 4 | **Byte-offset profile** of the same heat-map | Because row *r* holds bytes `[rW, rW+W)`, collapsing the map along rows gives importance as a function of **relative file position**. "The model relied on bytes at 12–19 % of the file" *is* a checkable claim about PE layout — unlike "it looked at this blob". |
| 5 | **Row-band occlusion** | Actually blank out a band of file offsets and measure the probability drop. This is **causal**, not gradient-inferred, so it can *confirm or refute* the Grad-CAM story rather than just decorate it. |
| 6 | **LIME**, twice (required) | Once with the naive `quickshift` segmentation (which assumes colour-coherent regions that byte-plots do not have) and once with a row-band segmentation (where every segment is a real range of file offsets). The comparison shows *why* the default LIME setup is misleading here. |
| 7 | **Coordinate-attention row gates** | Read the model's own learned per-row weights directly — no attribution method, no approximation. If §5's occlusion peaks line up with these gates, the mechanism we designed in Task 2b is doing what we claimed. |

Section 8 states what all of this does and does not establish.

⏱ ~10–20 min. This notebook reuses `models/{PREFIX}_best.pth` from Task 3a and
retrains a model only if that checkpoint is missing.""")

C.append(writefile_cell())
co(BOOT)
co(LOAD_SPLIT)

# ---------------------------------------------------------------- load model
md("""\
---
## 2 · Load the final model

Prefers the Task-3a checkpoint. On Kaggle, add notebook 3a's output as a dataset
(*Add Data → Notebook Output*) and it will be found automatically. If nothing is
found, a model is trained here so the notebook still produces real
explanations.""")

co('''\
import torch
from pathlib import Path

def find_checkpoint():
    names = ["best.pth", "proposed_v1.pth"]
    cands = [Path(CFG.mdl(n)) for n in names]
    cands += [Path(CFG.out_dir) / "models" / f"{CFG.prefix}_{n}" for n in names]
    if Path("/kaggle/input").exists():
        for n in names:
            cands += sorted(Path("/kaggle/input").rglob(f"*{n}"))
    for c in cands:
        if c.exists():
            return c
    return None

ckpt_path = find_checkpoint()
if ckpt_path is not None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ck.get("hparams") or {})
    hp.pop("n_classes", None)
    hp.pop("in_ch", None)
    saved_names = ck.get("class_names")
    if saved_names and list(saved_names) != CLASS_NAMES:
        print("!! checkpoint class list differs from the current scope. "
              "Set CFG.eval_scope to match the checkpoint before trusting this.")
    hp = {k: (tuple(v) if isinstance(v, list) else v) for k, v in hp.items()}
    model = M.ByteAttnNet(N_CLASSES, in_ch=1, **hp)
    model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    XAI_SIZE = int(ck.get("img_size", CFG.img_size))
    print(f"loaded {ckpt_path}")
    print(f"  hparams   : {hp}")
    print(f"  input size: {XAI_SIZE}")
else:
    print("No checkpoint found -> training a model here so the explanations are real.")
    M.set_seed(CFG.seed)
    kw = dict(channels=(16, 32, 64, 96)) if CFG.fast else {}
    model = M.ByteAttnNet(N_CLASSES, in_ch=1, attention="cbam+coord", **kw)
    a, b, c = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx, aug="byte")
    M.train_model(model, a, b, N_CLASSES, tag="XAI-model",
                  save_path=CFG.mdl("xai_fallback.pth"))
    XAI_SIZE = CFG.img_size

model = model.to(M.DEVICE).eval()
print(f"\\nparams {M.count_params(model):,} | device {M.DEVICE}")
''')

co('''\
# predictions on the held-out test set, so we can pick a correct and a wrong case
_, _, lte = M.make_loaders(imgs, df, tr_idx, va_idx, te_idx, aug="none",
                           in_channels=1, out_size=XAI_SIZE)
y_true, y_pred, y_prob = M.predict(model, lte)
conf = y_prob.max(axis=1)
correct = y_true == y_pred
print(f"test set: {len(y_true)} images | accuracy {correct.mean():.4f} | "
      f"macro-F1 {M.full_metrics(y_true, y_pred, y_prob, CLASS_NAMES)['f1_macro']:.4f}")

# Pick deliberately, not at random:
#   CASE A -- a CONFIDENT CORRECT prediction: what the model relies on when right
#   CASE B -- a CONFIDENT WRONG prediction: the interesting failure, because a
#             low-confidence miss is just "the model was unsure"
ok_idx = np.flatnonzero(correct)
bad_idx = np.flatnonzero(~correct)
if len(bad_idx) == 0:
    print("no misclassified test images -- using the least confident correct one "
          "as CASE B instead")
    bad_idx = np.array([int(np.argmin(np.where(correct, conf, np.inf)))])
CASE_A = int(ok_idx[np.argmax(conf[ok_idx])])
CASE_B = int(bad_idx[np.argmax(conf[bad_idx])])

def describe(local_i, label):
    g = int(te_idx[local_i])
    print(f"\\n{label}")
    print(f"  file        : {Path(df.path.iloc[g]).name}")
    print(f"  true family : {CLASS_NAMES[y_true[local_i]]}")
    print(f"  predicted   : {CLASS_NAMES[y_pred[local_i]]}  "
          f"(p = {y_prob[local_i, y_pred[local_i]]:.4f})")
    print(f"  p(true)     : {y_prob[local_i, y_true[local_i]]:.4f}")
    top = np.argsort(-y_prob[local_i])[:4]
    print("  top-4       : " + ", ".join(
        f"{CLASS_NAMES[t]} {y_prob[local_i, t]:.3f}" for t in top))
    return g

G_A = describe(CASE_A, "CASE A — confident CORRECT prediction")
G_B = describe(CASE_B, "CASE B — confident WRONG prediction")
CASES = [("A_correct", CASE_A, G_A), ("B_wrong", CASE_B, G_B)]
''')

# ---------------------------------------------------------------- Grad-CAM
md("""\
---
## 3 · Grad-CAM and Grad-CAM++ (required by the brief)

Grad-CAM (Selvaraju et al., ICCV 2017) weights the last convolutional feature
maps by the average gradient of the target logit with respect to each map, sums
them and rectifies:

$$\\text{cam} = \\mathrm{ReLU}\\Big(\\sum_k \\alpha_k A_k\\Big),\\qquad
\\alpha_k = \\frac{1}{HW}\\sum_{i,j}\\frac{\\partial y_c}{\\partial A_k^{ij}}$$

We hook the deepest stage of `ByteAttnNet` — i.e. *after* CBAM and coordinate
attention, so the map already reflects the attention gates.

Grad-CAM++ replaces $\\alpha_k$ with a second-order weighting, which behaves
better when several disjoint regions support the same class. That is the normal
situation for a byte-plot (code section *and* resource section both matter), so
we show both.""")

co('''\
def tensor_for(local_i):
    ds = M.ByteImageDataset(imgs, df.label.values, te_idx, out_size=XAI_SIZE,
                            aug="none", in_channels=1)
    x, y = ds[local_i]
    return x[None].to(M.DEVICE), int(y)

cam_store = {}
for tag, li, gi in CASES:
    x, y = tensor_for(li)
    cg = M.GradCAM(model)
    cam_true, _, p_true = cg(x, class_idx=int(y_true[li]), plus_plus=False)
    cam_pred, _, p_pred = cg(x, class_idx=int(y_pred[li]), plus_plus=False)
    campp, _, _ = cg(x, class_idx=int(y_pred[li]), plus_plus=True)
    cg.remove()
    cam_store[tag] = dict(x=x, cam_true=cam_true, cam_pred=cam_pred,
                          campp=campp, li=li, gi=gi)

    raw = imgs[gi]
    EXT = [0, 1, 1, 0]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))
    ax[0].imshow(raw, cmap="gray", vmin=0, vmax=255, extent=EXT, aspect="equal")
    ax[0].set(title=f"byte-plot\\n{Path(df.path.iloc[gi]).name}")
    for a, cam, ttl in [
            (ax[1], cam_pred, f"Grad-CAM for PREDICTED\\n{CLASS_NAMES[y_pred[li]]} "
                              f"(p={y_prob[li, y_pred[li]]:.3f})"),
            (ax[2], cam_true, f"Grad-CAM for TRUE\\n{CLASS_NAMES[y_true[li]]} "
                              f"(p={y_prob[li, y_true[li]]:.3f})"),
            (ax[3], campp, f"Grad-CAM++ for PREDICTED\\n{CLASS_NAMES[y_pred[li]]}")]:
        a.imshow(raw, cmap="gray", vmin=0, vmax=255, extent=EXT, aspect="equal")
        a.imshow(cam, cmap="jet", alpha=.45, extent=EXT, aspect="equal")
        a.set(title=ttl)
    for a in ax:
        a.set_xlabel("byte within the row")
        a.set_ylabel("byte offset in the file")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"3 · Grad-CAM — CASE {tag}   true={CLASS_NAMES[y_true[li]]}, "
                 f"pred={CLASS_NAMES[y_pred[li]]}", fontsize=13)
    fig.tight_layout()
    fig.savefig(CFG.fig(f"task3_gradcam_{tag}.png"), dpi=130, bbox_inches="tight")
    plt.show()
''')

md("""\
**Reading — and what we refuse to claim.** The maps differ between the predicted
and the true class in CASE B, which tells us the two classes are supported by
different parts of the input. That is a genuine observation. What we will *not*
say is "the model looked at the malicious payload" or any similar story: nothing
in these images licenses that. The blobs are where the gradient is large, full
stop. Sections 4–5 are where we turn this into something checkable.""")

# ---------------------------------------------------------------- byte offsets
md("""\
---
## 4 · From heat-map to **byte offsets** — the reading that is actually meaningful

The one structural fact we can exploit: a byte-plot of pixel width *W* stores
bytes `[rW, rW+W)` in row *r*. So the **row-mean of the Grad-CAM map is an
importance profile over relative file position**, and that maps onto real PE
structure: header at the top, code sections, then data/resources, then padding.

By contrast the *column* profile should be roughly flat — a column index is an
artefact of the chosen row width and carries no file semantics. If the column
profile is strongly peaked, that is a warning sign that the model latched onto a
rendering artefact rather than content, and we check for exactly that.""")

co('''\
prof_store = {}
for tag, li, gi in CASES:
    cam = cam_store[tag]["cam_pred"]
    b = M.cam_to_byte_offsets(cam, img_width_px=imgs.shape[2], top_frac=0.25)
    prof_store[tag] = b
    print(f"\\nCASE {tag}: top attributed byte-offset bands "
          f"(pred = {CLASS_NAMES[y_pred[li]]})")
    for k, bd in enumerate(b["bands"][:4], 1):
        print(f"  {k}. rows {bd['row_from']:>4}-{bd['row_to']:<4} -> "
              f"{bd['file_pct_from']:>5.1f}% - {bd['file_pct_to']:>5.1f}% of the "
              f"file   (mean importance {bd['mean_importance']:.3f})")
    cv_col = float(np.std(b["col_profile"]) / (np.mean(b["col_profile"]) + 1e-9))
    cv_row = float(np.std(b["row_profile"]) / (np.mean(b["row_profile"]) + 1e-9))
    print(f"  profile spread: rows CV={cv_row:.3f}  columns CV={cv_col:.3f}")
    print("   -> " + ("row profile is more structured than the column profile, "
                      "which is what we want: attribution follows FILE POSITION."
                      if cv_row > cv_col else
                      "columns are as structured as rows -- suspicious, this may "
                      "be a rendering artefact rather than file content."))
''')

co('''\
fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
for r, (tag, li, gi) in enumerate(CASES):
    b = prof_store[tag]
    cam = cam_store[tag]["cam_pred"]
    ax = axes[r, 0]
    ax.imshow(imgs[gi], cmap="gray", vmin=0, vmax=255, aspect="auto",
              extent=[0, 100, 100, 0])
    ax.imshow(cam, cmap="jet", alpha=.4, aspect="auto", extent=[0, 100, 100, 0])
    for bd in b["bands"][:3]:
        ax.axhspan(bd["file_pct_from"], bd["file_pct_to"], color="lime",
                   alpha=.18)
        ax.text(101, (bd["file_pct_from"] + bd["file_pct_to"]) / 2,
                f"{bd['file_pct_from']:.0f}-{bd['file_pct_to']:.0f}%",
                fontsize=8, va="center")
    ax.set(xlabel="byte within the row (%)", ylabel="position in the file (%)",
           title=f"CASE {tag}: Grad-CAM with the top 3 offset bands marked")

    ax = axes[r, 1]
    h = len(b["row_profile"])
    pos = np.linspace(0, 100, h)
    ax.plot(b["row_profile"], pos, lw=1.6, label="row profile (= file offset)")
    ax.plot(np.interp(np.linspace(0, 1, h),
                      np.linspace(0, 1, len(b["col_profile"])), b["col_profile"]),
            pos, lw=1.2, ls=":", color="grey",
            label="column profile (should be flat)")
    ax.invert_yaxis()
    ax.fill_betweenx(pos, 0, b["row_profile"], alpha=.25)
    ax.set(xlabel="normalised importance", ylabel="position in the file (%)",
           title=f"CASE {tag}: importance vs byte offset\\n"
                 f"true={CLASS_NAMES[y_true[li]]}, pred={CLASS_NAMES[y_pred[li]]}")
    ax.legend(fontsize=8, loc="lower right")
fig.suptitle("4 · Grad-CAM re-expressed as importance over relative file position",
             fontsize=13)
fig.tight_layout()
fig.savefig(CFG.fig("task3_byte_offset_profile.png"), dpi=130, bbox_inches="tight")
plt.show()
''')

co('''\
# Aggregate over many test images per family: is the attended offset band
# CONSISTENT within a family? A consistent band is a family-level structural
# claim; a scattered one means the map is noise.
rows = []
n_per = 3 if CFG.fast else 8
cg = M.GradCAM(model)
for lab in range(N_CLASSES):
    where = np.flatnonzero((y_true == lab) & correct)[:n_per]
    for li in where:
        x, _ = tensor_for(int(li))
        cam, _, _ = cg(x, class_idx=lab)
        prof = cam.mean(axis=1)
        prof = (prof - prof.min()) / (np.ptp(prof) + 1e-9)
        rows.append(dict(family=CLASS_NAMES[lab],
                         peak_pct=100.0 * int(np.argmax(prof)) / len(prof),
                         centroid_pct=100.0 * float((prof * np.arange(len(prof))).sum()
                                                    / (prof.sum() + 1e-9)) / len(prof)))
cg.remove()
pk = pd.DataFrame(rows)
if len(pk):
    agg = pk.groupby("family").agg(n=("peak_pct", "size"),
                                   peak_mean=("peak_pct", "mean"),
                                   peak_std=("peak_pct", "std"),
                                   centroid_mean=("centroid_pct", "mean"),
                                   centroid_std=("centroid_pct", "std")).round(1)
    agg = agg.sort_values("peak_std")
    display(agg)
    agg.to_csv(CFG.art("task3_offset_consistency.csv"))

    fig, ax = plt.subplots(figsize=(13, 5))
    o = agg.dropna(subset=["peak_std"])
    ax.errorbar(range(len(o)), o.peak_mean, yerr=o.peak_std, fmt="o", capsize=3)
    ax.set_xticks(range(len(o)))
    ax.set_xticklabels(o.index, rotation=90, fontsize=7.5)
    ax.set(ylabel="attended position in the file (%)", ylim=(-5, 105),
           title="4 · Where in the file does the model look, per family?\\n"
                 "small error bar = a CONSISTENT structural claim about that "
                 "family; large error bar = the attribution is unstable and "
                 "should not be interpreted")
    fig.tight_layout()
    fig.savefig(CFG.fig("task3_offset_consistency.png"), dpi=130,
                bbox_inches="tight")
    plt.show()
    tight = o[o.peak_std < 10]
    print(f"families with a stable attended offset (std < 10% of the file): "
          f"{len(tight)} of {len(o)}")
    print("  " + (", ".join(f"{i} ({r.peak_mean:.0f}%)" for i, r in tight.head(10).iterrows())
                  or "none"))
''')

# ---------------------------------------------------------------- occlusion
md("""\
---
## 5 · Row-band occlusion — the causal check

Grad-CAM is a *gradient* statement; it can be large where perturbing the input
changes nothing. So we test the claim directly: blank out one horizontal band of
the byte-plot at a time (i.e. one contiguous range of file offsets) and record
how much the predicted-class probability falls. A band whose removal costs
nothing was not being used, whatever the heat-map said.

If the occlusion peaks line up with §4's Grad-CAM bands, the two methods agree
and the offset reading is trustworthy. If they do not, Grad-CAM was decorative —
and we report that, because it is the honest outcome the brief asks for.""")

co('''\
occ_store = {}
NB = 8 if CFG.fast else 24
for tag, li, gi in CASES:
    x, _ = tensor_for(li)
    cls = int(y_pred[li])
    drops, base = M.occlusion_by_row_band(model, x, cls, n_bands=NB, fill=0.0)
    drops_mean, _ = M.occlusion_by_row_band(model, x, cls, n_bands=NB,
                                            fill=float(x.mean()))
    occ_store[tag] = dict(drops=drops, base=base, drops_mean=drops_mean)

    cam = cam_store[tag]["cam_pred"]
    prof = cam.mean(axis=1)
    edges = np.linspace(0, len(prof), NB + 1).astype(int)
    cam_band = np.array([prof[edges[i]:edges[i + 1]].mean() for i in range(NB)])
    cam_band = (cam_band - cam_band.min()) / (np.ptp(cam_band) + 1e-9)
    occ_norm = (drops - drops.min()) / (np.ptp(drops) + 1e-9)
    r = float(np.corrcoef(cam_band, occ_norm)[0, 1]) if np.ptp(drops) > 0 else float("nan")
    occ_store[tag]["agreement_r"] = r

    centres = 100 * (np.arange(NB) + .5) / NB
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    ax[0].bar(centres, drops, width=100 / NB * .85, label="fill = 0x00 (zero pad)")
    ax[0].plot(centres, drops_mean, "o--", color="tab:orange",
               label="fill = image mean")
    ax[0].axhline(0, c="k", lw=1)
    ax[0].set(xlabel="position in the file (%)",
              ylabel=f"drop in p({CLASS_NAMES[cls]})",
              title=f"CASE {tag}: causal importance by offset band\\n"
                    f"baseline p = {base:.4f}")
    ax[0].legend(fontsize=8)

    ax[1].plot(centres, cam_band, "s-", label="Grad-CAM (normalised)")
    ax[1].plot(centres, occ_norm, "o-", label="occlusion (normalised)")
    ax[1].set(xlabel="position in the file (%)", ylabel="normalised importance",
              title=f"CASE {tag}: do gradient and causal importance agree?\\n"
                    f"Pearson r = {r:.3f}")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(CFG.fig(f"task3_occlusion_{tag}.png"), dpi=130, bbox_inches="tight")
    plt.show()

    k = int(np.argmax(drops))
    print(f"CASE {tag}: most causally important band = "
          f"{100*k/NB:.0f}-{100*(k+1)/NB:.0f}% of the file "
          f"(removing it costs {drops[k]:.4f} probability)")
    print(f"           Grad-CAM / occlusion agreement r = {r:.3f}  -> " +
          ("the two methods agree; the offset reading is supported"
           if r > 0.4 else
           "WEAK agreement: Grad-CAM is NOT tracking causal importance for this "
           "sample. Report this rather than the heat-map."))
''')

# ---------------------------------------------------------------- LIME
md("""\
---
## 6 · LIME (required) — and why the default segmentation is wrong here

LIME (Ribeiro et al., KDD 2016) perturbs *superpixels* and fits a sparse linear
surrogate. Everything therefore depends on what a "superpixel" is.

* **Default (`quickshift`)** looks for colour-coherent regions. That is a natural
  image assumption. A byte-plot's texture is not spatially coherent in that
  sense, so quickshift returns blobs that do not correspond to anything — and
  LIME will still confidently rank them.
* **Ours (row-band grid)** partitions the image into 16 row bands × 4 columns, so
  every segment is a contiguous range of **file offsets**. A weight on such a
  segment is a statement about a region of the binary.

We run both, side by side, so the difference is visible rather than asserted.""")

co('''\
lime_store = {}
NS = 120 if CFG.fast else 800
for tag, li, gi in CASES:
    raw = imgs[gi]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes[0].set(title=f"CASE {tag}\\ntrue={CLASS_NAMES[y_true[li]]}, "
                      f"pred={CLASS_NAMES[y_pred[li]]}")
    axes[0].axis("off")

    for ax, seg, ttl in [
            (axes[1], "grid", "row-band segmentation (OURS)\\n"
                              "each segment = a range of file offsets"),
            (axes[2], "quickshift", "quickshift segmentation (LIME default)\\n"
                                    "assumes colour-coherent regions")]:
        try:
            ex = M.lime_explain(model, raw, CLASS_NAMES, in_channels=1,
                                out_size=XAI_SIZE, num_samples=NS,
                                segments=seg, top_labels=3)
            lab = int(y_pred[li]) if int(y_pred[li]) in ex.top_labels else ex.top_labels[0]
            im, mask = ex.get_image_and_mask(lab, positive_only=True,
                                             num_features=6, hide_rest=False)
            from skimage.segmentation import mark_boundaries
            ax.imshow(mark_boundaries(im / 255.0, mask))
            w = dict(ex.local_exp[lab])
            top = sorted(w.items(), key=lambda t: -abs(t[1]))[:5]
            ax.set(title=ttl + f"\\nR^2 of the local surrogate = "
                               f"{ex.score:.3f}" if hasattr(ex, "score") else ttl)
            lime_store[(tag, seg)] = dict(r2=getattr(ex, "score", float("nan")),
                                          top=top, n_segments=len(w))
            print(f"CASE {tag} / {seg:<10s}: {len(w)} segments, "
                  f"surrogate R^2 = {getattr(ex, 'score', float('nan')):.3f}, "
                  f"top weights {[round(v, 3) for _, v in top]}")
        except Exception as e:
            ax.text(.5, .5, f"LIME failed:\\n{type(e).__name__}", ha="center",
                    transform=ax.transAxes)
            print(f"CASE {tag} / {seg}: LIME failed -> {e!r}")
        ax.axis("off")
    fig.suptitle(f"6 · LIME, two segmentations — CASE {tag}", fontsize=13)
    fig.tight_layout()
    fig.savefig(CFG.fig(f"task3_lime_{tag}.png"), dpi=130, bbox_inches="tight")
    plt.show()
''')

co('''\
if lime_store:
    ls = pd.DataFrame([dict(case=k[0], segmentation=k[1], surrogate_r2=v["r2"],
                            n_segments=v["n_segments"])
                       for k, v in lime_store.items()])
    display(ls.round(3))
    ls.to_csv(CFG.art("task3_lime_surrogate_quality.csv"), index=False)
    print("The surrogate R^2 is the honest health check on a LIME explanation:")
    print("it is how well the sparse LINEAR model reproduces the CNN locally.")
    print("A low R^2 means the pretty picture is explaining a fit that does not")
    print("hold -- and low R^2 is the normal outcome on byte-plots.")
''')

# ---------------------------------------------------------------- coord gates
md("""\
---
## 7 · The model's own coordinate-attention row gates

No attribution method, no surrogate, no perturbation: we read the learned
per-row and per-column gates straight out of the `CoordAtt` modules. This is the
mechanism we designed in Task 2b, so this section is the direct test of whether
it does what we claimed.

The per-row gate is averaged over channels and plotted against relative file
position. If it is essentially flat, coordinate attention is *not* being used and
we should say so — which would also explain any null result in Task 3a's
attention ablation.""")

co('''\
coords = [m for m in model.modules() if isinstance(m, M.CoordAtt)]
print(f"CoordAtt modules found: {len(coords)}")
if not coords:
    print("This model has no coordinate attention (attention='none'/'se'/'cbam'), "
          "so there is nothing to read here.")
else:
    for tag, li, gi in CASES:
        x, _ = tensor_for(li)
        with torch.no_grad():
            model(x)
        fig, ax = plt.subplots(1, len(coords), figsize=(4.2 * len(coords), 4.4),
                               squeeze=False)
        for j, ca in enumerate(coords):
            ah = ca.last_h[0].mean(0).squeeze().float().cpu().numpy()   # (H,)
            aw = ca.last_w[0].mean(0).squeeze().float().cpu().numpy()   # (W,)
            pos = np.linspace(0, 100, len(ah))
            a = ax[0, j]
            a.plot(ah, pos, lw=1.8, label="per-ROW gate (file offset)")
            a.plot(np.interp(np.linspace(0, 1, len(ah)),
                             np.linspace(0, 1, len(aw)), aw), pos,
                   lw=1.2, ls=":", color="grey", label="per-COLUMN gate")
            a.invert_yaxis()
            a.axvline(0.5, color="r", ls="--", lw=.9)
            flat = float(np.std(ah))
            a.set(xlabel="gate value (sigmoid)", ylabel="position in the file (%)",
                  title=f"CoordAtt stage {j+1}\\nrow-gate std = {flat:.4f}"
                        + ("  (FLAT -> unused)" if flat < .01 else ""))
            a.legend(fontsize=7.5)
        fig.suptitle(f"7 · Learned coordinate-attention gates — CASE {tag} "
                     f"(pred = {CLASS_NAMES[y_pred[li]]})", fontsize=12.5)
        fig.tight_layout()
        fig.savefig(CFG.fig(f"task3_coordatt_gates_{tag}.png"), dpi=130,
                    bbox_inches="tight")
        plt.show()
''')

co('''\
# Does the deepest row gate agree with the CAUSAL occlusion profile?
# This is the strongest single check in the notebook: the mechanism we built and
# the causal measurement are two independent things, so agreement is meaningful.
if coords:
    for tag, li, gi in CASES:
        x, _ = tensor_for(li)
        with torch.no_grad():
            model(x)
        ah = coords[-1].last_h[0].mean(0).squeeze().float().cpu().numpy()
        d = occ_store[tag]["drops"]
        gate_band = np.interp(np.linspace(0, 1, len(d)),
                              np.linspace(0, 1, len(ah)), ah)
        if np.ptp(d) > 0 and np.ptp(gate_band) > 0:
            r = float(np.corrcoef(gate_band, d)[0, 1])
        else:
            r = float("nan")
        print(f"CASE {tag}: corr(deepest CoordAtt row gate, occlusion drop) "
              f"= {r:+.3f}")
        print("   -> " + ("the gate the model learned matches what actually "
                          "matters causally: the mechanism works as designed."
                          if r > 0.35 else
                          "no clear agreement. Report this: the row gate is not "
                          "tracking causal offset importance for this sample, so "
                          "the Task-2b justification is not confirmed here."))
''')

# ---------------------------------------------------------------- limitations
md("""\
---
## 8 · What this establishes, and what it does not (brief §6.2)

### Established
* **Grad-CAM and LIME were applied** to one correct and one incorrect prediction,
  as required, with both the required and a byte-plot-appropriate segmentation.
* **The byte-offset reformulation is a real, checkable statement.** Because row
  index = file offset, "importance is concentrated at *x*–*y* % of the file" can
  be compared against PE layout and against other samples of the same family.
  §4's per-family consistency table says for which families that claim is stable.
* **Occlusion gives causal evidence**, and §5 reports the correlation between it
  and Grad-CAM per case. Where they agree, the offset reading is supported; where
  they do not, we say Grad-CAM was decorative for that sample.
* **The coordinate-attention gates are readable directly** (§7), so the Task-2b
  design claim is testable rather than rhetorical — and §7's last cell reports
  whether it held.

### Not established — stated plainly
1. **A Grad-CAM blob on a byte-plot is not a human-verifiable explanation.**
   There is no object to recognise. We present the maps as diagnostics only, and
   we make no claim that any highlighted region "is the malicious code".
2. **LIME's local surrogate fits poorly here** (see the R² table in §6). The
   image looks like an explanation, but a low R² means the linear model it comes
   from does not reproduce the CNN even locally. Reporting the picture without
   the R² would be misleading, so we report both.
3. **Quickshift segmentation is not meaningful on byte-plots.** We include it
   only to demonstrate that, not as a result.
4. **Attribution ≠ causation, and both ≠ correctness.** Even a stable, causal,
   family-consistent offset band does not prove the model learned malware
   semantics. It could be tracking a compiler artefact, a packer's stub, or the
   uploader's resize behaviour — all of which correlate with family in this
   corpus without being anything a defender would want relied upon.
5. **The dataset caps interpretability.** The uploader states that v1 images were
   resized to 256×256, so 65,536 pixels stand in for files of very different
   true lengths. A "12 % into the file" reading is therefore approximate, and for
   files far from 64 KB the row-to-offset mapping is only ordinal.
6. **Two samples are two samples.** The brief asks for one correct and one
   incorrect prediction and that is what §3–§7 show; the per-family aggregation
   in §4 is the only part of this notebook with enough samples to generalise.

### What we would do with more time
Recover the original binaries and check whether the attended offset bands
coincide with actual PE section boundaries. That would convert §4 from *"the
model consistently attends to 12–19 % of the file"* into *"the model attends to
the import table"*, which is the explanation everyone in this literature claims
to have and nobody, including us, has actually demonstrated on byte-plots.""")

co('''\
M.save_json(dict(
    case_A=dict(file=Path(df.path.iloc[G_A]).name,
                true=CLASS_NAMES[y_true[CASE_A]],
                pred=CLASS_NAMES[y_pred[CASE_A]],
                p_pred=float(y_prob[CASE_A, y_pred[CASE_A]]),
                top_bands=prof_store["A_correct"]["bands"][:3],
                gradcam_occlusion_r=occ_store["A_correct"]["agreement_r"]),
    case_B=dict(file=Path(df.path.iloc[G_B]).name,
                true=CLASS_NAMES[y_true[CASE_B]],
                pred=CLASS_NAMES[y_pred[CASE_B]],
                p_pred=float(y_prob[CASE_B, y_pred[CASE_B]]),
                top_bands=prof_store["B_wrong"]["bands"][:3],
                gradcam_occlusion_r=occ_store["B_wrong"]["agreement_r"]),
    lime=[dict(case=k[0], segmentation=k[1], surrogate_r2=v["r2"])
          for k, v in lime_store.items()],
    limitation="Grad-CAM/LIME on byte-plots have no semantically verifiable "
               "regions; see section 8.",
), CFG.art("task3_xai_summary.json"))

print("Task 3b artefacts:")
for p in sorted(Path(CFG.out_dir).rglob("*")):
    if p.is_file() and any(k in p.name for k in
                           ("gradcam", "lime", "occlusion", "coordatt",
                            "byte_offset", "offset_consistency", "xai")):
        print(f"  {p.stat().st_size/1024:9.1f} KB  {p.name}")
''')

build(C, REPO / "code" / "task3" / f"{PREFIX}_task3_explainability.ipynb",
      "CSE475 Task 3b - explainability")
print("task3b done")
