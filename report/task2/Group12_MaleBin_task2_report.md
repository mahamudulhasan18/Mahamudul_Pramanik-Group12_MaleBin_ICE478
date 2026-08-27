# Task 2 Report — Baselines & Proposed Model

**Group00 · CSE475 Summer 2026 · Track 3 (CNN + Attention) · Dataset: MaleBin**

> Fill every `⟨…⟩` from your Kaggle runs of
> `task2_baselines.ipynb` and `task2_proposed_model.ipynb`.
> Sections 4–7 are **Pillar B** of the grade (brief §6.5, §8) — they must be in
> your own words. Export to `report/task2/Group00_MaleBin_task2_report.pdf`.

---

## 1. Preprocessing and protocol

| Step | What we do | Why |
|---|---|---|
| Split | `StratifiedGroupKFold` on the duplicate-group key from Task 1 §4, stratified on family. 20 % test, 15 % of the remainder as validation | Brief §6.2 requires a source-based split; MaleBin has no source column, so the duplicate group *is* our source proxy |
| Features | Pixels only | File size, path, folder name and PNG compression ratio are metadata. Using them is the image-domain equivalent of the packet-ID leak §6.2 forbids |
| Resize | 256 → 224, bilinear | Matches the pretrained backbones' expected input; applied identically to every model |
| Normalise | **Per-image** standardisation (subtract that image's mean, divide by its std) | "How bright is this file overall" is the average byte value — a nuisance factor that changes with padding and packing. Standardising per image keeps the texture, which is the signal |
| Channels | 1 for from-scratch models; grayscale repeated to 3 for ImageNet backbones | The standard protocol in this literature (PAFE, IMCEC, Alshomrani et al.), so the comparison stays fair |
| Imbalance | Inverse-frequency class weights in the loss; judged by macro-F1 + per-class recall | Brief §6.2 |
| Augmentation | Byte-aware: vertical roll ±12 %, row crop 85–100 %, random erasing. **No flips, no rotations** | See §5 |
| Selection | Early stopping on **validation macro-F1** | It is the metric we are graded on, so it is the metric we optimise |

Every model uses the same split, input size, optimiser family, schedule, epoch
budget, early-stopping criterion and augmentation. Anything else would make the
comparison meaningless.

## 2. What the honest split costs — measured

Same model, same hyper-parameters, same epochs; only the split protocol changes.

| Protocol | Accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| Random stratified (what every related-work paper does) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| **Duplicate-grouped stratified (ours)** | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

Inflation from splitting randomly: **⟨+…⟩ macro-F1 (⟨…⟩ % relative)**.

This one number is why our headline sits below the published 99 %+ figures. It is
a protocol difference, not a weaker model, and we keep it visible in every
report.

## 3. Baselines

| Model | Why it is here | Params | macro-F1 | Accuracy | weighted-F1 | macro-recall | AUC (OvR) | Train (s) | Inference (ms/img) |
|---|---|---|---|---|---|---|---|---|---|
| SimpleCNN (scratch, 1-ch, no attention) | The like-for-like control: isolates our architecture from ImageNet weights | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| ResNet50 (ImageNet) | The default backbone in this literature | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| DenseNet121 (ImageNet) | Dense connectivity reuses low-level texture; a top performer in Jayasudha et al. (2023) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| EfficientNet-B0 (ImageNet) | Comparable parameter budget to ours, so "we won by being bigger" is unavailable | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

**Baseline to beat: ⟨…⟩, macro-F1 ⟨…⟩.** This is the model Task 3's Wilcoxon and
McNemar tests are run against.

⟨Insert `task2_baseline_comparison.png`, `task2_baseline_curves.png`, the
confusion matrix / ROC / PR figures, and `task2_perclass_recall.png`.⟩

**Observations.**
* Accuracy exceeds macro-F1 by ⟨…⟩ on average — the imbalance premium.
* Families where **every** baseline stays under 0.60 recall: ⟨…⟩. These are the
  Task-3 target.
* Parameters vs macro-F1 shows ⟨no / a weak⟩ monotone relationship — bigger is
  not better here. ⟨If SimpleCNN is competitive with ResNet50, note it: that is
  direct evidence the ImageNet prior does not transfer to byte-plots, which is
  the premise of §4–§5.⟩

## 4. ByteAttnNet — Pillar-B Q1: how it works, step by step

⟨Insert `task2_architecture.png`.⟩

**Input.** 256×256 grayscale byte-plot → 224×224, standardised per image.

**Stem.** One 3×3 convolution, stride 2, → BatchNorm → ReLU, giving 48 channels
at 112×112. Stride 2 immediately, because the uploader already resized these
images, so single-pixel structure is partly a resampling artefact — and it halves
the cost of every later layer.

**Four stages,** each `multi-scale conv block(s) → attention → pool`:

*Multi-scale conv block* — three convolutions on the same input in parallel, then
concatenated and fused by a 1×1 convolution, plus a residual shortcut:

| branch | kernel | dilation | effective view | what it sees |
|---|---|---|---|---|
| 1 | 3×3 | 1 | 3×3 | fine byte texture (opcode-level n-grams) |
| 2 | 3×3 | 2 | 5×5 | basic-block / function-level repetition |
| 3 | 3×3 | 3 | 7×7 | section-level layout |

Dilation rather than literally larger kernels: a real 7×7 convolution has 5.4×
the parameters of a 3×3.

*Attention* — CBAM then coordinate attention:

1. **CBAM channel gate.** Squeeze each feature map twice (average and max
   pooling), push both through a shared bottleneck MLP, add, sigmoid → one weight
   per channel: *which texture detectors does this file need?* Both pools are
   used because average says how widespread a texture is while max says whether
   it occurs at all — for a small packed section only max fires.
2. **CBAM spatial gate.** Collapse channels to per-pixel max and mean,
   concatenate, one 7×7 convolution, sigmoid → one weight per location: *which
   regions matter?*
3. **Coordinate attention.** Pool along **one axis at a time**: average over
   columns → a vector of length *H* (one value per row); average over rows → a
   vector of length *W*. Concatenate, one shared 1×1 conv + BN + hardswish, split
   back, produce a per-row gate and a per-column gate, multiply by both. Because
   pooling never collapses both axes at once, **position survives** — and a row is
   a byte-offset band.

2×2 max pooling after stages 1–3: 112 → 56 → 28 → 14, channels 48 → 96 → 192 → 320.

**Readout.** GeM pooling: `(mean(x^p))^(1/p)` with *p* learnable, clamped to
[1, 8]. *p* = 1 is average pooling, *p* → ∞ is max pooling, so the network
chooses where to sit instead of us guessing. Our run learned **p = ⟨…⟩**, meaning
family identity lives in ⟨the average texture energy / a few peak blocks⟩.

**Head.** Dropout(0.3) → one linear layer to 39 logits. No hidden FC layer: with
~2.5 M convolutional parameters on an effective *n* of a few thousand distinct
binaries, a wide FC head is where memorisation starts.

**Loss / optimisation.** Class-weighted cross-entropy, label smoothing 0.05,
AdamW, OneCycle LR, early stopping on validation macro-F1.

Total: **⟨2,487,242⟩ parameters, ⟨9.97⟩ MB** — roughly 10× smaller than ResNet50.

## 5. Pillar-B Q2: why each choice is there

| Choice | Justification |
|---|---|
| Trained from scratch, 1 input channel | Task 1 §3: byte-plots have no oriented edges, objects or colour opponency. An ImageNet prior is actively wrong and must be unlearned; replicating grayscale to 3 channels also triples the stem's FLOPs for zero information |
| Multi-scale dilated conv | Task 1 §5: signal exists at byte level (`entropy`, `grad_mag`), block level (`row_autocorr`) and section level (`row_var`). One kernel size must compromise. PAFE (2024) and IMCMK-CNN (2024) both report gains from multi-scale kernels; dilation gets the same receptive fields at 3×3 cost |
| CBAM channel attention | Different families are identified by different textures; a per-channel gate lets one network specialise per family without extra capacity. This is the mechanism PAFE's FFSE and IMCMK-CNN's improved-SE use |
| CBAM spatial attention | Task 1 §3: the informative region is a *band* whose location varies by family; the rest is padding. A spatial gate suppresses padding |
| **Coordinate attention** | The gap from Task 1 §7. Row index = byte offset. SE and CBAM's channel branch global-pool, destroying it. Swin keeps 2-D position but treats it as a generic image coordinate at far higher cost. CoordAtt (Hou et al., CVPR 2021) gives a per-row and per-column gate for two 1×1 convolutions, so the network can learn "for family *X*, look 15–25 % into the file" |
| No flips, no rotations | Flipping reverses byte order; rotating transposes offset into row-width. Both produce inputs that cannot exist. Task 3 ablates this against the naive recipe |
| Byte-aware augmentation instead | What legitimately varies between variants of one binary: where a section starts (→ vertical roll), file length (→ row crop), a region repacked (→ random erasing). Each transform corresponds to a real polymorphic operation |
| GeM pooling | Classes differ in texture *energy*; whether average or peak matters is empirical, so we make it learnable |
| Inverse-frequency class weights | Task 1 §2 imbalance + §6.2's macro-F1 rule. Basak et al. (2024) report their model "struggles with underrepresented classes"; weighting is the cheapest direct fix |
| ~2.5 M parameters | Effective *n* is the number of distinct binaries (Task 1 §4), far below the image count. A 23 M-parameter ResNet50 on that is asking for memorisation |

## 6. Pillar-B Q3: why it should beat the baselines

Four falsifiable claims, each with the Task-3 experiment that tests it:

1. **The pretrained baselines carry the wrong prior.** ResNet50 / DenseNet121 /
   EfficientNet-B0 start from ImageNet filters — Gabor-like edges and colour
   blobs — and must spend capacity unlearning them, with 10–25 M parameters on a
   small effective *n*. *Test:* if SimpleCNN is already competitive in §3, the
   prior is confirmed unhelpful. ⟨Our §3 result: …⟩
2. **Nobody models byte offset, and byte offset is real information.** PE section
   layout is family-characteristic. CoordAtt is the only module here that can
   represent "attend to this offset band". *Test:* the `attention = coord` vs `se`
   vs `none` ablation, plus the Task-3b row-band occlusion check.
3. **Multi-scale beats single-scale on texture.** *Test:* ablation
   `multiscale = False` at matched parameter count.
4. **Byte-aware augmentation beats the natural-image recipe.** *Test:* ablation
   `aug = byte` vs `naive` vs `none`.

**Where we expect to lose, stated in advance.** Against the *published* Malimg
numbers (99.2–99.4 %) we expect to come in lower, because those all use random
splits and we do not — §2 measured the gap. Our claim is only that we beat *our
own* baselines under an identical honest protocol, and Task 3's significance tests
examine exactly that.

## 7. First results and Pillar-B Q4: honest reading

| Model | Params | macro-F1 | Accuracy | weighted-F1 | macro-recall |
|---|---|---|---|---|---|
| Best baseline (⟨…⟩) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| **ByteAttnNet v1** | ⟨2.49 M⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| Difference | ⟨…⟩× fewer | ⟨+…⟩ | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |

⟨Insert `task2_first_results.png`, `task2_proposed_cm.png`,
`task2_perclass_delta.png`.⟩

Write here, plainly:

* **Did it beat the best baseline on macro-F1, by how much, and at what parameter
  ratio?** Then state: this is **one split**, and whether the difference is real
  is decided in Task 3 by 5-fold CV + Wilcoxon + McNemar, not here.
* **Which families improved and which got worse.** Use the per-class delta plot.
  If a rare family got worse, say so and say why — usually class weighting traded
  majority precision for minority recall.
* **What the learned GeM exponent tells you** (p = ⟨…⟩).
* **What is still unproven.** Every attention claim: CBAM and CoordAtt are
  switched on together here, so §6's claims 2–4 are untested. Task 3's ablation is
  what turns "we designed it this way because…" into evidence, and it is entirely
  possible CoordAtt contributes nothing. Say that now; the rubric rewards an
  honest negative result over a mysterious win.
* **The published-number gap** and its measured cause (§2), plus the dataset's own
  limitations (outdated Malimg samples, resize distortion).
