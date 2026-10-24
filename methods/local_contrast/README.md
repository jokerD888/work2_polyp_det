# L1/L2: MSSL V1 + local foreground/background contrast

First paired experiment, fixed on 2026-10-24. This is an implementation of a
standard supervised region contrast objective, not a claim of a new contrast
learning principle or of proven candidate recovery. No AP has informed its
initial hyperparameters.

L1 (`box`) and L2 (`mask_priority`) both retain MSSL V1 and train from the
original ImageNet R50 initialization. The feature is the actual 256-channel
stride8 Encoder output passed into the detection predictor. A read-only
pre-hook captures that tensor. No projection head, learned parameters,
replacement features, detach or inference-time module is introduced.

## Fixed loss

For each eligible instance, sample unique foreground/background grid cells
without replacement, at most 32 each and at least 2 each. Normalize the feature
vectors to unit length (epsilon 1e-6), use cosine logits divided by temperature
0.1. Every anchor has all other same-class points as positives and all other
points in the denominator. Exclude self. Average the positive log probabilities
for each anchor, the anchors within each class, the two classes equally, and
eligible instances equally. The auxiliary weight is 0.1. Compute logits and
logsumexp in FP32 even under AMP. Empty eligibility returns a graph-connected
zero. Total loss is native MaskDINO + unchanged weight1.0 MSSL V1 + 0.1 contrast.

These values are research starting values, not an optimized setting. Both
conditions use exactly these values; actual losses and gradients may differ.

## Paired geometry and fallback

Construct pixel regions after the existing 1024 LSJ augmentation. For a region
with area A, erode with square radius ceil(.05 sqrt(A)), and dilate with square
radius ceil(.20 sqrt(A)); both radii are at least 1 pixel. The foreground is the
eroded core. The local background is the dilation minus the original region.
Square filters mean a Chebyshev margin; this is explicitly a new final region
definition, not identical to the earlier feasibility audit's distance geometry.

Box pixels use pixel-center containment in the transformed supplied xyxy box.
Mask pixels use the original single-box semantic mask after the exact same
augmentation. Original multi-box images remain unreliable for instance Mask,
even if cropping leaves one box. Native boxes, native masks, Hungarian matching,
DN and MSSL eligibility are unchanged.

Exclude padding and other provided boxes from foreground. Exclude padding,
the complete semantic foreground union, and ALL provided boxes from background.
The full union is loaded even on multi-box images, transformed by the same LSJ
object, and used only as a background exclusion; it does not create instance
masks or change native supervision. Grid eligibility requires at least .95
pixel occupancy in the selected region and in the valid image, in exact
nonoverlapping stride8 cells. This is a spatial purity rule, not a guarantee
about receptive-field purity.

For every instance, BOTH conditions calculate the Box pools and, when reliable,
the Mask pools:

1. If Box supports fewer than 2 points in either class, both skip the instance.
2. If reliable Mask also supports at least 2 in each class, L1 uses Box and L2
   uses Mask. Each class's shared budget is min(32, Box count, Mask count).
3. If no reliable Mask exists OR Mask support is insufficient, both use the
   identical Box pools, with min(32, Box count) for each class.

The point budget may differ between foreground and background; the loss weights
the classes equally. Log actual budgets and shortfalls. Never repeat a position
to fill a budget. A private NumPy generator is seeded by SHA256(seed, image ID,
microbatch sequence, instance index); neither Torch nor global NumPy/Python RNG
is consumed. Box fallback samples are identical in the paired runs.

## Verification and execution

CPU tests cover explicit loss math, gradients, separation, class counts,
uniqueness, private RNG, semantic/padding exclusion, unchanged labels, shared
fallback and tiny-target skipping. Real GPU engineering checks compare original
MSSL mapping, every native predictor tensor and CPU/CUDA RNG, native scalar
losses, actual shared-feature/backbone gradients, optimizer discovery and
image-only inference. Prediction comparisons are exact; CUDA scalar reductions
allow relative 2*FP32 epsilon and zero absolute tolerance, alongside independent
native self-repeat measurements. No optimizer step or AP is used in that gate.

The finite campaign freezes source/config/input hashes, runs regression tests,
then disposable off/L1/L2 cold smokes with 20 successful AMP accumulated updates
each (40 attempts maximum). It compares image/augmentation prefixes, native
initializations, finite losses, exact model/Adam moments/scheduler/scaler reload
and image-only predictions. Formal DefaultTrainer startup is separately checked.
All formal training starts fresh from the original asset, never Smoke weights.

Formal protocol: Seed42, microbatch1 x accumulation16, 2750 planned optimizer
iterations, native 1024 LSJ, unchanged optimizer/schedule/matching/native losses.
Select maximum Kvasir Val bbox AP every275 iterations, earliest tie. Record AMP
skips and successful updates. No intermediate AP tuning. Inference is unchanged.

After BOTH full trainings and source checks, verify all target image hashes,
run each source-selected model once on the existing 1473-frame PolypGen cohort,
and report AP/AP50/AP75/AP90/AP95, matched misses/FP, center, single/multi-object
and relative-area groups. Main metric is AP50:95; moderate strict-IoU decreases
are assessed jointly with the main gain and detection tradeoffs. No automatic
method adoption, multi-seed expansion or additional ablations. PolypsSet/PolypDB
remain reserved; Copy-Paste is not part of this experiment.

Campaign status: `runs/local-contrast-full-20261024/state.json` and `queue.json`.
Engineering receipt: `runs/local-contrast-acceptance-20261024-r2/verification.json`.

```bash
python -m methods.local_contrast.campaign launch \
  --acceptance runs/local-contrast-acceptance-20261024-r2 \
  --output runs/local-contrast-full-20261024
```
