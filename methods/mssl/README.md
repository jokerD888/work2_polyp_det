# MSSL V1 — Mask-Supported Side Localization

**Status (2026-10-20):** All five existing models, mechanism analyses and the CIoU comparison are complete. Use the [frozen release index](../../docs/mssl_v1_release_20261020.md) for the actual training source bundles, configurations, checkpoint hashes and results. The working tree also contains later analysis/competitor code; it is not the original first-run source snapshot.

Training-only auxiliary localization for the pinned official MaskDINO R50.
No new learned parameters. Native decoder, matching, DN, mask losses, box
parameterization, initialization and inference are inherited unchanged.

## Exact V1 definition

For a matched final decoder prediction, convert native cxcywh to four edges.
On each axis, project the augmented GT mask into foreground mass m(u), with
pixel centres u=k+0.5. For the provided GT edge e* and corresponding GT
width/height s=max(size,4 pixels), define

```
pi_e(u) = softmax_u(log(m(u)) - abs((u-e*)/s)/0.15), for m(u)>0
F_e(z) = sum_u pi_e(u) * sigmoid(((z-e*)/s - (u-e*)/s)/0.05)
L_mssl = mean_e 0.5 * (F_e(pred_e)-F_e(e*))**2
L_total = L_native + 1.0 * L_mssl
```

GT-only support is detached; there is no predicted-mask teacher. GT Box
remains the coordinate target even if it differs from Mask2Box. The four-side
loss is averaged over eligible matched GT in each microbatch. Empty masks,
degenerate boxes and original multi-box images receive no extra loss. Native
losses still run. No additional loss is attached to encoder, DN or intermediate
decoder predictions. The final native matching is reused without another RNG
draw or Hungarian assignment.

The numbers above are explicit research initial values, **not official MSSL
settings or proven optima**. CDF saturation, asymmetric inward/outward response
and a possible reduction to ordinary mask-conditioned loss weighting remain
scientific limitations. Current Seed42 evidence supports a scoped AP gain on the observed PolypGen cohort; independent generalization and cross-seed stability remain unverified. See the release index for counterexamples and intervals.
The `support: box` counterfactual uses exact 1-D cell occupancy of the same
provided rectangle, with the same loss and directional weights.

## Shared data protocol

Original single-box training images use the original Kvasir semantic mask
(grayscale >127), without clipping it to the box, for BOTH native mask losses
and MSSL. Original multi-box images keep the existing native cropped-semantic
labels, and are excluded only from MSSL. Cropping cannot turn an original
multi-box image into an eligible single-box image. All supplied detection
boxes are preserved through geometric augmentation. Zero instance-mask padding
uses the previously fixed shared mapper.

This starts a **new control protocol**. Old runs using cropped masks or a
different resolution/schedule cannot serve as its paired baseline. A fresh
MSSL-disabled control shares the same raw-mask processing and accumulation.
Single semantic masks are a dataset adaptation, not independently annotated
instance masks. Used raw mask files are SHA256-recorded and verified again
before prediction. Source images and annotation files are never modified.

## Official settings and explicit adaptations

Pinned native config:
`configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml`
at MaskDINO commit `3831d8514a3728535ace8d4ecc7d28044c42dd14`.

Inherited: R50 ImageNet initialization, 300 queries, 9 decoder layers, all
native matching/loss weights, AdamW LR 1e-4, backbone multiplier 0.1, WD 0.05,
full-model clip norm 0.01, AMP, 1024 LSJ 0.1–2.0 and horizontal flip, RGB,
native inference/mask scoring. Test resize is the resolved Detectron2 default
800 short edge / max 1333, with TTA disabled.

Kvasir adaptations: one class, 880/120 train/val, 50 equivalent epochs =
2750 optimizer iterations at effective batch 16. The official LR decay
fractions 327778/368750 and 355092/368750 are mapped to 2444/2648 iterations.
Warmup factor 1.0 / 10 iterations is kept. Evaluation/checkpoint every 275
iterations, best **bbox AP50:95** on val, 100 detections, native bbox-only COCO
evaluation, seed 42, 2 loader workers. Val selection is not an independent test.

On this 12GB RTX 5070, microbatch 1 accumulates 16 times per optimizer update.
Loss division occurs before AMP backward, gradients are zeroed once per update,
and scaler/optimizer/LR scheduler advance once per optimizer iteration. Native
per-microbatch GT normalization and DN construction remain in place. This is
**not exact equivalence to a physical batch of 16**; method/control both use
the same adaptation. Automatic GradScaler overflow backoff and skipped updates
are recorded, alongside successful optimizer updates and per-side MSSL losses.

## Run

In the WSL model environment, from the workspace with native source on PYTHONPATH:

```bash
python -m methods.mssl train --config methods/mssl/configs/kvasir_official50.yaml --run-id NEW_ID
python -m methods.mssl predict --config methods/mssl/configs/kvasir_official50.yaml --run-id NEW_ID
# Matched control: include --baseline in BOTH train and predict.
```

Windows wrapper: `scripts/run_mssl_wsl.ps1` accepts the same arguments.

`python -m methods.mssl.launch --config ... --precheck PASSED_REPORT --output NEW_CAMPAIGN`
freezes a separate source bundle and starts a detached pipeline: 16-iteration
smoke train, checkpoint reload/full val prediction, numerical health gate,
formal MSSL train/predict, then fresh disabled-control train/predict. Any error
or frozen-source drift stops progression. This is a local background process,
not a recurring automation. Training records and snapshots are under `runs/`.

## Acceptance

- Nine focused tests: finite differences/sign, correct-coordinate zero with
  mask margin, mask increment/Box control, empty/ineligible data, option errors,
  exact native criterion/RNG parity with aux+encoder, GPU accumulation, and
  reading training metrics when the final JSON row contains evaluation only.
- Real GPU acceptance: 15 actual augmented records including original
  multi-box examples; all 105 native losses/RNG exact; nonzero gradient into
  final boxes and native bbox head; successful full AMP optimizer step;
  image-only inference identical with criterion removed.
- At 1024, the acceptance sample peaked at about 6144 MiB allocated memory.
  Initial native FP16 loss scaling overflowed and automatically backed off to
  64 before a finite disposable update. This is recorded, not hidden or counted
  as a successful update. Continuous smoke tests the actual accumulation path.

Detailed real-GPU evidence: `runs/mssl-implementation-20261018/precheck-3/report.json`.
Source controls and existing experiments are not modified by this module.
