# Fixed support-reference experiments

The opt-in `support_reference: {variant: ...}` field selects `native`,
`blend05`, `geometry`, or `support`. Existing MSSL configurations are unchanged.
The geometry and support variants store 5,444 new parameters; geometry zeros
the 64 profile inputs and keeps the same 16 scalar inputs. Equal stored
parameters do not imply equal effective capacity. Blend05 uses the native
normalized-cxcywh 0.5 mean before inverse-sigmoid, with no new parameters.

All variants preserve Top300 selection/content, DN reference construction,
native loss definitions/matching rules/scoring and the original data protocol. The adapter
uses pre-Decoder predictions only. No PolypGen GT trains or selects its rule.

## Finite execution

In the native detector environment, from the project root:

```powershell
python -m methods.support_reference.smoke_campaign launch --output runs/reference-smoke
python -m methods.support_reference.preflight --smoke runs/reference-smoke --output runs/reference-external-preflight
python -m methods.support_reference.campaign launch --smoke runs/reference-smoke --preflight runs/reference-external-preflight --output runs/reference-full
```

Each command requires the preceding verification to pass. Launchers detach
finite controllers and freeze their Python/config sources. State, child PID,
task queue, individual stage logs and verification receipts live under the
specified output. An engineering failure stops the controller; it does not
silently change configuration, resume from smoke, or launch another variant.

The smoke executes four conditions independently from the recorded original
ImageNet R50 asset: 20 **successful** AMP optimizer updates each, cap40
attempts, microbatch1 and accumulation16. AMP-skipped attempts do not count
as updates. All four must share native initial parameters and the consumed
image/box augmentation prefix. New parameters must be discovered once by
the optimizer, receive finite nonzero gradients beyond the zero last-layer
initialization, and update. Strict model/optimizer/scheduler/scaler checkpoint
replay and image prediction parity are required. No AP is evaluated.

The collector preflight compares observed versus plain predictions on two
Kvasir Train images per smoke model, without optimization or AP. Full runs
then repeat 19 CPU and 15 existing MSSL regression checks and instantiate
the actual formal Trainer for all three variants. The native parameter hash
and optimizer discovery are checked **before** any formal update.

Only three independent Baseline full runs are queued: blend05, geometry,
support. Each starts fresh from the original pretrained asset, Seed42,
2,750 planned optimizer iterations, original learning-rate schedule,
microbatch1/accumulation16, 1,024 LSJ training and Kvasir Val checkpoint
selection every275 iterations, earliest maximum-AP tie. Smoke checkpoints
are engineering evidence and never warm-start full training.

After all three trainings, strict selected-checkpoint inference is evaluated
once on the fixed 1,473-image PolypGen development cohort. Small per-query
geometry exports retain Encoder boxes, native Mask boxes, actual references,
final300 boxes, Top100 IDs and scores. GT enters only CPU scoring. Reports
include AP50:95/50/75/90/95, one-to-one and marginal coverage, marginal GT
gains/losses versus the original native model, center/area/multiplicity
groups, duplicate/localization/background FP and negative-frame FP.
GT joins use annotation IDs, not incidental annotation order. Separately
trained queries are not treated as shared query identities.

The queue ends after comparison; MSSL combinations, parameter searches,
PolypsSet and PolypDB are not queued. A successful smoke establishes runtime
feasibility, not a second innovation or a performance improvement. The formal
support result must be judged against the ordinary geometry control and the
matched-training blend control before considering MSSL complementarity.

## Individual formal entry point

```powershell
python -m methods.support_reference train --config path/to/resolved.yaml --run-id unique-run-id
python -m methods.support_reference predict --config path/to/resolved.yaml --run-id unique-run-id
```

Dataset/runtime/native-repository paths must point to the intended host.
The local campaign launcher reuses this project's archived baseline protocol;
its `runs/` references must exist. Inference reload is strict, including the
adapter keys. An ImageNet backbone initialization is intentionally partial;
its asset hash and post-load native parameter hash are recorded separately.
