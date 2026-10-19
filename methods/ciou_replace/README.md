# CIoU-Replace

**Status (2026-10-20):** The 13-stage campaign is complete. The [frozen release index](../../docs/mssl_v1_release_20261020.md) is the current entry point. The launch command below documents the historical execution interface; rerunning the completed campaign is unnecessary.

One prespecified mature localization competitor for the completed MSSL study.
All native enabled box-regression GIoU slots become pinned Detectron2 CIoU;
Hungarian GIoU costs, L1, masks, normalization and inference remain native.
No MSSL auxiliary loss, extra trainable parameters, target tuning or new datasets.

Training reuses `methods.mssl.worker` and its mapper/AMP accumulation. The optional
`box_regression` contract selects an independent model/criterion registry entry;
old configurations without it retain their original behavior. Frozen historical
source bundles and four trained checkpoints are never rewritten.

From the project under the existing WSL environment:

```bash
python -m methods.ciou_replace.campaign launch --output runs/ciou-competition-20261019
```

The detached finite pipeline freezes source, original source/target inputs and
the Detectron2 loss SHA256, then runs unit/regression tests, all-branch real GPU
acceptance, 50-iteration smoke/reload, one Seed42/2750-iteration training run,
source prediction, checkpoint gradient replay, fixed1473 PolypGen inference and
five-group COCO/bootstrap comparison. It reuses the exact prior 2000 draws and
checks the four original models' vectors against their archived results.

`state.json` tracks stage/PIDs/completion; failed checks stop subsequent stages.
Logs and final `summary.md` stay in the campaign directory. This is local WSL:
the computer and WSL must remain running; no recurring automation is involved.

Audit detail: with native INITIAL_PRED and mask-enhanced initialization,
`loss_giou_0` and `loss_giou_dn_0` concern detached initialization coordinates.
Both native GIoU and CIoU have zero gradients to bbox parameters there; this is
reported, not circumvented by changing native detach behavior. All learned
regression branches must show finite actual bbox gradients on the audit panel.
Loss keys retain `loss_giou*` only for native suffix/weight compatibility; their
replacement semantics are explicitly recorded in metadata.

Primary protocol: `docs/mssl_ciou_next_stage_plan_20261019.md`.
