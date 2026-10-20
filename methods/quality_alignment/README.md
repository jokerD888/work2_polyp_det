# IA-BCE-Final

Fixed transfer of AlignDETR `IA_BCE_loss` (commit
`4b125f7fe2ff8c3468cd71763fcf971d09ee38db`) in its one-to-one,
rank-weight-one form. This is a component experiment, not a new quality head
or a reproduction of the full AlignDETR method.

Only final ordinary decoder classification changes. Positive targets are
`max(p**.25 * IoU**.75, .01).detach()`; unmatched class entries use
`p**2 * softplus(logit)` with differentiable weights. Stable FP32 soft BCE
replaces the author's direct logs. Native normalization and external weight4
remain; aux, encoder, DN, matching, box and mask losses remain native.
MSSL+Q keeps the frozen V1 mask support loss. No prediction parameters are added.

`campaign launch --output ABSOLUTE_NEW_DIRECTORY` freezes both configurations
and all executable source, then detaches a finite queue. Both real GPU
prechecks and both 50-iteration smokes must pass before either 2750-iteration
formal run. All new runs start from the common initialization. Failures stop
the queue and preserve logs. No branch depends on interim AP.

The queue includes source checkpoint selection, full source prediction,
best-checkpoint gradient replays, the predefined 1473-frame PolypGen evaluation,
the original source quality diagnosis, and the four-group comparison. The old
paired bootstrap draws are reused, and historical A/B vectors are checked.
`protocol.json`, `queue.json`, `state.json`, per-stage logs and final `summary.md`
provide the audit trail. No PolypsSet/PolypDB evaluation or automatic new method.

The same external classification weight does not equate gradient scales.
Soft targets, positive/negative weighting and focal shape change together.
Sparse gradient probes start at diagnostic scale64 with predetermined
overflow backoff (64, 8, 1, 1/8, 1/64; unscaled before norms),
do not write parameter gradients or consume RNG, and do not tune the loss.

`reference/align_losses.py` retains the exact upstream Apache-2.0 source and
copyright. Tests AST-load its two functions with the equivalent torchvision
box-IoU interface, checking values and gradients separately from numerical
stabilization at saturated logits.

For the explicitly paused MSSL+Q campaign, `python -m
methods.quality_alignment.resume launch --output QUEUE_ROOT` continues the
existing finite queue using the saved iteration549 checkpoint. It validates
the original inputs and freezes a separate recovery controller/checkpoint
archive; the original training/loss source snapshot stays unchanged. Model,
optimizer, scheduler, AMP scale, update count and historical best selection
are restored, and the native data cursor is advanced to iteration550. Completed
experiments are skipped. Original RNG/worker state was not checkpointed, so
this continuation does not claim bitwise uninterrupted-training equivalence.
