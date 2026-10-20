# Fixed MSSL E/F experiment

E adds `2 Lf`; F adds `Lf + La`. `La` is the last ordinary auxiliary decoder output,
with that layer's existing Hungarian assignment. Each term preserves V1's four-side
mean and eligible-object normalization. Outer `loss_mssl` weight stays exactly one.
La is measured in E but receives zero objective coefficient. DN and encoder outputs
cannot enter either route. Native losses and inference are unchanged.

Both configurations are frozen before training. Both real-data GPU prechecks and both
50-iteration smoke gates precede either 2750-iteration, Seed42, ImageNet-initialized
formal run. Source-Val best AP selects checkpoints; PolypGen is a fixed external
development review after both trainings. No intermediate AP tuning or automatic grid.

Sparse shared-bbox-head diagnostics report gf, ga, their actual aggregate and native
final/all regression gradients. A common, measurement-only scale backs off on FP16
overflow without changing GradScaler, RNG, parameters, or `.grad`. Zero-norm cosines
are null. Floating point head-linearity residuals are diagnostic; exact coefficient
and coordinate-gradient algebra is checked separately. Bigger gradients are not a
success criterion. Equal nominal weights do not equal actual gradient strengths.

Run `python -m methods.mssl_scope.campaign launch --output ABSOLUTE_NEW_DIRECTORY`.
The detached finite queue stores protocol/source hashes, queue/state JSON, stage logs,
all acceptance reports, fixed evaluations and summary.md. Failure stops
subsequent work. E/F are optimization of one mechanism, not a second innovation;
if neither provides convincing overall AP benefit, retain V1 and end this search.
