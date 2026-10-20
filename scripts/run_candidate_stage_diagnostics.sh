#!/usr/bin/env bash
# Run only on the training machine. Finite inference queue; never train a model.
set -euo pipefail
if [[ "${1:-}" != "--training-machine" ]]; then
  echo "Usage on training machine: POLYPGEN_ROOT=/path/to/data bash scripts/run_candidate_stage_diagnostics.sh --training-machine" >&2
  exit 2
fi
: "${POLYPGEN_ROOT:?Set POLYPGEN_ROOT to the directory containing data_C1 through data_C6}"
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${POLYPBENCH_PYTHON:-python}"
task_annotations="${POLYPGEN_ANNOTATIONS:-$task_root/data/polypgen_zeroshot_stills_v2/provided_boxes.coco.json}"
task_assets="${CANDIDATE_ASSETS_ROOT:-$task_root/runs}"
task_output="${CANDIDATE_STAGE_OUTPUT:-$task_root/runs/candidate-stage-remote-$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$task_root"
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$task_python" -m polypbench.analysis.candidate_stage_jobs prepare \
  --bundle "$task_root/docs/protocols/candidate_stage_20261021/bundle" \
  --workspace "$task_root" --assets-root "$task_assets" \
  --image-root "$POLYPGEN_ROOT" --annotations "$task_annotations" --output "$task_output"
"$task_python" -m polypbench.analysis.candidate_stage_jobs run --queue-root "$task_output"
echo "Completed. Return this whole directory: $task_output"
