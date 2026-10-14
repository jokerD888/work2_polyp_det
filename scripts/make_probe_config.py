"""Make a short GPU preflight from a formal configuration without editing it."""
import argparse
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--iterations", type=int, default=4)
args = parser.parse_args()
if args.iterations < 2:
    parser.error("At least two iterations are required")
source = Path(args.config).resolve()
target = Path(args.output).resolve()
if target.exists():
    parser.error("Probe config already exists; choose a fresh output")
cfg = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
cfg["project_root"] = str((source.parent / cfg["project_root"]).resolve()).replace("\\", "/")
# WSL sees the same file through /mnt/<drive>; retain relative project root when possible.
import os
cfg["project_root"] = os.path.relpath(cfg["project_root"], target.parent).replace("\\", "/")
cfg["name"] += "_probe"
cfg["purpose"] = "gpu_preflight_only"
cfg.pop("training_budget", None)
cfg["overrides"].update({"SOLVER.MAX_ITER": args.iterations, "SOLVER.STEPS": [args.iterations - 1], "SOLVER.CHECKPOINT_PERIOD": args.iterations, "TEST.EVAL_PERIOD": max(1, args.iterations // 2)})
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(target)
