"""Freeze independent smoke/formal configs without editing baseline or method.

Run in WSL/Linux after precheck review. Does not start any model or process.
"""
import argparse
import copy
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="runs/_launcher/scda-proposal-kvasir-s42")
    parser.add_argument("--smoke-iterations", type=int, default=50)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    control = (root / args.control).resolve()
    if not control.is_relative_to(root / "runs"):
        raise ValueError("Control directory must stay inside project runs")
    control.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load((root / "configs/maskdino_kvasir_official.yaml").read_text(encoding="utf-8-sig"))
    base["project_root"] = os.path.relpath(root, control).replace("\\", "/")
    method = yaml.safe_load((root / "methods/scda_proposal/configs/kvasir.yaml").read_text(encoding="utf-8-sig"))
    formal = copy.deepcopy(method)
    formal["base_experiment"] = "formal_base.yaml"
    smoke_base = copy.deepcopy(base)
    smoke_base["name"] += "_scda_smoke"
    smoke_base["purpose"] = "gpu_preflight_only"
    smoke_base.pop("training_budget", None)
    smoke_base["overrides"].update({"SOLVER.MAX_ITER": args.smoke_iterations,
                                     "SOLVER.CHECKPOINT_PERIOD": args.smoke_iterations,
                                     "TEST.EVAL_PERIOD": args.smoke_iterations})
    # Keep initial warmup/LR and original milestones: first 50 iterations match
    # the beginning of the formal schedule, rather than accelerating its LR.
    smoke = copy.deepcopy(method)
    smoke["name"] += "_smoke"
    smoke["base_experiment"] = "smoke_base.yaml"
    smoke["scda"]["diagnostics"] = True
    files = {"formal_base.yaml": base, "formal.yaml": formal,
             "smoke_base.yaml": smoke_base, "smoke.yaml": smoke}
    for name, value in files.items():
        path = control / name
        if path.exists():
            raise FileExistsError(f"Refuse to overwrite frozen configuration: {path}")
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    print(control)


if __name__ == "__main__":
    main()
