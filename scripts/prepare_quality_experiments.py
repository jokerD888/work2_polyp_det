"""Freeze new configs from the actual completed F-Reg-Quality experiment.

No subprocess, model construction, environment changes, calibration or training.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.scda_proposal.config import SCDAOptions
from polypbench.config import MASKDINO_COMMIT
from polypbench.io import read_json, sha256, write_json


def prepare(reference_path, output):
    reference_path, output = Path(reference_path).resolve(), Path(output).resolve()
    reference = read_json(reference_path)
    if Path(reference["project_root"]).resolve() != ROOT:
        raise ValueError("Use this machine's real reference experiment.json; do not transplant absolute paths")
    options = SCDAOptions(**reference["scda"]).validate()
    if (not options.enabled or options.reg_selection != "quality_iou" or
            options.lambda_cls != 0 or options.lambda_box != .05 or
            reference["repo"]["commit"] != MASKDINO_COMMIT):
        raise ValueError("Reference must be the completed F-Reg-Quality definition")
    overrides = reference["overrides"]
    expected = {"SOLVER.IMS_PER_BATCH": 16, "SOLVER.MAX_ITER": 5500,
                "SOLVER.BASE_LR": .0004, "SOLVER.WARMUP_ITERS": 125,
                "SOLVER.STEPS": [4400, 4950], "SOLVER.CHECKPOINT_PERIOD": 275,
                "TEST.EVAL_PERIOD": 275}
    if (any(overrides.get(k) != v for k, v in expected.items()) or
            reference["training_budget"] != {"train_images": 880, "iterations_per_epoch": 55, "epochs": 100} or
            reference["checkpoint_selection"] != "best_val_ap" or reference["seed"] != 42):
        raise ValueError("Reference differs from the confirmed seed42/B16/100epoch protocol")
    # A method's resolved model.config may be absolute; the baseline parser
    # accepts absolute paths as well. Keep exact data/model/worker settings.
    fields = ("project_root", "purpose", "seed", "checkpoint_selection", "checkpoint_keep",
              "training_budget", "repo", "runtime", "dataset", "model", "overrides",
              "evaluation", "output_root")
    base = {key: copy.deepcopy(reference[key]) for key in fields if key in reference}
    base.update(name="maskdino_quality_reference_b16", adapter="maskdino")
    base["runtime"]["num_gpus"] = 1
    base["runtime"]["visible_gpus"] = "0"
    reference_cfg = {"base_experiment": "reference_base.yaml", "name": "scda_freg_quality_reference_b16",
                     "scda": asdict(options)}
    reference_cfg["scda"]["diagnostics"] = False
    adapter = yaml.safe_load((ROOT / "methods/proposal_dcnv4/configs/dcn_freg_quality.yaml").read_text())
    adapter["base_experiment"] = "reference_quality.yaml"
    box_base = copy.deepcopy(base)
    box_base["runtime"]["visible_gpus"] = "1"
    box_cfg = yaml.safe_load((ROOT / "methods/scda_proposal/configs/kvasir_box_gated_iou.yaml").read_text())
    box_cfg["base_experiment"] = "box_base.yaml"
    box_cfg["box_gate_calibration"] = "calibration.json"
    files = {"reference_base.yaml": base, "reference_quality.yaml": reference_cfg,
             "dcn_quality.yaml": adapter, "box_base.yaml": box_base, "box_gated_iou.yaml": box_cfg}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in files.items():
        (output / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    write_json(output / "preparation.json", {"reference": str(reference_path),
               "reference_sha256": sha256(reference_path), "training_started": False,
               "calibration_started": False, "gpu0": "F-Reg-Quality + DCNv4",
               "gpu1": "Box-Gated IoU; requires calibration.json first",
               "files_sha256": {name: sha256(output / name) for name in files}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-experiment", required=True)
    parser.add_argument("--output", default="runs/_launcher/quality-boxgate-a6000-s42")
    args = parser.parse_args()
    print(prepare(args.reference_experiment, args.output))


if __name__ == "__main__":
    main()
