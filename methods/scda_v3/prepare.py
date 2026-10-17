"""Prepare machine-specific V3 configs from the completed low-LR reference.

Only creates a NEW configuration directory. No model, subprocess or training.
"""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import yaml

from polypbench.io import read_json, sha256, write_json
from .config import V3Options
from .experiment import load_experiment


def prepare(reference_path, output):
    root = Path(__file__).resolve().parents[2]
    reference_path, output = Path(reference_path).resolve(), Path(output).resolve()
    reference = read_json(reference_path)
    if Path(reference["project_root"]).resolve() != root:
        raise ValueError("Use this machine's real completed reference, not transplanted absolute paths")
    scda = reference.get("scda", {})
    if (scda.get("reg_selection") != "quality_iou" or scda.get("lambda_cls") != 0 or
            scda.get("lambda_box") != .05 or not scda.get("enabled")):
        raise ValueError("Reference must be F-Reg-Quality, without DCNv4")
    if reference.get("dcnv4", {}).get("enabled"):
        raise ValueError("Do not inherit a DCNv4 combination for V3")
    fields = ("project_root", "purpose", "seed", "checkpoint_selection", "checkpoint_keep",
              "training_budget", "repo", "runtime", "dataset", "model", "overrides", "evaluation", "output_root")
    base = {key: copy.deepcopy(reference[key]) for key in fields if key in reference}
    base.update(adapter="maskdino", name="v3_low_lr_reference")
    output.mkdir(parents=True, exist_ok=False)
    try:
        files = []
        for label, policy, gpu in (("v3a", "legacy_quality_iou", "0"), ("v3b", "mask_multiscale", "1")):
            local = copy.deepcopy(base)
            local["runtime"].update(num_gpus=1, visible_gpus=gpu)
            baseline = output/f"{label}_base.yaml"
            baseline.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")
            method = {"name": f"scda_{label}_kvasir_b16_lr1e4", "base_experiment": baseline.name,
                      "v3": asdict(V3Options(assignment_policy=policy))}
            filename = output/f"{label}.yaml"
            filename.write_text(yaml.safe_dump(method, sort_keys=False), encoding="utf-8")
            load_experiment(filename)  # refuses wrong LR, workers, budget, initialization, etc.
            files.extend((baseline, filename))
        write_json(output/"preparation.json", {"status": "prepared", "training_started": False,
                   "reference": str(reference_path), "reference_sha256": sha256(reference_path),
                   "files_sha256": {p.name: sha256(p) for p in files}})
    except BaseException as error:
        # Preserve partial artifacts and explicit failure; never quietly launch.
        write_json(output/"preparation.json", {"status": "failed", "training_started": False,
                   "error": str(error)})
        raise
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.reference_experiment, args.output))


if __name__ == "__main__": main()
