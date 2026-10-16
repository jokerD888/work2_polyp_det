"""One-time train-only box-gate calibration. No optimizer, eval or training run.

Model forwards use an explicit --allow-gpu switch. CPU unit tests cover fitting
and gate geometry; this script is for the prepared training-machine environment.
"""
import argparse
import dataclasses
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="methods/scda_proposal/configs/kvasir_freg_quality.yaml")
    parser.add_argument("--output", default="runs/scda-boxgate-calibration-s42/calibration.json")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("Real train-image forwards require explicit --allow-gpu on an idle GPU")
    if args.samples <= 0 or args.batch_size <= 0:
        parser.error("samples and batch-size must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing calibration: {output}")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from methods.scda_proposal.experiment import load_experiment
    from methods.scda_proposal.config import SCDAOptions
    from methods.scda_proposal.box_gate_calibration import (
        RHO_GRID, IMAGENET_R50_SHA256, fit_count_calibration, protocol_signature, source_identity)
    from polypbench.io import sha256
    experiment = load_experiment(args.config)
    options = SCDAOptions(**experiment["scda"])
    if options.reg_selection != "quality_iou" or options.lambda_cls != 0 or options.lambda_box != .05:
        raise ValueError("Calibrate against the confirmed F-Reg-Quality reference")
    identity = source_identity(root)
    signature = protocol_signature(experiment, options)
    sys.path.insert(0, experiment["repo"]["path"])
    import numpy as np
    import torch
    import train_net  # native registrations
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage
    from detectron2.utils.file_io import PathManager
    from methods.scda_proposal.integration import register
    from methods.scda_proposal.mapper import SCDALSJMapper
    from methods.scda_proposal.precheck import build_cfg
    from methods.scda_proposal.box_gate import calibration_observation
    register(experiment["repo"]["path"])
    seed = experiment["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    cfg = build_cfg(experiment, True, output.parent)
    weights = PathManager.get_local_path(cfg.MODEL.WEIGHTS)
    weights_sha = sha256(weights)
    if weights_sha != IMAGENET_R50_SHA256:
        raise ValueError("Calibration must start from the agreed ImageNet R50, not a trained checkpoint")
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.to("cuda").train()
    # Batch size is a memory-bound forward choice, not an optimization batch.
    # Mapping every selected image once fixes augmentation independently of it.
    entry = experiment["dataset"]["train"]
    records = load_coco_json(entry["annotations"], entry["image_root"],
                             extra_annotation_keys=["id", "scda_reliable"])
    if args.samples > len(records):
        raise ValueError("Requested sample count exceeds training images")
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed)).tolist()[:args.samples]
    mapper = SCDALSJMapper(cfg, options.mask_policy)
    mapped = [mapper(records[index]) for index in order]
    # Audit callback is training-only. No validation images/evaluator are loaded.
    rows = []
    image_ids = [int(records[index]["image_id"]) for index in order]
    def callback(current, native_losses, auxiliary, payload, targets, assignments):
        for b, target in enumerate(targets):
            row = calibration_observation(payload, b, target, options.tau_reg, RHO_GRID)
            row["image_id"] = image_ids[len(rows)]
            rows.append(row)
    model.scda_options = dataclasses.replace(model.scda_options, diagnostics=True)
    model.scda_audit_callback = callback
    with torch.no_grad(), EventStorage(0) as storage:
        for start in range(0, len(mapped), args.batch_size):
            with torch.autocast("cuda", enabled=cfg.SOLVER.AMP.ENABLED):
                model(mapped[start:start + args.batch_size])
            storage.step()
    if len(rows) != args.samples:
        raise RuntimeError("Incomplete calibration observations")
    fit = fit_count_calibration(rows)
    chosen_index = RHO_GRID.index(fit["rho"])
    level_count = len(rows[0]["full_counts_per_level"])
    descriptives = {
        "eligible_gt": sum(row["eligible_gt"] for row in rows),
        "mask_capped_selected_count": sum(min(options.k_reg, row["mask_count"]) for row in rows),
        "box_capped_selected_count": sum(min(options.k_reg, row["box_counts"][chosen_index]) for row in rows),
        "mask_shortfall_gt": sum(row["eligible_gt"] > 0 and row["mask_count"] < options.k_reg for row in rows),
        "box_shortfall_gt": sum(row["eligible_gt"] > 0 and row["box_counts"][chosen_index] < options.k_reg for row in rows),
        "mask_pool_per_level": [sum(row["mask_counts_per_level"][i] for row in rows) for i in range(level_count)],
        "box_pool_per_level": [sum(row["box_counts_per_level_by_rho"][chosen_index][i] for row in rows)
                               for i in range(level_count)],
        "note": "Descriptive only; fitting uses total pool counts, not per-GT/per-level matches or gradient norms"}
    record = {"schema_version": 1, "status": "calibrated", "split": "train",
              "parameter_stage": "ImageNet initialization; no updates",
              "seed": seed, "samples": args.samples, "forward_batch_size": args.batch_size,
              "image_ids": image_ids, "protocol_signature": signature,
              "source_sha256": identity, "train_annotations_sha256": sha256(entry["annotations"]),
              "initialization_sha256": weights_sha, "observations": rows, "fit": fit,
              "matching_descriptives": descriptives,
              "native_commit": experiment["repo"]["commit"], "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "scope": "global train-pool retention match; not per-image/per-level/gradient matching"}
    if source_identity(root) != identity:
        raise RuntimeError("Source changed during calibration")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects evidence even against concurrent calibrators.
    with output.open("x", encoding="utf-8") as stream:
        import json
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print({"output": str(output), **{k: v for k, v in fit.items() if k not in ("rho_grid", "box_counts_by_rho")}})


if __name__ == "__main__":
    main()
