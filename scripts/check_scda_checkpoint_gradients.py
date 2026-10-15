"""Measure train-batch auxiliary gradients after smoke updates; no optimizer step."""
import argparse
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party/MaskDINO"))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    import numpy as np
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from detectron2.utils.events import EventStorage
    from methods.scda_proposal.experiment import load_experiment
    from methods.scda_proposal.precheck import build_cfg
    from methods.scda_proposal.integration import register
    from methods.scda_proposal.mapper import SCDALSJMapper
    from methods.scda_proposal.diagnostics import gradient_audit
    from methods.scda_proposal.audit_summary import validate_v2_assignments, summarize_decoupling
    from polypbench.io import write_json, sha256
    import train_net
    experiment = load_experiment(args.config)
    register(experiment["repo"]["path"])
    entry = experiment["dataset"]["train"]
    DatasetCatalog.register("bench_train", lambda: load_coco_json(
        entry["annotations"], entry["image_root"], "bench_train", extra_annotation_keys=["id"]))
    MetadataCatalog.get("bench_train").set(json_file=entry["annotations"], image_root=entry["image_root"], evaluator_type="coco")
    cfg = build_cfg(experiment, True, Path(args.output).parent)
    random.seed(experiment["seed"]); np.random.seed(experiment["seed"]); torch.manual_seed(experiment["seed"])
    torch.set_num_threads(2)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(args.checkpoint)
    model.cuda().train()
    mapper = SCDALSJMapper(cfg, "single_instance")
    batch = []
    for record in DatasetCatalog.get("bench_train"):
        if len(record["annotations"]) != 1:
            continue
        mapped = mapper(record)
        if len(mapped["instances"]) and mapped["instances"].gt_masks.any():
            batch.append(mapped)
        if len(batch) == 4:
            break
    if len(batch) != 4:
        raise RuntimeError("No reliable four-image diagnostic batch")
    report = {"checkpoint_sha256": sha256(args.checkpoint), "scda_options": experiment["scda"],
              "note": "Post-smoke train-only AMP gradient diagnostic, no optimizer updates or val tuning"}
    def callback(current, native, auxiliary, payload, targets, assignments):
        report["gradients"] = gradient_audit(current, native, auxiliary, payload)
        audits = [assignment.audit for assignment in assignments]
        report["policy"] = validate_v2_assignments(audits)
        report["assignments"] = summarize_decoupling(audits)
    model.scda_audit_callback = callback
    with EventStorage():
        with torch.autocast("cuda", enabled=True):
            losses = model(batch)
        total = sum(losses.values())
        if not torch.isfinite(total): raise FloatingPointError("Non-finite smoke checkpoint loss")
        total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite smoke checkpoint gradient")
    report["losses"] = {key: float(value.detach()) for key, value in losses.items() if key.startswith("loss_scda")}
    for group, objective in (("class_embed", "cls_weighted"), ("bbox_embed", "box_weighted"),
                             ("encoder", "box_weighted"), ("backbone", "box_weighted")):
        norm = report["gradients"]["parameter_space"][group]["gradient_norms"][objective]
        if not np.isfinite(norm) or norm <= 0:
            raise AssertionError(f"No finite nonzero {objective} gradient to {group} after smoke updates")
    report["passed"] = True
    write_json(args.output, report)
    print(f"Post-smoke gradient checks passed: {args.output}")


if __name__ == "__main__":
    main()
