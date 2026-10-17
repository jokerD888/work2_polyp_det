"""Dump actual per-param-group learning rates for a prepared experiment chain.

Verifies the real optimizer LRs (including MaskDINO's backbone multiplier and
any adapter params) match the intended 1e-4 protocol before launch. CPU only.
"""
import argparse
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", required=True)
parser.add_argument("--family", choices=("scda", "dcnv4"), required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/MaskDINO"))
os_ok = __import__("os").environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from polypbench.io import read_json, write_json

if args.family == "scda":
    from methods.scda_proposal.experiment import load_experiment
    from methods.scda_proposal.precheck import build_cfg
else:
    from methods.proposal_dcnv4.experiment import load_experiment
    experiment = load_experiment(args.config)
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    import train_net as native
    from methods.scda_proposal.config import add_scda_config
    from methods.proposal_dcnv4.config import add_dcnv4_config
    from methods.proposal_dcnv4.integration import configure_model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    native.add_maskdino_config(cfg)
    add_scda_config(cfg)
    add_dcnv4_config(cfg)
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    for key, value in experiment["scda"].items():
        setattr(cfg.MODEL.SCDA, key.upper(), value)
    for key, value in experiment["dcnv4"].items():
        setattr(cfg.MODEL.PROPOSAL_DCNV4, key.upper(), value)
    configure_model(cfg)
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = experiment["model"]["weights"]
    cfg.DATASETS.TRAIN = ("bench_train",)
    cfg.DATASETS.TEST = ("bench_val",)
    cfg.SEED = experiment["seed"]
    cfg.OUTPUT_DIR = str(Path(args.output).parent)
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment["model"].get("detections_per_image", 100)
    cfg.freeze()

if args.family == "scda":
    experiment = load_experiment(args.config)
    cfg = build_cfg(experiment, True, Path(args.output).parent)

from detectron2.modeling import build_model
import train_net as native

if args.family == "scda":
    from methods.scda_proposal.integration import register
    register(experiment["repo"]["path"])
else:
    from methods.proposal_dcnv4.integration import register
    register(experiment["repo"]["path"])

import torch
torch.manual_seed(0)
model = build_model(cfg)
optimizer = native.Trainer.build_optimizer(cfg, model)

name_of = {id(p): n for n, p in model.named_parameters()}
groups = []
dcnv4_groups = {}
for gi, group in enumerate(optimizer.param_groups):
    params = group["params"]
    names = [name_of.get(id(p), "?") for p in params]
    entry = {"group": gi, "lr": group["lr"], "n_params": len(params),
             "n_tensors": sum(p.numel() for p in params),
             "name_samples": names[:3],
             "modules": sorted({n.split(".")[0] + "." + (n.split(".")[1] if "." in n else "") for n in names})}
    groups.append(entry)
    for n in names:
        if "dcnv4" in n:
            dcnv4_groups.setdefault(group["lr"], 0)
            dcnv4_groups[group["lr"]] += 1

record = {"config": args.config, "family": args.family,
          "base_lr_cfg": cfg.SOLVER.BASE_LR, "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
          "n_groups": len(groups), "groups": groups,
          "dcnv4_param_tensors_by_lr": dcnv4_groups,
          "max_lr": max(g["lr"] for g in groups),
          "all_lrs_in_expected_set": all(g["lr"] in (1e-4, 1e-5) for g in groups)}
write_json(args.output, record)
print({"max_lr": record["max_lr"], "n_groups": len(groups),
       "lrs": [g["lr"] for g in groups],
       "dcnv4_tensors_by_lr": dcnv4_groups,
       "expected_set_ok": record["all_lrs_in_expected_set"]})
