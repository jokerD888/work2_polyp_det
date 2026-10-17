"""Strict low-LR Kvasir protocol, independent of the legacy high-LR preset."""
from dataclasses import asdict
from pathlib import Path
import re
import yaml

from polypbench.config import load_config, MASKDINO_COMMIT
from polypbench.io import sha256
from methods.scda_proposal.box_gate_calibration import native_config_identity
from .config import V3Options


def load_experiment(path):
    path = Path(path).resolve()
    method = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(method, dict) or set(method) != {"name", "base_experiment", "v3"}:
        raise ValueError("V3 config requires only name/base_experiment/v3")
    if not isinstance(method["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", method["name"]):
        raise ValueError("Invalid V3 name")
    baseline = (path.parent / method["base_experiment"]).resolve()
    cfg = load_config(baseline)
    options = V3Options(**method["v3"]).validate()
    expected = {"SOLVER.IMS_PER_BATCH": 16, "SOLVER.BASE_LR": .0001,
                "SOLVER.BACKBONE_MULTIPLIER": .1, "SOLVER.MAX_ITER": 5500,
                "SOLVER.WARMUP_ITERS": 125, "SOLVER.WARMUP_FACTOR": .001,
                "SOLVER.STEPS": [4400, 4950], "SOLVER.GAMMA": .1,
                "SOLVER.OPTIMIZER": "ADAMW", "SOLVER.WEIGHT_DECAY": .05,
                "SOLVER.AMP.ENABLED": True, "SOLVER.CLIP_GRADIENTS.ENABLED": True,
                "SOLVER.CLIP_GRADIENTS.CLIP_TYPE": "full_model",
                "SOLVER.CLIP_GRADIENTS.CLIP_VALUE": .01, "SOLVER.REFERENCE_WORLD_SIZE": 0,
                "INPUT.IMAGE_SIZE": 512, "INPUT.MIN_SCALE": .5, "INPUT.MAX_SCALE": 1.5,
                "INPUT.RANDOM_FLIP": "horizontal", "INPUT.MIN_SIZE_TEST": 512,
                "INPUT.MAX_SIZE_TEST": 512, "DATALOADER.NUM_WORKERS": 4,
                "SOLVER.CHECKPOINT_PERIOD": 275, "TEST.EVAL_PERIOD": 275}
    if any(cfg["overrides"].get(key) != value for key, value in expected.items()):
        raise ValueError("V3 requires the confirmed low-LR B16/workers4/5500iter protocol")
    if (cfg["seed"] != 42 or cfg["repo"]["commit"] != MASKDINO_COMMIT or
            cfg["model"]["num_classes"] != 1 or cfg["model"]["weights"] !=
            "detectron2://ImageNetPretrained/torchvision/R-50.pkl" or
            cfg["checkpoint_selection"] != "best_val_ap" or
            cfg["training_budget"] != {"train_images": 880, "iterations_per_epoch": 55, "epochs": 100} or
            cfg["dataset"].get("bbox_source") != "provided" or
            cfg["dataset"].get("training_boxes") != "annotations" or
            cfg["dataset"].get("mask_format") != "bitmask"):
        raise ValueError("V3 requires seed42, same ImageNet R50, supplied bbox, bitmask and official 880/120 protocol")
    if any(key.startswith(("MODEL.SCDA", "MODEL.PROPOSAL_DCNV4")) for key in cfg["overrides"]):
        raise ValueError("Legacy methods/adapters cannot be mixed into V3")
    cfg.update(name=method["name"], adapter="scda_v3", v3=asdict(options), source_config=str(path),
               baseline_config=str(baseline), baseline_config_sha256=sha256(baseline),
               native_model_config=native_config_identity(cfg["model"]["config"]),
               implementation_status="acceptance_evidence_required_before_training")
    return cfg
