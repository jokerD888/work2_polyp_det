import copy
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

MASKDINO_COMMIT = "3831d8514a3728535ace8d4ecc7d28044c42dd14"


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Experiment must be a YAML mapping")
    cfg = copy.deepcopy(cfg)
    root = (path.parent / cfg.pop("project_root", "..")).resolve()

    def resolve(value):
        value = os.path.expandvars(os.path.expanduser(str(value)))
        return str((root / value).resolve())

    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["name"]):
        raise ValueError("name may contain letters, digits, underscores and hyphens")
    if cfg.get("adapter") != "maskdino":
        raise ValueError("Only the maskdino adapter is currently implemented")
    if cfg["dataset"].get("supervision") != "bbox+mask":
        raise ValueError("The native MaskDINO baseline requires bbox+mask supervision")
    if cfg["dataset"].get("mask_format", "polygon") not in ("polygon", "bitmask"):
        raise ValueError("dataset.mask_format must be polygon or bitmask")
    if cfg["dataset"].get("training_boxes", "mask") not in ("mask", "annotations"):
        raise ValueError("dataset.training_boxes must be mask or annotations")
    if cfg["dataset"].get("training_boxes") == "annotations" and cfg["dataset"].get("mask_format") != "bitmask":
        raise ValueError("Supplied training boxes require the bitmask adapter")
    if cfg["dataset"].get("bbox_source") == "provided" and cfg["dataset"].get("training_boxes") != "annotations":
        raise ValueError("Provided bbox labels must use training_boxes: annotations")
    cfg["repo"]["path"] = resolve(cfg["repo"]["path"])
    cfg["repo"].setdefault("commit", MASKDINO_COMMIT)
    cfg["model"]["config"] = str((Path(cfg["repo"]["path"]) / cfg["model"]["config"]).resolve())
    weights = cfg["model"].get("weights", "")
    cfg["model"]["weights"] = weights if not weights or "://" in weights else resolve(weights)
    for split in ("train", "val", "test"):
        if split in cfg["dataset"]:
            for key in ("annotations", "image_root"):
                cfg["dataset"][split][key] = resolve(cfg["dataset"][split][key])
    if not {"train", "val"}.issubset(cfg["dataset"]):
        raise ValueError("Both train and val splits are required")
    cfg["output_root"] = resolve(cfg.get("output_root", "runs"))
    cfg["runtime"] = cfg.get("runtime", {})
    cfg["runtime"].setdefault("python", "python")
    cfg["runtime"].setdefault("num_gpus", 1)
    cfg.setdefault("seed", 42)
    cfg.setdefault("overrides", {})
    if cfg.get("checkpoint_selection", "final") not in ("final", "best_val_ap"):
        raise ValueError("checkpoint_selection must be final or best_val_ap")
    if cfg.get("checkpoint_keep", 2) < 1:
        raise ValueError("checkpoint_keep must be positive")
    cfg.setdefault("evaluation", {"max_dets": [1, 10, 100]})
    if cfg["runtime"]["num_gpus"] < 1:
        raise ValueError("MaskDINO runner requires at least one GPU")
    batch = cfg["overrides"].get("SOLVER.IMS_PER_BATCH", 16)
    if batch < cfg["runtime"]["num_gpus"] or batch % cfg["runtime"]["num_gpus"]:
        raise ValueError("SOLVER.IMS_PER_BATCH must be divisible by runtime.num_gpus")
    if cfg["model"]["num_classes"] < 1:
        raise ValueError("num_classes must be positive")
    controlled = {"DATASETS.TRAIN", "DATASETS.TEST", "MODEL.WEIGHTS", "MODEL.SEM_SEG_HEAD.NUM_CLASSES", "OUTPUT_DIR", "SEED", "TEST.AUG.ENABLED", "TEST.DETECTIONS_PER_IMAGE"}
    collision = controlled.intersection(cfg["overrides"])
    if collision:
        raise ValueError(f"Use dedicated experiment fields instead of overrides: {sorted(collision)}")
    if cfg["overrides"].get("INPUT.DATASET_MAPPER_NAME", "coco_instance_lsj") != "coco_instance_lsj":
        raise ValueError("This baseline adapter validates the official polygon-mask LSJ mapper only")
    maximum = cfg["evaluation"].get("max_dets", [1, 10, 100])
    if maximum != [1, 10, 100]:
        raise ValueError("The initial bbox protocol fixes max_dets=[1,10,100]")
    if cfg["model"].get("detections_per_image", 100) > 300 * cfg["model"]["num_classes"]:
        raise ValueError("detections_per_image exceeds query/class candidates")
    cfg["project_root"] = str(root)
    cfg["source_config"] = str(path)
    return cfg


def fingerprint(cfg):
    # A run is tied to its resolved configuration; cross-machine runs get new IDs.
    value = copy.deepcopy(cfg)
    value.pop("source_config", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
