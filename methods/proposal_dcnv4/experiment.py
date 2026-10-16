"""Load a new experiment without altering the baseline/F-v2 presets."""
from dataclasses import asdict, replace
from pathlib import Path
import re

import yaml

from methods.scda_proposal.config import SCDAOptions
from methods.scda_proposal.experiment import load_experiment as load_scda
from polypbench.config import load_config
from polypbench.io import sha256

from .config import DCNv4Options
from .integration import NATIVE_COMMIT


def load_experiment(path, *, operator=None):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or set(raw) != {"base_experiment", "base_kind", "name", "dcnv4"}:
        raise ValueError("Use exactly base_experiment/base_kind/name/dcnv4")
    base = (path.parent / raw["base_experiment"]).resolve()
    if raw["base_kind"] == "native":
        cfg = load_config(base)
        baseline = base
        cfg["scda"] = asdict(SCDAOptions())
        cfg["baseline_config"] = str(base)
        cfg["baseline_config_sha256"] = sha256(base)
    elif raw["base_kind"] == "scda":
        cfg = load_scda(base)  # inherit an explicit, reviewed SCDA preset
        baseline = Path(cfg["baseline_config"])
        scda = cfg["scda"]
        if not scda["enabled"] or scda["reg_selection"] not in ("core_plus_transition", "quality_iou"):
            raise ValueError("Combined preset must inherit confirmed F-v2 or F-Reg-Quality")
        if scda["reg_selection"] == "quality_iou" and (
            scda["lambda_cls"] != 0.0 or scda["lambda_box"] != 0.05
        ):
            raise ValueError("Combined F-Reg-Quality requires lambda_cls=0 and lambda_box=0.05")
    else:
        raise ValueError("base_kind must be native or scda")
    if cfg["repo"]["commit"] != NATIVE_COMMIT:
        raise ValueError("Unreviewed MaskDINO version")
    if cfg["dataset"].get("training_boxes") != "annotations" or cfg["dataset"].get("mask_format") != "bitmask":
        raise ValueError("Preserve the current bbox-authoritative bitmask mapper")
    if not isinstance(raw["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", raw["name"]):
        raise ValueError("Invalid experiment name")
    if any(key.startswith(("MODEL.PROPOSAL_DCNV4.", "MODEL.SCDA.")) for key in cfg["overrides"]):
        raise ValueError("Method switches belong in method configuration")
    options = DCNv4Options(**raw["dcnv4"])
    configured_operator = options.operator
    if configured_operator not in ("dcnv4", "conv3x3"):
        raise ValueError("Configured operator must be dcnv4 or conv3x3")
    if operator is not None:
        options = replace(options, operator=operator)
    options.validate()
    cfg.update(name=raw["name"], adapter="proposal_dcnv4", dcnv4=asdict(options),
               source_config=str(path), method_base_config=str(base),
               method_base_config_sha256=sha256(base),
               baseline_config=str(baseline), implementation_status="CPU_ONLY_GPU_GATES_PENDING",
               operator_selection={"configured": configured_operator,
                                   "effective": options.operator,
                                   "source": "command_line" if operator is not None else "configuration"})
    return cfg
