"""Method config inherits the current baseline instead of inventing a schedule."""
from dataclasses import asdict
from pathlib import Path
import re

import yaml

from polypbench.config import load_config as load_baseline_config
from polypbench.io import sha256

from .config import SCDAOptions
from .integration import NATIVE_COMMIT


def load_experiment(path):
    path = Path(path).resolve()
    method = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    allowed = {"base_experiment", "name", "scda"}
    if not isinstance(method, dict) or set(method) - allowed:
        raise ValueError("Method config allows only base_experiment/name/scda; base protocol changes require explicit review")
    baseline = (path.parent / method["base_experiment"]).resolve()
    cfg = load_baseline_config(baseline)
    if cfg["repo"]["commit"] != NATIVE_COMMIT:
        raise ValueError("F-v1 implementation is pinned to the current baseline commit")
    if cfg["dataset"].get("bbox_source") != "provided" or cfg["dataset"].get("training_boxes") != "annotations":
        raise ValueError("Confirmed Kvasir method preserves supplied bbox targets")
    if cfg["dataset"].get("mask_format") != "bitmask":
        raise ValueError("Confirmed method requires the existing bitmask data")
    options = SCDAOptions(**method.get("scda", {})).validate()
    if options.mask_policy != "single_instance":
        raise ValueError("Only the user-confirmed single-instance Kvasir policy is enabled in this preset")
    native_overrides = {name for name in cfg["overrides"] if name.startswith("MODEL.SCDA.")}
    if native_overrides:
        raise ValueError("SCDA belongs in the method config, not the baseline overrides")
    name = method["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Invalid method experiment name")
    cfg.update({"name": name, "adapter": "scda_proposal", "scda": asdict(options),
                "source_config": str(path), "baseline_config": str(baseline),
                "baseline_config_sha256": sha256(baseline),
                "implementation_status": "code_only_unverified"})
    return cfg
