"""CPU-only fitting/provenance helpers; no model imports or side effects."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from polypbench.io import sha256


RHO_GRID = [i / 100 for i in range(1, 101)]
IMAGENET_R50_SHA256 = "16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6"
SOURCE_FILES = (
    "polypbench/adapters/instance_lsj.py", "polypbench/adapters/bitmask_mapper.py",
    "methods/scda_proposal/config.py", "methods/scda_proposal/scda_assigner.py",
    "methods/scda_proposal/box_gate.py", "methods/scda_proposal/box_gate_calibration.py",
    "methods/scda_proposal/mapper.py", "methods/scda_proposal/model.py",
    "methods/scda_proposal/payload.py", "methods/scda_proposal/decoder.py",
    "methods/scda_proposal/integration.py", "methods/scda_proposal/precheck.py",
    "methods/scda_proposal/structures.py", "methods/scda_proposal/scda_loss.py",
    "scripts/calibrate_scda_box_gate.py",
)


def protocol_signature(cfg, options):
    reference = asdict(options)
    reference.update(reg_selection="quality_iou", box_gate_rho=None, diagnostics=False)
    values = {key: cfg[key] for key in ("dataset", "model", "seed", "overrides",
                                       "training_budget", "evaluation", "checkpoint_selection")}
    values["native_commit"] = cfg["repo"]["commit"]
    values["reference_scda"] = reference
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_identity(root):
    return {name: sha256(Path(root) / name) for name in SOURCE_FILES}


def fit_count_calibration(rows):
    """Match total eligible pool size, NOT capped selections or per-image masks.

    The predeclared grid is 0.01..1.00, ties choose the smaller rho. Pool counts
    include native Top-Q exclusion. No AP, validation labels or model updates.
    """
    if not rows:
        raise ValueError("No training observations for box-gate calibration")
    total_full = total_mask = 0
    candidates = [0] * len(RHO_GRID)
    for row in rows:
        full, mask, counts = row["full_count"], row["mask_count"], row["box_counts"]
        if (type(full) is not int or type(mask) is not int or not 0 <= mask <= full or
                len(counts) != len(RHO_GRID) or
                any(type(value) is not int or not 0 <= value <= full for value in counts) or
                any(a > b for a, b in zip(counts, counts[1:]))):
            raise ValueError("Invalid or nonmonotonic calibration counts")
        total_full += full
        total_mask += mask
        candidates = [a + b for a, b in zip(candidates, counts)]
    if not total_full or not total_mask:
        raise ValueError("Reference training sample has no mask-gated candidates; do not invent rho")
    index = min(range(len(RHO_GRID)), key=lambda i: (abs(candidates[i] - total_mask), RHO_GRID[i]))
    if not candidates[index]:
        raise ValueError("Best aggregate-count match has no box-gated candidates; user review required")
    return {"rho": RHO_GRID[index], "rho_grid": RHO_GRID, "full_count": total_full,
            "mask_count": total_mask, "box_count": candidates[index],
            "mask_retention": total_mask / total_full, "box_retention": candidates[index] / total_full,
            "count_difference": candidates[index] - total_mask,
            "relative_count_error": (candidates[index] - total_mask) / total_mask,
            "box_counts_by_rho": candidates,
            "objective": "min abs(total box pool - total mask pool); tie smaller rho"}


def load_calibration(path, cfg, reference_options):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Box-Gated calibration missing: {path}. Run scripts/calibrate_scda_box_gate.py "
            "with the matching F-Reg-Quality config first; rho has no implicit default.")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    if (record.get("schema_version") != 1 or record.get("status") != "calibrated" or
            record.get("split") != "train" or record.get("parameter_stage") != "ImageNet initialization; no updates"):
        raise ValueError("Need a completed train-only ImageNet-initialization calibration")
    if record.get("initialization_sha256") != IMAGENET_R50_SHA256:
        raise ValueError("Calibration must use the agreed ImageNet R50 initialization")
    if record.get("protocol_signature") != protocol_signature(cfg, reference_options):
        raise ValueError("Box-Gated calibration protocol differs from this experiment")
    if record.get("source_sha256") != source_identity(cfg["project_root"]):
        raise ValueError("Box-Gated calibration code changed; regenerate evidence before training")
    if record.get("train_annotations_sha256") != sha256(cfg["dataset"]["train"]["annotations"]):
        raise ValueError("Calibration training annotations changed")
    result = fit_count_calibration(record["observations"])
    if record.get("fit") != result:
        raise ValueError("Calibration rho/counts do not match recorded observations")
    # Embed full evidence in experiment.json; a raw external coefficient alone
    # is insufficient. The artifact hash is part of the run fingerprint.
    return result["rho"], {"path": str(path), "sha256": sha256(path), "record": record}
