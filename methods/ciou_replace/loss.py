"""Pinned Detectron2 CIoU semantics, without decoding/clipping input boxes."""
from pathlib import Path
import inspect

import torch
from detectron2.layers.losses import ciou_loss

from polypbench.io import sha256

REFERENCE_SHA256 = "6426d06f915f12f0eb5bf4f31f94092b23422d663337a22fc27a8b4ea8630ccb"


def reference_record():
    path = Path(inspect.getfile(ciou_loss)).resolve()
    digest = sha256(path)
    if digest != REFERENCE_SHA256:
        raise RuntimeError(f"CIoU reference drift: {path}: {digest}")
    return {"path": str(path), "sha256": digest, "epsilon": 1e-7,
            "alpha_stop_gradient": True, "iou_gate": False,
            "input": "decoded normalized XYXY; no clipping/sigmoid/exp"}


def ciou_pairs(boxes, targets):
    if boxes.shape != targets.shape or boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("CIoU requires corresponding Nx4 XYXY boxes")
    # Retain double precision for reference tests; explicitly upcast AMP inputs.
    dtype = torch.float64 if boxes.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        boxes, targets = boxes.to(dtype), targets.to(dtype)
        if not boxes.numel():
            return boxes.sum(-1) * 0
        for label, value in (("prediction", boxes), ("target", targets)):
            size = value[:, 2:] - value[:, :2]
            valid = torch.isfinite(value).all(-1) & (size > 0).all(-1)
            if not bool(valid.all()):
                bad = (~valid).nonzero().flatten()[:8]
                raise FloatingPointError(
                    f"Invalid {label} XYXY; coordinates/gradients were NOT clipped: "
                    f"{value[bad].detach().cpu().tolist()}")
        result = ciou_loss(boxes, targets, reduction="none", eps=1e-7)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("CIoU produced a nonfinite loss")
        return result
