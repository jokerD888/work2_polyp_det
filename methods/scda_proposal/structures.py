"""Plain records, not modules: no parameters, buffers, or checkpoint state."""
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SCDATarget:
    labels: torch.Tensor
    boxes: torch.Tensor  # cxcywh / augmented image (W,H,W,H), like native GT
    masks: torch.Tensor  # full ImageList padded pixel canvas, G,H,W
    reliable: torch.Tensor  # G bool, propagated alongside GT through crop/filter
    gt_ids: torch.Tensor  # stable ids for deterministic multi-instance arbitration
    valid_pixels: torch.Tensor  # H,W bool, excludes both LSJ and batch padding
    image_size: tuple[int, int]
    source: str = "unknown"
    original_gt_count: int | None = None


@dataclass
class TaskAssignment:
    token_indices: torch.Tensor
    gt_indices: torch.Tensor


@dataclass
class ImageAssignment:
    cls: TaskAssignment
    reg: TaskAssignment
    box_scale: torch.Tensor  # N,4; native proposal coordinates -> valid image
    audit: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossResult:
    losses: dict[str, torch.Tensor]  # already lambda weighted, add AFTER native weights
    unweighted: dict[str, torch.Tensor]
    global_gt_counts: torch.Tensor
