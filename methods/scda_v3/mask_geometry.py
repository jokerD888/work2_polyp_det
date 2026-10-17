"""Confirmed visible-mask SDF and actual-grid scale preference, no gradients."""
import math
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def mask_sdf(mask, valid_pixels):
    from scipy.ndimage import distance_transform_edt
    if mask.shape != valid_pixels.shape or mask.ndim != 2 or not torch.isfinite(mask).all():
        raise ValueError("SDF needs a finite aligned 2D mask/valid-pixel canvas")
    visible = (mask.bool() & valid_pixels).cpu().numpy()
    area = int(visible.sum())
    if not area:
        raise ValueError("SDF is undefined for an empty visible mask")
    padded = np.pad(visible, 1, constant_values=False)
    field = distance_transform_edt(padded) - distance_transform_edt(~padded)
    radius = math.sqrt(area / math.pi)
    return torch.as_tensor(field[1:-1, 1:-1] / radius, device=mask.device,
                           dtype=torch.float64), radius * 2


@torch.no_grad()
def sample_cell_centers(sdf, cells):
    h, w = sdf.shape
    centers = (cells[:, :2] + cells[:, 2:]) / 2
    # Pixel centers are (x+.5,y+.5). align_corners=False maps them exactly.
    grid = centers / centers.new_tensor([w, h]) * 2 - 1
    return F.grid_sample(sdf[None, None].double(), grid[None, None].double(),
                         mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]


def level_preference(shapes, canvas, diameter, target_cells=8.):
    h_pad, w_pad = canvas
    scores = [abs(math.log2(diameter / (target_cells * math.sqrt((h_pad / h) * (w_pad / w)))))
              for h, w in shapes]
    return sorted(range(len(shapes)), key=lambda level: (scores[level], level)), scores
