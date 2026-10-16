"""Confirmed box-only spatial gate; mask is used only for shared GT validity.

The gate is a half-open concentric box of (rho*w, rho*h). No mask pixel
position, per-GT mask area or mask-derived count changes its token decisions.
Calibration compares aggregate train counts before capped IoU selection.
"""
import torch

from .structures import ImageAssignment, TaskAssignment
from .scda_assigner import (token_geometry, fully_valid_cells, rectangle_coverage,
                            _stable_rank_quality, _token_gt_iou)


def shrink_box_contains(center, box, rho):
    center, box = center.double(), box.double()
    half = box[2:] * (float(rho) / 2)
    return ((center >= box[:2] - half) & (center < box[:2] + half)).all(-1)


def candidate_geometry(payload, b, target):
    cells, scales, keys = token_geometry(payload, b, target)
    center = payload["base_anchor"][b, :, :2].float() * scales[:, :2]
    valid = (payload["valid_token_mask"][b] & fully_valid_cells(target.valid_pixels, cells) &
             torch.isfinite(center).all(-1) & (center >= 0).all(-1) & (center < 1).all(-1))
    selected = torch.zeros(len(cells), dtype=torch.bool, device=cells.device)
    selected[payload["topk_indices"][b]] = True
    return cells, scales, keys, center, valid, valid & ~selected


def gt_validity(target, j):
    box = target.boxes[j].float()
    valid_box = bool(torch.isfinite(box).all() and (box[2:] > 0).all())
    # Same augmented-GT set as F-Reg-Quality. This Boolean checks existence,
    # not shape/coverage. Native mask supervision remains present in all runs.
    mask = target.masks[j]
    nonempty = bool(torch.isfinite(mask).all() and mask.bool().any())
    return bool(target.reliable[j]), valid_box, nonempty


@torch.no_grad()
def calibration_observation(payload, b, target, tau_reg=.25, grid=None):
    """One fixed train image; GT-level counts only, no fitting or val access."""
    grid = grid if grid is not None else [i / 100 for i in range(1, 101)]
    cells, scales, keys, center, valid, available = candidate_geometry(payload, b, target)
    counts = [0] * len(grid)
    mask_count = full_count = 0
    levels = len(payload["spatial_shapes"])
    mask_levels, full_levels = [0] * levels, [0] * levels
    box_levels = [[0] * levels for _ in grid]
    eligible_gt = 0
    for j in range(len(target.labels)):
        if not all(gt_validity(target, j)):
            continue
        eligible_gt += 1
        box = target.boxes[j].float()
        inside = ((center >= box[:2] - box[2:] / 2) &
                  (center < box[:2] + box[2:] / 2)).all(-1)
        full = available & inside
        coverage = rectangle_coverage((target.masks[j].bool() & target.valid_pixels)[None], cells)[0]
        mask_pool = full & (coverage >= tau_reg)
        full_count += int(full.sum())
        mask_count += int(mask_pool.sum())
        # All predeclared radii in one operation/host transfer. Avoid one GPU
        # synchronization for every grid point; geometry is the same FP64 rule.
        radii = torch.tensor(grid, dtype=torch.float64, device=center.device)[:, None, None]
        half = box.double()[2:] * (radii / 2)
        gates = ((center.double()[None] >= box.double()[:2] - half) &
                 (center.double()[None] < box.double()[:2] + half)).all(-1)
        current_counts = (gates & full[None]).sum(-1).tolist()
        counts = [a + value for a, value in zip(counts, current_counts)]
        current_levels = torch.stack([
            (gates & full[None] & (keys[:, 0] == level)[None]).sum(-1)
            for level in range(levels)], -1).tolist()
        box_levels = [[a + b for a, b in zip(old, new)]
                      for old, new in zip(box_levels, current_levels)]
        for level in range(levels):
            on_level = keys[:, 0] == level
            full_levels[level] += int((full & on_level).sum())
            mask_levels[level] += int((mask_pool & on_level).sum())
    if eligible_gt > 1:
        raise ValueError("Box-Gated calibration requires the confirmed original-single-instance policy")
    return {"eligible_gt": eligible_gt, "full_count": full_count, "mask_count": mask_count,
            "box_counts": counts, "mask_counts_per_level": mask_levels,
            "full_counts_per_level": full_levels, "box_counts_per_level_by_rho": box_levels}


@torch.no_grad()
def assign_box_gate_image(payload, b, target, options):
    cells, scales, keys, center, valid, available = candidate_geometry(payload, b, target)
    empty = lambda: TaskAssignment(torch.empty(0, dtype=torch.long, device=cells.device),
                                    torch.empty(0, dtype=torch.long, device=cells.device))
    reg_tokens, reg_gt, entries = [], [], []
    claimed = torch.zeros(len(cells), dtype=torch.bool, device=cells.device)
    eligible_gt = 0
    for j in torch.argsort(target.gt_ids, stable=True).tolist():
        reliable, valid_box, nonempty = gt_validity(target, j)
        box = target.boxes[j].float()
        entry = {"gt_index": j, "gt_id": int(target.gt_ids[j]), "source": target.source,
                 "reliable": reliable, "valid_box": valid_box, "nonempty_mask": nonempty,
                 "n_cls_selected": 0, "n_reg_selected": 0, "n_cls_owned": 0,
                 "cls_token_indices": [], "reg_token_indices": [], "cls_coverage": [],
                 "reg_coverage": [], "cls_subset_reg": True,
                 "reg_selection": options.reg_selection, "box_gate_rho": options.box_gate_rho}
        if options.diagnostics:
            entries.append(entry)
        if not (reliable and valid_box and nonempty):
            entry["skip_reason"] = "unreliable_mask" if not reliable else ("invalid_box" if not valid_box else "empty_mask")
            continue
        eligible_gt += 1
        if eligible_gt > 1:
            # Do not invent mask-free arbitration for an unapproved multi-GT
            # extension. Original multi-instance Kvasir images have no reliable GT.
            raise ValueError("Box-Gated supports only the confirmed original-single-instance policy")
        inside = ((center >= box[:2] - box[2:] / 2) &
                  (center < box[:2] + box[2:] / 2)).all(-1)
        gate = shrink_box_contains(center, box, options.box_gate_rho)
        eligible = available & inside & gate
        claimed |= eligible
        pool = torch.nonzero(eligible, as_tuple=True)[0]
        quality = _token_gt_iou(payload, b, pool, box, scales)
        distance = (((center - box[:2]) / box[2:]) ** 2).sum(-1).double()
        ranked = torch.round(quality * 1e4) / 1e4
        order = _stable_rank_quality(torch.arange(len(pool), device=pool.device),
                                     ranked, distance[pool], keys[pool])[:options.k_reg]
        chosen = pool[order]
        reg_tokens.append(chosen)
        reg_gt.append(torch.full_like(chosen, j))
        if not options.diagnostics:
            continue
        # Coverage is diagnostic ONLY, evaluated after the selected set is fixed.
        mask_coverage = rectangle_coverage((target.masks[j].bool() & target.valid_pixels)[None], cells)[0]
        mask_pool = available & inside & (mask_coverage >= options.tau_reg)
        entry.update({"area_pixels": float(box[2:].prod() * target.image_size[0] * target.image_size[1]),
                      "n_geometry": int((valid & inside).sum()), "n_cls_eligible": 0,
                      "n_cls_before_topq_exclusion": 0,
                      "n_reg_before_topq_exclusion": int((valid & inside & gate).sum()),
                      "n_reg_eligible": len(pool), "n_reg_owned": len(pool),
                      "n_reg_selected": len(chosen), "reg_shortfall": options.k_reg - len(chosen),
                      "reg_full_quota": len(chosen) == options.k_reg, "cls_shortfall": 0,
                      "cls_levels": [], "reg_levels": keys[chosen, 0].tolist(),
                      "reg_token_indices": chosen.tolist(), "reg_iou": quality[order].tolist(),
                      "reg_coverage": mask_coverage[chosen].tolist(),
                      "box_gate_reference_mask_pool_count": int(mask_pool.sum()),
                      "box_gate_full_box_pool_count": int((available & inside).sum()),
                      "both_tasks_empty": len(chosen) == 0,
                      "per_level": [{"level": level,
                                     "n_reg_eligible": int((eligible & (keys[:, 0] == level)).sum()),
                                     "mask_reference_eligible": int((mask_pool & (keys[:, 0] == level)).sum())}
                                    for level in range(len(payload["spatial_shapes"]))]})
    reg = TaskAssignment(torch.cat(reg_tokens), torch.cat(reg_gt)) if reg_tokens else empty()
    audit = {}
    if options.diagnostics:
        audit = {"n_gt": len(target.labels), "n_original_gt": target.original_gt_count,
                 "n_valid": int(valid.sum()), "n_available": int(available.sum()),
                 "n_claimed_tokens": int(claimed.sum()), "n_conflicting_tokens": 0,
                 "per_gt": entries, "reg_selection": options.reg_selection,
                 "padding_policy": options.padding_policy, "geometry": options.geometry,
                 "box_gate_rho": options.box_gate_rho}
    return ImageAssignment(empty(), reg, scales, audit)
