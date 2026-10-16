"""Versioned pure-function assignment. Only labels and geometry are detached.

Pixel masks are piecewise constant on unit pixel squares. Continuous rectangle
integration exactly accounts for fractional feature-grid cell edges. No fixed
stride assumptions, adaptive pooling, or learned state. The confirmed F-v1/v2
policies never rank by predicted scores; the opt-in quality_iou policy ranks
regression candidates by their current predicted-box IoU with the GT, read
from the detached payload inside this no_grad assignment.
"""
import torch
from torch.nn import functional as F

from .structures import ImageAssignment, TaskAssignment


def _prefix_at(prefix, x, y):
    """Integral from origin to (x,y), bilinear extension of integer prefix sums."""
    height, width = prefix.shape[-2] - 1, prefix.shape[-1] - 1
    x, y = x.clamp(0, width), y.clamp(0, height)
    ix, iy = x.floor().long(), y.floor().long()
    nx, ny = (ix + 1).clamp(max=width), (iy + 1).clamp(max=height)
    fx, fy = x - ix, y - iy
    a = prefix[..., iy, ix]
    b = prefix[..., iy, nx]
    c = prefix[..., ny, ix]
    d = prefix[..., ny, nx]
    return a + (b - a) * fx + (c - a) * fy + (d - b - c + a) * fx * fy


def rectangle_coverage(masks, cells):
    """Masks G,H,W; cells N,4 xyxy in that canvas; returns G,N float64."""
    masks = masks.to(dtype=torch.float64)
    prefix = F.pad(masks.cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    x0, y0, x1, y1 = cells.unbind(-1)
    integral = (_prefix_at(prefix, x1, y1) - _prefix_at(prefix, x0, y1) -
                _prefix_at(prefix, x1, y0) + _prefix_at(prefix, x0, y0))
    area = (x1 - x0) * (y1 - y0)
    return (integral / area).clamp(0, 1)


def fully_valid_cells(valid_pixels, cells):
    """Reject any intersected padding pixel using exact integer prefix sums."""
    prefix = F.pad((~valid_pixels).long().cumsum(-2).cumsum(-1), (1, 0, 1, 0))
    lower, upper = cells[:, :2].floor().long(), cells[:, 2:].ceil().long()
    height, width = valid_pixels.shape
    x0, y0 = lower[:, 0].clamp(0, width), lower[:, 1].clamp(0, height)
    x1, y1 = upper[:, 0].clamp(0, width), upper[:, 1].clamp(0, height)
    invalid_count = prefix[y1, x1] - prefix[y0, x1] - prefix[y1, x0] + prefix[y0, x0]
    return invalid_count == 0


def token_geometry(payload, batch_index, target):
    """Keep native flatten order; convert its coordinate domain to image units.

Native proposal normalization uses valid_W/valid_H derived from its own mask.
Its current forward often disables that mask. We never alter it. The per-level
factor below maps this native domain through the actual padded feature grid to
the native GT domain (augmented image width/height). LSJ padding is handled
separately by full-cell exclusion, not by changing native proposal generation.
"""
    shapes = payload["spatial_shapes"].tolist()
    starts = payload["level_start_index"].tolist()
    h_pad, w_pad = target.valid_pixels.shape
    h_image, w_image = target.image_size
    device = payload["raw_cls"].device
    cells, scales, keys = [], [], []
    expected_start = 0
    for level, ((h, w), start) in enumerate(zip(shapes, starts)):
        if start != expected_start:
            raise ValueError("Non-contiguous native proposal level layout")
        expected_start += h * w
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        xd, yd = x.to(torch.float64), y.to(torch.float64)
        cell = torch.stack((xd * (w_pad / w), yd * (h_pad / h),
                            (xd + 1) * (w_pad / w), (yd + 1) * (h_pad / h)), -1)
        cells.append(cell.reshape(-1, 4).to(torch.float64))
        native_mask = payload["mask_flatten"][batch_index, start:start + h * w].reshape(h, w)
        valid_h = (~native_mask[:, 0]).sum()
        valid_w = (~native_mask[0, :]).sum()
        scale = torch.stack((valid_w * (w_pad / w / w_image),
                             valid_h * (h_pad / h / h_image)))
        scales.append(scale.repeat(2).float().expand(h * w, 4))
        # Explicit lexicographic (level,u=x,v=y), not accidental row-major ties.
        keys.append(torch.stack((torch.full_like(x, level), x, y), -1).reshape(-1, 3))
    if expected_start != payload["raw_cls"].shape[1]:
        raise ValueError("Proposal count does not match spatial_shapes")
    return torch.cat(cells), torch.cat(scales), torch.cat(keys)


def _stable_rank(indices, coverage, distance, keys):
    """Stable successive sorts: coverage desc, distance asc, level/u/v asc."""
    for column in (2, 1, 0):
        indices = indices[torch.argsort(keys[indices, column], stable=True)]
    indices = indices[torch.argsort(distance[indices], stable=True)]
    return indices[torch.argsort(coverage[indices], descending=True, stable=True)]


def _stable_rank_quality(indices, quality, distance, keys):
    """Stable successive sorts: quality desc, distance asc, level/u/v asc."""
    for column in (2, 1, 0):
        indices = indices[torch.argsort(keys[indices, column], stable=True)]
    indices = indices[torch.argsort(distance[indices], stable=True)]
    return indices[torch.argsort(quality[indices], descending=True, stable=True)]


def _token_gt_iou(payload, batch_index, tokens, gt_box, box_scale):
    """Paired IoU between the token's current predicted box and the GT, FP32.

    Same coordinate convention as the auxiliary box loss: normalized cxcywh
    proposal scaled into the valid-image domain before comparison. Pure
    ranking input for quality_iou selection; detached with the assignment.
    """
    from .scda_loss import cxcywh_to_xyxy
    prediction = cxcywh_to_xyxy(payload["raw_box_sigmoid"][batch_index, tokens].float() * box_scale[tokens])
    truth = cxcywh_to_xyxy(gt_box.float()[None].expand(tokens.numel(), 4))
    intersection = (torch.minimum(prediction[:, 2:], truth[:, 2:]) -
                    torch.maximum(prediction[:, :2], truth[:, :2])).clamp(min=0).prod(-1)
    area_p = (prediction[:, 2:] - prediction[:, :2]).clamp(min=0).prod(-1)
    area_t = (truth[:, 2:] - truth[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_p + area_t - intersection).clamp(min=torch.finfo(torch.float32).tiny)


def _empty_task(device):
    empty = torch.empty(0, dtype=torch.long, device=device)
    return TaskAssignment(empty, empty.clone())


@torch.no_grad()
def assign_image(payload, b, target, options):
    cells, box_scale, keys = token_geometry(payload, b, target)
    device = cells.device
    n = cells.shape[0]
    gt_count = len(target.labels)
    anchor = payload["base_anchor"][b].float()
    center = anchor[:, :2] * box_scale[:, :2]
    valid = (payload["valid_token_mask"][b] & fully_valid_cells(target.valid_pixels, cells) &
             torch.isfinite(center).all(-1) & (center >= 0).all(-1) & (center < 1).all(-1))
    selected = torch.zeros(n, dtype=torch.bool, device=device)
    selected[payload["topk_indices"][b]] = True
    available = valid & ~selected
    owner = torch.full((n,), -1, dtype=torch.long, device=device)
    best_coverage = cells.new_full((n,), -1)
    best_distance = cells.new_full((n,), float("inf"))
    claimants = torch.zeros(n, dtype=torch.long, device=device)
    per_gt = []
    coverages, distances = {}, {}
    # GT ID is the final ownership tie break. This same owner is used by BOTH
    # tasks. Assignment never depends on logits, learned centers, or IoU scores.
    gt_order = torch.argsort(target.gt_ids, stable=True).tolist()
    for j in gt_order:
        box = target.boxes[j].float()
        reliable = bool(target.reliable[j])
        valid_box = bool(torch.isfinite(box).all() and (box[2:] > 0).all())
        mask = target.masks[j]
        nonempty_mask = bool(torch.isfinite(mask).all() and mask.bool().any())
        entry = None
        if options.diagnostics:
            entry = {
                "gt_index": j, "gt_id": int(target.gt_ids[j]),
                "source": target.source, "reliable": reliable,
                "valid_box": valid_box, "nonempty_mask": nonempty_mask,
                "area_pixels": float(box[2:].prod() * target.image_size[0] * target.image_size[1]) if valid_box else None,
                "mask_area_pixels": int(mask.bool().sum()),
                "n_valid": int(valid.sum()), "n_geometry": 0,
                "n_cls_before_topq_exclusion": 0, "n_reg_before_topq_exclusion": 0,
                "n_cls_eligible": 0, "n_reg_eligible": 0,
                "n_cls_owned": 0, "n_reg_owned": 0,
                "n_cls_selected": 0, "n_reg_selected": 0,
                "cls_token_indices": [], "reg_token_indices": [],
                "cls_coverage": [], "reg_coverage": [],
            }
            entry["fill_ratio"] = entry["mask_area_pixels"] / entry["area_pixels"] if valid_box else None
            per_gt.append(entry)
        if not (reliable and valid_box and nonempty_mask):
            if entry is not None:
                entry["skip_reason"] = "unreliable_mask" if not reliable else ("invalid_box" if not valid_box else "empty_mask")
            continue
        # Coverage numerator counts foreground only on valid image pixels.
        coverage = rectangle_coverage((mask.bool() & target.valid_pixels)[None], cells)[0]
        distance = (((center - box[:2]) / box[2:]) ** 2).sum(-1).to(torch.float64)
        inside = ((center >= box[:2] - box[2:] / 2) &
                  (center < box[:2] + box[2:] / 2)).all(-1)
        geometry = valid & inside
        if options.reg_selection == "quality_iou_boxonly":
            # Q-BoxOnly ablation: the GT-mask coverage gate is the only removed
            # constraint. Box containment, validity, padding and Top-Q exclusion
            # still apply; the IoU ranking below is unchanged.
            eligible = available & inside
        else:
            eligible = available & inside & (coverage >= options.tau_reg)
        claimants += eligible.long()
        better = eligible & ((coverage > best_coverage) |
                              ((coverage == best_coverage) & (distance < best_distance)))
        owner[better] = j
        best_coverage[better] = coverage[better]
        best_distance[better] = distance[better]
        coverages[j], distances[j] = coverage, distance
        if entry is not None:
            entry.update({
                "n_geometry": int(geometry.sum()),
                "n_cls_before_topq_exclusion": int((geometry & (coverage >= options.tau_cls)).sum()),
                "n_reg_before_topq_exclusion": int((geometry & (coverage >= options.tau_reg)).sum()),
                "n_cls_eligible": int((available & inside & (coverage >= options.tau_cls)).sum()),
                "n_reg_eligible": int(eligible.sum()),
            })
    cls_p, cls_j, reg_p, reg_j = [], [], [], []
    entry_by_gt = {entry["gt_index"]: entry for entry in per_gt}
    for j in gt_order:
        if j not in coverages:
            continue
        entry = entry_by_gt.get(j)
        coverage, distance = coverages[j], distances[j]
        high = torch.nonzero((owner == j) & (coverage >= options.tau_cls), as_tuple=True)[0]
        low = torch.nonzero((owner == j) & (coverage >= options.tau_reg), as_tuple=True)[0]
        high = _stable_rank(high, coverage, distance, keys)
        cls_indices = high[:options.k_cls]
        transition = torch.nonzero((owner == j) & (coverage >= options.tau_reg) &
                                   (coverage < options.tau_cls), as_tuple=True)[0]
        transition = _stable_rank(transition, coverage, distance, keys)
        if options.reg_selection == "core_plus_transition":
            # User-confirmed F-v2: reuse the ACTUAL cls set, then up to two
            # medium-coverage cells. Never refill one band from the other.
            reg_indices = torch.cat((cls_indices, transition[:2]))
            counterfactual = cls_indices  # raising tau_reg to tau_cls empties this band
        elif options.reg_selection == "quality_iou":
            # Same pool and quota as the confirmed rule; ranking is by the
            # token's current box-vs-GT IoU instead of mask coverage alone.
            # Ranking quantizes IoU to 1e-4 so FP32 arithmetic noise cannot
            # override the documented distance/lexicographic tie-breaks.
            quality = _token_gt_iou(payload, b, low, box, box_scale)
            ranked = torch.round(quality * 1e4) / 1e4
            local = _stable_rank_quality(torch.arange(low.numel(), device=low.device),
                                         ranked, distance[low], keys[low])
            reg_indices = low[local[:options.k_reg]]
            if entry is not None:
                entry["reg_iou"] = quality[local[:options.k_reg]].tolist()
            counterfactual = high[:options.k_reg]
        elif options.reg_selection == "quality_iou_boxonly":
            # Same IoU ranking as quality_iou, over the mask-free pool
            # (owner == j only): the coverage gate was already dropped above.
            pool = torch.nonzero(owner == j, as_tuple=True)[0]
            quality = _token_gt_iou(payload, b, pool, box, box_scale)
            ranked = torch.round(quality * 1e4) / 1e4
            local = _stable_rank_quality(torch.arange(pool.numel(), device=pool.device),
                                         ranked, distance[pool], keys[pool])
            reg_indices = pool[local[:options.k_reg]]
            if entry is not None:
                entry["reg_iou"] = quality[local[:options.k_reg]].tolist()
            counterfactual = high[:options.k_reg]
        else:
            reg_indices = _stable_rank(low, coverage, distance, keys)[:options.k_reg]
            counterfactual = high[:options.k_reg]
        for task, owned, indices, k, p_list, j_list in (
            ("cls", high, cls_indices, options.k_cls, cls_p, cls_j),
            ("reg", low, reg_indices, options.k_reg, reg_p, reg_j),
        ):
            if entry is not None:
                entry[f"n_{task}_owned"] = owned.numel()
                entry[f"n_{task}_selected"] = indices.numel()
                entry[f"{task}_full_quota"] = indices.numel() == k
                entry[f"{task}_levels"] = keys[indices, 0].tolist()
                entry[f"{task}_token_indices"] = indices.tolist()
                entry[f"{task}_coverage"] = coverage[indices].tolist()
            p_list.append(indices)
            j_list.append(torch.full_like(indices, j))
        if entry is None:
            continue
        cls_set, reg_set = set(entry["cls_token_indices"]), set(entry["reg_token_indices"])
        transition_count = sum(options.tau_reg <= value < options.tau_cls for value in entry["reg_coverage"])
        entry.update({
            "reg_selection": options.reg_selection,
            "n_transition_owned": transition.numel(),
            "n_reg_core_selected": len(reg_set) - transition_count,
            "n_reg_transition_selected": transition_count,
            "cls_shortfall": options.k_cls - len(cls_set),
            "reg_shortfall": options.k_reg - len(reg_set),
            "reg_core_shortfall": options.k_cls - len(cls_set) if options.reg_selection == "core_plus_transition" else None,
            "reg_transition_shortfall": 2 - transition_count if options.reg_selection == "core_plus_transition" else None,
            "cls_subset_reg": cls_set <= reg_set,
            "cls_reg_overlap_fraction": len(cls_set & reg_set) / len(cls_set) if cls_set else None,
            "reg_only_fraction": len(reg_set - cls_set) / len(reg_set) if reg_set else None,
            "reg_transition_count": transition_count,
            "reg_transition_fraction": transition_count / len(reg_set) if reg_set else None,
            "reg_high_threshold_token_indices": counterfactual.tolist(),
            "reg_counterfactual_definition": "same policy with tau_reg=tau_cls; F-v2 has no cross-band refill",
            "reg_threshold_changes_selection": set(counterfactual.tolist()) != reg_set,
            "both_tasks_empty": not (cls_set or reg_set),
        })
        entry["per_level"] = []
        for level in range(len(payload["spatial_shapes"])):
            on_level = keys[:, 0] == level
            geometry = valid & ((center >= target.boxes[j, :2] - target.boxes[j, 2:] / 2) &
                                (center < target.boxes[j, :2] + target.boxes[j, 2:] / 2)).all(-1)
            entry["per_level"].append({
                "level": level, "n_geometry": int((geometry & on_level).sum()),
                "n_cls_eligible": int((geometry & available & on_level & (coverage >= options.tau_cls)).sum()),
                "n_reg_eligible": int((geometry & available & on_level & (coverage >= options.tau_reg)).sum()),
                "n_cls_owned": int(((owner == j) & on_level & (coverage >= options.tau_cls)).sum()),
                "n_reg_owned": int(((owner == j) & on_level & (coverage >= options.tau_reg)).sum()),
            })
    def collect(tokens, owners):
        return TaskAssignment(torch.cat(tokens), torch.cat(owners)) if tokens else _empty_task(device)
    audit = {}
    if options.diagnostics:
        audit = {
            "n_gt": gt_count, "n_original_gt": target.original_gt_count,
            "n_valid": int(valid.sum()),
            "n_available": int(available.sum()), "n_conflicting_tokens": int((claimants > 1).sum()),
            "n_claimed_tokens": int((claimants > 0).sum()), "per_gt": per_gt,
            "padding_policy": options.padding_policy, "geometry": options.geometry,
            "reg_selection": options.reg_selection,
            "per_level_valid": [
                {"level": level, "n_valid": int((valid & (keys[:, 0] == level)).sum()),
                 "n_available": int((available & (keys[:, 0] == level)).sum())}
                for level in range(len(payload["spatial_shapes"]))
            ],
        }
    return ImageAssignment(collect(cls_p, cls_j), collect(reg_p, reg_j), box_scale, audit)


def assign_scda(payload, targets, options):
    options.validate()
    if len(targets) != payload["raw_cls"].shape[0]:
        raise ValueError("SCDA targets and payload have different batch sizes")
    return [assign_image(payload, b, target, options) for b, target in enumerate(targets)]
