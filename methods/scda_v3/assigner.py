"""Legacy-exact V3-A, or confirmed SDF/scale-stratified V3-B assignment."""
import torch

from methods.scda_proposal.scda_assigner import (
    assign_scda, token_geometry, fully_valid_cells, rectangle_coverage,
    _stable_rank_quality, _token_gt_iou)
from methods.scda_proposal.structures import ImageAssignment, TaskAssignment
from .mask_geometry import mask_sdf, sample_cell_centers, level_preference


def empty_task(device):
    p = torch.empty(0, dtype=torch.long, device=device)
    return TaskAssignment(p, p.clone())


@torch.no_grad()
def assign(payload, targets, options):
    options.validate()
    if len(targets) != payload["raw_cls"].shape[0]:
        raise ValueError("V3 targets and payload batch sizes differ")
    # Never permit a cropped multi-box original to become auxiliary-eligible.
    for target in targets:
        if target.reliable.any() and (len(target.labels) != 1 or target.original_gt_count != 1):
            raise ValueError("V3 only accepts reliable ORIGINAL single-instance images")
    if options.assignment_policy == "legacy_quality_iou":
        result = assign_scda(payload, targets, options.legacy())
        for assignment in result:
            assignment.cls = empty_task(payload["raw_cls"].device)
        return result
    return [assign_multiscale(payload, b, target, options) for b, target in enumerate(targets)]


@torch.no_grad()
def assign_multiscale(payload, b, target, options):
    cells, scale, keys = token_geometry(payload, b, target)
    device = cells.device
    n = len(cells)
    anchor_center = payload["base_anchor"][b, :, :2].float() * scale[:, :2]
    valid = (payload["valid_token_mask"][b] & fully_valid_cells(target.valid_pixels, cells) &
             torch.isfinite(anchor_center).all(-1) & (anchor_center >= 0).all(-1) &
             (anchor_center < 1).all(-1))
    topq = torch.zeros(n, dtype=torch.bool, device=device)
    topq[payload["topk_indices"][b]] = True
    selected, owners, entries = [], [], []
    for j in torch.argsort(target.gt_ids, stable=True).tolist():
        box, mask = target.boxes[j].float(), target.masks[j]
        entry = {"gt_id": int(target.gt_ids[j]), "gt_index": j, "selected": [],
                 "shortfall": options.k_reg, "same_level_refill": 0, "other_level_refill": 0}
        entries.append(entry)
        if not bool(target.reliable[j]):
            entry["skip_reason"] = "unreliable_mask"
            continue
        if not bool(torch.isfinite(box).all() and (box[2:] > 0).all()):
            entry["skip_reason"] = "invalid_box"
            continue
        if not torch.isfinite(mask).all():
            entry["skip_reason"] = "nonfinite_mask"
            continue
        visible = mask.bool() & target.valid_pixels
        if not visible.any():
            entry["skip_reason"] = "empty_visible_mask"
            continue
        coverage = rectangle_coverage(visible[None], cells)[0]
        inside = ((anchor_center >= box[:2] - box[2:] / 2) &
                  (anchor_center < box[:2] + box[2:] / 2)).all(-1)
        pool = torch.nonzero(valid & ~topq & inside & (coverage >= options.tau_reg), as_tuple=True)[0]
        entry["pool_count"] = pool.numel()
        if not pool.numel():
            entry["skip_reason"] = "no_candidates"
            continue
        field, diameter = mask_sdf(mask, target.valid_pixels)
        sdf = sample_cell_centers(field, cells)
        order, scores = level_preference(payload["spatial_shapes"].tolist(),
                                         target.valid_pixels.shape, diameter, options.target_cells)
        level_pools = {level: pool[keys[pool, 0] == level] for level in order}
        preferred = [level for level in order if level_pools[level].numel()][:options.preferred_levels]
        distance = (((anchor_center - box[:2]) / box[2:]) ** 2).sum(-1).double()
        quality = torch.zeros(n, device=device)
        quality[pool] = torch.round(_token_gt_iou(payload, b, pool, box, scale) * 1e4) / 1e4
        def rank(tokens):
            return _stable_rank_quality(tokens, quality, distance, keys)
        picks, labels = [], []
        regions = {}
        region_by_token = torch.zeros(n, dtype=torch.long, device=device)
        for level in order:
            candidates = rank(level_pools[level])
            candidates = candidates[torch.argsort(sdf[candidates], stable=True)]
            cut = (candidates.numel() + 1) // 2
            boundary, core = candidates[:cut], candidates[cut:]
            region_by_token[boundary] = 1
            region_by_token[core] = 2
            regions[level] = {"boundary": boundary, "core": core}
            if level not in preferred:
                continue
            for name in ("core", "boundary"):
                ordered = rank(regions[level][name])
                if ordered.numel():
                    picks.append(int(ordered[0])); labels.append(name)
        chosen_mask = torch.zeros(n, dtype=torch.bool, device=device)
        chosen_mask[picks] = True
        preferred_mask = torch.zeros(n, dtype=torch.bool, device=device)
        for level in preferred:
            preferred_mask[level_pools[level]] = True
        for allowed, label, counter in ((preferred_mask, "same_level_refill", "same_level_refill"),
                                        (~preferred_mask, "other_level_refill", "other_level_refill")):
            remaining = rank(pool[~chosen_mask[pool] & allowed[pool]])[:max(0, options.k_reg - len(picks))]
            picks.extend(remaining.tolist()); labels.extend([label] * remaining.numel())
            chosen_mask[remaining] = True
            entry[counter] = remaining.numel()
        tokens = torch.tensor(picks, device=device, dtype=torch.long)
        selected.append(tokens); owners.append(torch.full_like(tokens, j))
        entry.update(preferred_levels=preferred, scale_order=order, scale_scores=scores,
                     mask_diameter=diameter, selected=picks, regions=labels,
                     levels=keys[tokens, 0].tolist(), sdf=sdf[tokens].tolist(),
                     geometric_regions=["boundary" if value == 1 else "core"
                                        for value in region_by_token[tokens].tolist()],
                     coverage=coverage[tokens].tolist(), iou=quality[tokens].tolist(),
                     shortfall=options.k_reg - len(picks),
                     per_level_pool={str(level): int(len(level_pools[level])) for level in order})
    reg = TaskAssignment(torch.cat(selected), torch.cat(owners)) if selected else empty_task(device)
    return ImageAssignment(empty_task(device), reg, scale,
                           {"assignment_policy": "mask_multiscale", "n_gt": len(target.labels),
                            "n_original_gt": target.original_gt_count, "per_gt": entries})
