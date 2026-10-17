"""Experiment C1: same-token cross-model counterfactual for SCDA regression.

Question: is the H1-A gap (Clean A non-top best IoU 0.4001 vs F-Reg 0.6811)
attributable to (1) same-token regression improvement, or (2) changed
candidate selection?

Both models (Clean A / F-Reg-Quality best@4674, LR=1e-4 seed42) run the
IDENTICAL fixed augmented train samples from the H1-A harness. For every
reliable GT the quality_iou eligibility pool is recomputed exactly (the
assigner's own geometry helpers are imported, not reimplemented); eligibility
differs between models ONLY through top-Q membership. Paired per-token IoU
deltas (F-Reg minus Clean A) are computed on
  - fixed Clean-A-chosen top-4 token indices  (pure regression view)
  - fixed F-Reg-chosen top-4 token indices    (pure regression view)
  - each model's own top-4                    (selection + regression view)
  - the UNION of both models' eligible pools  (tagged with pool membership,
    class rank, feature level, mask coverage, top-Q flags)
GTs without any eligible candidate are reported separately.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/ouc1/code/work2_polyp_det/work2_polyp_det")
from harness_diag import (CLEAN_A_BEST, FREG_BEST, build_model, json_dump,
                          load_weights, map_fixed_records, register_dataset, seed_all)

OUT = HERE
BATCH = 8
DETAIL_LIMIT = 150      # GTs whose full per-token tables are kept
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def quantiles(vals):
    if not vals:
        return None
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"n": len(s), "mean": sum(s) / len(s), "p10": q(0.10), "p25": q(0.25),
            "p50": q(0.50), "p75": q(0.75), "p90": q(0.90),
            "frac_improved": sum(v > 0 for v in s) / len(s)}


def pool_of(payload, b, target, options):
    """Exact quality_iou per-GT eligibility (owner arbitration included)."""
    from methods.scda_proposal.scda_assigner import (fully_valid_cells,
                                                     rectangle_coverage, token_geometry)
    cells, box_scale, keys = token_geometry(payload, b, target)
    anchor = payload["base_anchor"][b].float()
    center = anchor[:, :2] * box_scale[:, :2]
    valid = (payload["valid_token_mask"][b] & fully_valid_cells(target.valid_pixels, cells) &
             torch.isfinite(center).all(-1) & (center >= 0).all(-1) & (center < 1).all(-1))
    selected = torch.zeros(cells.shape[0], dtype=torch.bool, device=cells.device)
    selected[payload["topk_indices"][b]] = True
    available = valid & ~selected
    owner = torch.full((cells.shape[0],), -1, dtype=torch.long, device=cells.device)
    best_cov = cells.new_full((cells.shape[0],), -1.0)
    best_dist = cells.new_full((cells.shape[0],), float("inf"))
    geom = {}
    for j in torch.argsort(target.gt_ids, stable=True).tolist():
        box = target.boxes[j].float()
        if not (bool(target.reliable[j]) and torch.isfinite(box).all() and (box[2:] > 0).all()):
            continue
        mask = target.masks[j]
        if not (torch.isfinite(mask).all() and mask.bool().any()):
            continue
        coverage = rectangle_coverage((mask.bool() & target.valid_pixels)[None], cells)[0]
        distance = (((center - box[:2]) / box[2:]) ** 2).sum(-1).to(torch.float64)
        inside = ((center >= box[:2] - box[2:] / 2) &
                  (center < box[:2] + box[2:] / 2)).all(-1)
        better = (available & inside & (coverage >= options.tau_reg) &
                  ((coverage > best_cov) | ((coverage == best_cov) & (distance < best_dist))))
        owner[better] = j
        best_cov[better] = coverage[better]
        best_dist[better] = distance[better]
        geom[j] = coverage
    pools = {j: torch.nonzero((owner == j) & (coverage >= options.tau_reg), as_tuple=True)[0]
             for j, coverage in geom.items()}
    return pools, {"box_scale_all": box_scale.cpu(), "keys_all": keys.cpu()}


def token_gt_iou(raw_box, box_scale, tokens, gt_box):
    """Exact assigner convention (scda_assigner._token_gt_iou, fp32)."""
    from methods.scda_proposal.scda_loss import cxcywh_to_xyxy
    p = cxcywh_to_xyxy(raw_box[tokens].float() * box_scale[tokens])
    t = cxcywh_to_xyxy(gt_box.float()[None].expand(tokens.numel(), 4))
    inter = (torch.minimum(p[:, 2:], t[:, 2:]) -
             torch.maximum(p[:, :2], t[:, :2])).clamp(min=0).prod(-1)
    ap = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(-1)
    at = (t[:, 2:] - t[:, :2]).clamp(min=0).prod(-1)
    return inter / (ap + at - inter).clamp(min=torch.finfo(torch.float32).tiny)


def class_ranks(scores_all, tokens):
    order = torch.argsort(scores_all, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=order.device)
    return rank[tokens]


def snapshot_model(model, records, indices, options):
    from methods.scda_proposal.scda_assigner import assign_scda
    model.train()
    out = []
    with torch.no_grad():
        for start in range(0, len(indices), BATCH):
            chunk = [records[i] for i in indices[start:start + BATCH]]
            from detectron2.structures import ImageList
            imgs = [r["image"].to(model.device) for r in chunk]
            imgs = [(x - model.pixel_mean) / model.pixel_std for x in imgs]
            images = ImageList.from_tensors(imgs, model.size_divisibility)
            gt_instances = [r["instances"].to(model.device) for r in chunk]
            targets = model.prepare_targets(gt_instances, images)
            features = model.backbone(images.tensor)
            outputs, _ = model.sem_seg_head(features, targets=targets)
            payload = outputs.pop("scda_payload")
            aux = model._scda_targets(chunk, gt_instances, targets, images)
            assignments = assign_scda(payload, aux, options)
            for b in range(len(chunk)):
                pools, geometry = pool_of(payload, b, aux[b], options)
                reg = assignments[b].reg
                out.append({
                    "img": int(indices[start + b]),
                    "scores": torch.sigmoid(payload["raw_cls"][b].float()).max(-1)[0].cpu(),
                    "raw_boxes": payload["raw_box_sigmoid"][b].cpu(),
                    "topk": payload["topk_indices"][b].cpu(),
                    "geometry": geometry,
                    "gt_boxes": aux[b].boxes.cpu(),
                    "pools": {j: t.cpu() for j, t in pools.items()},
                    "chosen": {int(g): reg.token_indices[reg.gt_indices == g].cpu()
                               for g in torch.unique(reg.gt_indices).tolist()},
                })
    return out


def main():
    seed_all(42)
    model, cfg, experiment = build_model()
    from methods.scda_proposal.config import SCDAOptions
    options = SCDAOptions()
    dicts = register_dataset(experiment)
    print(f"mapping {len(dicts)} train images (deterministic seed 42)...", flush=True)
    records = map_fixed_records(cfg, experiment, dicts)
    indices = list(range(len(records)))[:LIMIT] if LIMIT else list(range(len(records)))

    snaps = {}
    for label, ckpt in [("CleanA", CLEAN_A_BEST), ("FReg", FREG_BEST)]:
        load_weights(model, ckpt)
        snaps[label] = snapshot_model(model, records, indices, options)
        print(f"{label}: {len(snaps[label])} images snapshotted", flush=True)

    by_img = {lbl: {r["img"]: r for r in rs} for lbl, rs in snaps.items()}
    same_topq = diff_topq = 0
    union_records, common_records = [], []
    per_level = {}
    fixed = {"cleanA_chosen": [], "freg_chosen": []}
    own_top4 = {"CleanA": [], "FReg": []}
    no_pool = []
    details = []
    detail_left = DETAIL_LIMIT
    nearmiss = []

    for r_ca in snaps["CleanA"]:
        r_fr = by_img["FReg"][r_ca["img"]]
        if torch.equal(r_ca["topk"], r_fr["topk"]):
            same_topq += 1
        else:
            diff_topq += 1
        scale_all = r_ca["geometry"]["box_scale_all"]
        for j, t_ca in r_ca["pools"].items():
            t_fr = r_fr["pools"].get(j)
            gt = r_ca["gt_boxes"][j]
            if (t_fr is None or t_fr.numel() == 0) and t_ca.numel() == 0:
                no_pool.append({"img": r_ca["img"], "gt": int(j)})
                continue
            if t_fr is None or t_fr.numel() == 0:
                t_fr = torch.empty(0, dtype=torch.long)
            t_ca = t_ca.to(torch.long)
            t_fr = t_fr.to(torch.long)
            union = torch.cat([t_ca, t_fr[~torch.isin(t_fr, t_ca)]])
            in_ca = torch.isin(union, t_ca)
            in_fr = torch.isin(union, t_fr)
            iou_ca = token_gt_iou(r_ca["raw_boxes"], scale_all, union, gt)
            iou_fr = token_gt_iou(r_fr["raw_boxes"], scale_all, union, gt)
            delta = (iou_fr - iou_ca).tolist()
            common = in_ca & in_fr
            union_records.append({"img": r_ca["img"], "gt": int(j), "deltas": delta,
                                  "n_common": int(common.sum()),
                                  "iou_ca": [round(v, 4) for v in iou_ca.tolist()],
                                  "iou_fr": [round(v, 4) for v in iou_fr.tolist()]})
            if common.any():
                common_records.append({"img": r_ca["img"], "gt": int(j),
                                       "deltas": [d for d, c in zip(delta, common.tolist()) if c]})
            lev = r_ca["geometry"]["keys_all"][union, 0].tolist()
            for lv in set(lev):
                per_level.setdefault(int(lv), []).extend(
                    d for d, l in zip(delta, lev) if l == lv)
            rank_fr = class_ranks(r_fr["scores"], union)
            for t_i in range(len(delta)):
                if delta[t_i] >= 0.1 and not bool(in_fr[t_i]):
                    nearmiss.append({
                        "img": r_ca["img"], "gt": int(j), "token": int(union[t_i]),
                        "delta": round(delta[t_i], 4),
                        "iou_ca": round(float(iou_ca[t_i]), 4),
                        "iou_fr": round(float(iou_fr[t_i]), 4),
                        "rank_fr": int(rank_fr[t_i]), "level": int(lev[t_i])})
            if detail_left > 0:
                details.append({
                    "img": r_ca["img"], "gt": int(j), "tokens": union.tolist(),
                    "level": lev, "in_ca_pool": in_ca.tolist(), "in_fr_pool": in_fr.tolist(),
                    "in_topq_ca": torch.isin(union, r_ca["topk"]).tolist(),
                    "in_topq_fr": torch.isin(union, r_fr["topk"]).tolist(),
                    "rank_ca": class_ranks(r_ca["scores"], union).tolist(),
                    "rank_fr": rank_fr.tolist(),
                    "iou_ca": [round(v, 4) for v in iou_ca.tolist()],
                    "iou_fr": [round(v, 4) for v in iou_fr.tolist()]})
                detail_left -= 1

    # fixed-index views and own-choice views
    for snap_lbl in ("CleanA", "FReg"):
        for snap in snaps[snap_lbl]:
            scale_all = snap["geometry"]["box_scale_all"]
            for j, chosen in snap["chosen"].items():
                if chosen.numel() == 0:
                    continue
                gt = snap["gt_boxes"][j]
                sc = token_gt_iou(snap["raw_boxes"], scale_all, chosen, gt)
                own_top4[snap_lbl].append({
                    "img": snap["img"], "gt": int(j), "tokens": chosen.tolist(),
                    "iou_own": [round(v, 4) for v in sc.tolist()]})
    for view, snap_lbl in (("cleanA_chosen", "CleanA"), ("freg_chosen", "FReg")):
        for e in own_top4[snap_lbl]:
            r_ca, r_fr = by_img["CleanA"][e["img"]], by_img["FReg"][e["img"]]
            toks = torch.tensor(e["tokens"], dtype=torch.long)
            gt = r_ca["gt_boxes"][e["gt"]]
            fixed[view].append({
                "img": e["img"], "gt": e["gt"], "tokens": e["tokens"],
                "iou_ca": [round(v, 4) for v in token_gt_iou(r_ca["raw_boxes"], scale_all, toks, gt).tolist()],
                "iou_fr": [round(v, 4) for v in token_gt_iou(r_fr["raw_boxes"], scale_all, toks, gt).tolist()]})

    def paired(view):
        d = [f - c for e in fixed[view] for c, f in zip(e["iou_ca"], e["iou_fr"])]
        ca = [v for e in fixed[view] for v in e["iou_ca"]]
        fr = [v for e in fixed[view] for v in e["iou_fr"]]
        return {"n_gt": len(fixed[view]), "delta": quantiles(d),
                "iou_ca": quantiles(ca), "iou_fr": quantiles(fr)}

    all_d = [d for u in union_records for d in u["deltas"]]
    common_d = [d for s in common_records for d in s["deltas"]]
    summary = {
        "n_images": len(snaps["CleanA"]),
        "topq_membership": {"same": same_topq, "different": diff_topq},
        "n_gt_with_empty_pool": len(no_pool),
        "union_pool": {"n_gt": len(union_records),
                       "delta_freg_minus_cleana": quantiles(all_d),
                       "iou_ca_mean": (quantiles([v for u in union_records for v in u["iou_ca"]]) or {}) and
                       sum(v for u in union_records for v in u["iou_ca"]) /
                       max(1, sum(len(u["iou_ca"]) for u in union_records)),
                       "iou_fr_mean": sum(v for u in union_records for v in u["iou_fr"]) /
                       max(1, sum(len(u["iou_fr"]) for u in union_records))},
        "common_pool": {"n_gt": len(common_records),
                        "delta_freg_minus_cleana": quantiles(common_d)},
        "delta_by_level": {str(lv): quantiles(v) for lv, v in sorted(per_level.items())},
        "fixed_cleanA_chosen": paired("cleanA_chosen"),
        "fixed_freg_chosen": paired("freg_chosen"),
        "own_top4_best": {lbl: quantiles([max(e["iou_own"]) for e in v]) for lbl, v in own_top4.items()},
        "near_miss": {"n": len(nearmiss),
                      "rank_fr_median": (lambda r: sorted(r)[len(r) // 2] if r else None)(
                          [e["rank_fr"] for e in nearmiss]),
                      "examples": sorted(nearmiss, key=lambda e: -e["delta"])[:20]},
    }
    json_dump(OUT / "c1_summary.json", summary)
    json_dump(OUT / "c1_details.json", {"details": details, "fixed": fixed,
                                        "own_top4": own_top4, "no_pool": no_pool})
    print(json.dumps(summary, indent=1, ensure_ascii=False)[:3000])
    print("C1_DONE")


if __name__ == "__main__":
    main()
