"""H1-A: stage-wise box quality on fixed training images, no training.

For each checkpoint (Clean A best, F-Reg-Quality best; both LR=1e-4 seed42):
run the payload-instrumented model in train mode (DN active, matching the
native forward), and per reliable GT record best/mean IoU of
  E_top300 : encoder raw proposals selected by the native top-300
  E_scda   : proposals the quality_iou SCDA rule assigns (<=4 per GT)
  M_init   : mask2box decoder initialization of the same 300 queries
  D_final  : decoder final boxes (all 300 queries and score-top subsets)
Same-query transitions E->M->D are tracked per query slot and per GT.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from harness_diag import (CLEAN_A_BEST, FREG_BEST, ROOT, build_model, cxcywh_to_xyxy_px,
                          json_dump, load_weights, make_batches, map_fixed_records,
                          pairwise_iou_best, register_dataset, build_experiment, seed_all)

OUT = Path(__file__).parent
BATCH = 8
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all 880


def score_per_query(logits):
    return torch.sigmoid(logits.float()).max(-1)[0]


def measure(model, records, options, limit):
    from methods.scda_proposal.scda_assigner import assign_scda
    model.train()
    indices = list(range(len(records)))[:limit] if limit else list(range(len(records)))
    per_gt = []
    per_query_all = []
    n_empty_assignment = 0
    n_images = 0
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
            outputs, mask_dict = model.sem_seg_head(features, targets=targets)
            payload = outputs.pop("scda_payload")
            aux_targets = model._scda_targets(chunk, gt_instances, targets, images)
            assignments = assign_scda(payload, aux_targets, options)

            raw = payload["raw_box_sigmoid"]            # B,N,4 normalized cxcywh
            topk = payload["topk_indices"]              # B,300
            init_boxes = payload["decoder_init_boxes"]  # B,300,4
            final_boxes = outputs["pred_boxes"]         # B,300,4
            final_logits = outputs["pred_logits"]       # B,300,C
            _, _, h_pad, w_pad = images.tensor.shape

            for b in range(len(chunk)):
                target = aux_targets[b]
                gt_px = cxcywh_to_xyxy_px(target.boxes, w_pad, h_pad)  # G,4
                if gt_px.shape[0] == 0:
                    continue
                assignment = assignments[b]
                scale = assignment.box_scale
                iou_raw = pairwise_iou_best(cxcywh_to_xyxy_px(raw[b] * scale, w_pad, h_pad), gt_px)
                iou_init = pairwise_iou_best(cxcywh_to_xyxy_px(init_boxes[b], w_pad, h_pad), gt_px)
                iou_final = pairwise_iou_best(cxcywh_to_xyxy_px(final_boxes[b], w_pad, h_pad), gt_px)
                scores = score_per_query(final_logits[b])            # 300
                tk = topk[b]
                # per-GT stats
                for g in range(gt_px.shape[0]):
                    entry = {"image_idx": int(indices[start + b]),
                             "gt": g, "reliable": bool(target.reliable[g])}
                    e300 = iou_raw[g, tk]
                    entry["E_top300_best"] = float(e300.max())
                    entry["E_top300_mean"] = float(e300.mean())
                    sel = assignment.reg.token_indices[assignment.reg.gt_indices == g]
                    if sel.numel():
                        e_scda = iou_raw[g, sel]
                        entry["E_scda_best"] = float(e_scda.max())
                        entry["E_scda_mean"] = float(e_scda.mean())
                        entry["E_scda_k"] = int(sel.numel())
                    else:
                        entry["E_scda_best"] = None
                        entry["E_scda_k"] = 0
                        n_empty_assignment += 1
                    entry["M_init_best"] = float(iou_init[g].max())
                    entry["M_init_mean"] = float(iou_init[g].mean())
                    entry["D_final_best"] = float(iou_final[g].max())
                    entry["D_final_mean"] = float(iou_final[g].mean())
                    for topk_n in (10, 50, 100):
                        idx = scores.topk(min(topk_n, scores.numel())).indices
                        entry[f"D_scoretop{topk_n}_best"] = float(iou_final[g, idx].max())
                    # which query wins per stage (same-query tracking)
                    entry["argmax_E"] = int(e300.argmax())
                    entry["argmax_M"] = int(iou_init[g].argmax())
                    entry["argmax_D"] = int(iou_final[g].argmax())
                    per_gt.append(entry)
                # per-query arrays (query slot i <-> encoder proposal topk[i])
                per_query_all.append({
                    "iouE": iou_raw[:, tk].float().cpu(),      # G,300
                    "iouM": iou_init.float().cpu(),
                    "iouD": iou_final.float().cpu(),
                    "score": scores.float().cpu(),
                })
                n_images += 1
    return per_gt, per_query_all, n_empty_assignment, n_images


def summarize(per_gt, per_query_all, n_empty, n_images, label):
    def field_stats(key):
        vals = [e[key] for e in per_gt if e.get(key) is not None]
        if not vals:
            return None
        vals = sorted(vals)
        q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]
        return {"mean": sum(vals) / len(vals), "p25": q(0.25), "p50": q(0.5),
                "p75": q(0.75), "n": len(vals), "frac_ge_0.5": sum(v >= 0.5 for v in vals) / len(vals)}

    summary = {"label": label, "n_images": n_images, "n_gt": len(per_gt),
               "n_gt_without_scda_candidates": n_empty,
               "stages": {k: field_stats(k) for k in
                          ["E_top300_best", "E_top300_mean", "E_scda_best", "E_scda_mean",
                           "M_init_best", "M_init_mean", "D_final_best", "D_final_mean",
                           "D_scoretop10_best", "D_scoretop50_best", "D_scoretop100_best"]}}
    # same-query retention: does the query that wins at stage X stay good at the next?
    retention = {}
    for thr in (0.3, 0.5, 0.7):
        m_keep = d_keep_from_m = d_keep_from_e = n_e = n_m = 0
        for pq in per_query_all:
            best_e = pq["iouE"].argmax(1)
            best_m = pq["iouM"].argmax(1)
            for g in range(pq["iouE"].shape[0]):
                i, j = best_e[g], best_m[g]
                if pq["iouE"][g, i] >= thr:
                    n_e += 1
                    m_keep += float(pq["iouM"][g, i] >= thr)
                    d_keep_from_e += float(pq["iouD"][g, i] >= thr)
                if pq["iouM"][g, j] >= thr:
                    n_m += 1
                    d_keep_from_m += float(pq["iouD"][g, j] >= thr)
        retention[f"thr{thr}"] = {
            "P(M>=thr | E>=thr at argmaxE)": m_keep / max(n_e, 1),
            "P(D>=thr | E>=thr at argmaxE)": d_keep_from_e / max(n_e, 1),
            "P(D>=thr | M>=thr at argmaxM)": d_keep_from_m / max(n_m, 1),
            "n_E": n_e, "n_M": n_m}
    summary["same_query_retention"] = retention
    # per-GT argmax consistency across stages
    n = sum(1 for e in per_gt if e["argmax_E"] == e["argmax_M"])
    m = sum(1 for e in per_gt if e["argmax_M"] == e["argmax_D"])
    k = sum(1 for e in per_gt if e["argmax_E"] == e["argmax_D"])
    summary["argmax_query_agreement"] = {"E==M": n / len(per_gt), "M==D": m / len(per_gt),
                                         "E==D": k / len(per_gt)}
    # correlations between per-GT stage-best values
    def corr(key_a, key_b):
        xs = [e[key_a] for e in per_gt if e.get(key_a) is not None and e.get(key_b) is not None]
        ys = [e[key_b] for e in per_gt if e.get(key_a) is not None and e.get(key_b) is not None]
        if len(xs) < 3:
            return None
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        vx = sum((x - mx) ** 2 for x in xs) ** 0.5
        vy = sum((y - my) ** 2 for y in ys) ** 0.5
        return cov / (vx * vy) if vx > 0 and vy > 0 else None
    summary["pearson"] = {"E_best~M_best": corr("E_top300_best", "M_init_best"),
                          "M_best~D_best": corr("M_init_best", "D_final_best"),
                          "E_best~D_best": corr("E_top300_best", "D_final_best"),
                          "E_scda_best~M_best": corr("E_scda_best", "M_init_best")}
    return summary


def main():
    seed_all(42)
    model, cfg, experiment = build_model()
    from methods.scda_proposal.config import SCDAOptions
    options = SCDAOptions()
    dicts = register_dataset(experiment)
    print(f"mapping {len(dicts)} train images (deterministic, seed 42)...", flush=True)
    records = map_fixed_records(cfg, experiment, dicts)
    results = {}
    for label, ckpt in [("Clean-A_1e-4_best4674", CLEAN_A_BEST), ("F-Reg-Quality_1e-4_best4674", FREG_BEST)]:
        info = load_weights(model, ckpt)
        print(f"=== {label}: {info} ===", flush=True)
        per_gt, per_query, n_empty, n_images = measure(model, records, options, LIMIT)
        results[label] = summarize(per_gt, per_query, n_empty, n_images, label)
        json_dump(OUT / f"h1a_pergt_{label}.json", per_gt)
        print(json.dumps(results[label], indent=1)[:2000], flush=True)
    json_dump(OUT / "h1a_results.json", results)
    print("H1A_DONE")


if __name__ == "__main__":
    main()
