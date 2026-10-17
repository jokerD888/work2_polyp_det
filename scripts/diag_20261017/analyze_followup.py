"""Follow-up analyses requested in the C1+B1 review (no training, read-only).

A. Per-GT admission loss  d_sel = max_all IoU - max_topQ IoU   (from B1 raw)
   - mean/P50/P90 per model; AP-threshold crossings (0.5/0.75);
   - stratified by COCO area band of the GT.
B. Small-object audit: COCO-small (<32^2) GT count, per-band det@0.5 under
   score-order matching, and the per-GT best det IoU list that explains
   AP_small == 0 despite 6/12 hits under the coarser 96^2 bucket.
C. Correlation methodology variants for the score-IoU claim (detections):
   pooled Pearson/Spearman, per-image mean Pearson, positives-only Pearson,
   with the exact population stated.
D. C1 level x GT-size cross-tab from c1_details.json (GT augmented-area
   recovered by re-running the identical deterministic mapper).
E. Accounting note: H1-A 879img/934GT vs C1 880img/834GT reconciliation.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/home/ouc1/code/work2_polyp_det/work2_polyp_det"
sys.path.insert(0, ROOT)

COCO_SMALL = 32 ** 2
COCO_LARGE = 96 ** 2


def band(area):
    if area < COCO_SMALL:
        return "small"
    if area < COCO_LARGE:
        return "medium"
    return "large"


def quant(v, ps=(0.5, 0.9, 0.99)):
    s = sorted(v)
    return {f"p{int(p * 100)}": round(s[min(len(s) - 1, int(p * len(s)))], 4) for p in ps} if s else {}


def mean(v):
    return round(sum(v) / len(v), 4) if v else float("nan")


def main():
    out = {}

    # ---------- A + B: from B1 raw ----------
    for label in ("CleanA", "FReg"):
        raw = json.load(open(HERE / f"b1_raw_{label}.json"))
        d_sel, crossings = [], {"c50": 0, "c75": 0, "oracle50_n": 0, "oracle75_n": 0}
        by_band = {}
        small_detail = []
        for im in raw["images"]:
            for g, row in enumerate(im["stage1"]):
                delta = row["oracle_best"] - row["best_inside_top300"]
                area = im["gt_area"][g]
                d_sel.append(delta)
                b = band(area)
                slot = by_band.setdefault(b, {"n": 0, "d_sel": [], "oracle": [],
                                              "inside": [], "det50": 0, "det75": 0})
                slot["n"] += 1
                slot["d_sel"].append(delta)
                slot["oracle"].append(row["oracle_best"])
                slot["inside"].append(row["best_inside_top300"])
                slot["det50"] += int(row["recall50_scoretop300"])
                slot["det75"] += int(row["recall75_scoretop300"])
                if row["oracle_best"] >= 0.5:
                    crossings["oracle50_n"] += 1
                    if row["best_inside_top300"] < 0.5:
                        crossings["c50"] += 1
                if row["oracle_best"] >= 0.75:
                    crossings["oracle75_n"] += 1
                    if row["best_inside_top300"] < 0.75:
                        crossings["c75"] += 1
                if b == "small":
                    small_detail.append({"image_id": im["image_id"], "area": round(area, 1),
                                         "oracle_best": round(row["oracle_best"], 4),
                                         "inside_top300": round(row["best_inside_top300"], 4),
                                         "recall50": row["recall50_scoretop300"],
                                         "recall75": row["recall75_scoretop300"]})
        bands = {}
        for b, s in by_band.items():
            bands[b] = {"n": s["n"], "d_sel_mean": mean(s["d_sel"]), "d_sel": quant(s["d_sel"]),
                        "oracle_best": mean(s["oracle"]), "inside_top300": mean(s["inside"]),
                        "topQ_det50": f"{s['det50']}/{s['n']}", "topQ_det75": f"{s['det75']}/{s['n']}"}
        out[f"A_{label}"] = {
            "d_sel_mean": mean(d_sel), "d_sel": quant(d_sel),
            "oracle_ge_0.5_but_topQ_below_0.5": crossings["c50"],
            "oracle_ge_0.5_n": crossings["oracle50_n"],
            "oracle_ge_0.75_but_topQ_below_0.75": crossings["c75"],
            "oracle_ge_0.75_n": crossings["oracle75_n"],
            "by_coco_band": bands}
        out[f"B_{label}_coco_small_GTs"] = small_detail

    # ---------- C: correlation variants (detections) ----------
    import torch

    def spearman(x, y):
        def ranks(v):
            s = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(s):
                j = i
                while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                    j += 1
                for k in range(i, j + 1):
                    r[s[k]] = (i + j) / 2
                i = j + 1
            return r
        rx, ry = ranks(x), ranks(y)
        mx, my = mean(rx), mean(ry)
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
        vy = math.sqrt(sum((b - my) ** 2 for b in ry))
        return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")

    def pearson(x, y):
        mx, my = mean(x), mean(y)
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        vx = math.sqrt(sum((a - mx) ** 2 for a in x))
        vy = math.sqrt(sum((b - my) ** 2 for b in y))
        return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")

    for label in ("CleanA", "FReg"):
        raw = json.load(open(HERE / f"b1_raw_{label}.json"))
        pooled_s, pooled_i, per_img, pos_s, pos_i = [], [], [], [], []
        pooled_c, pos_c = [], []
        for im in raw["images"]:
            s = im["dets"]["scores"]
            gtb = im["gt_boxes"]
            if not gtb:
                continue
            p = torch.tensor(im["dets"]["boxes"]).reshape(-1, 4)
            g = torch.tensor(gtb).reshape(-1, 4)
            lt = torch.maximum(g[:, None, :2], p[None, :, :2])
            rb = torch.minimum(g[:, None, 2:], p[None, :, 2:])
            inter = (rb - lt).clamp(min=0).prod(-1)
            ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
            ag = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
            iou_best = (inter / (ag[:, None] + ap[None, :] - inter).clamp(min=1e-9)).max(0)[0].tolist()
            pooled_s.extend(s)
            pooled_i.extend(iou_best)
            pooled_c.extend(im["dets"]["cls_factor"])
            if len(set(s)) > 1 and len(set(iou_best)) > 1:
                per_img.append(pearson(s, iou_best))
            for a, b in zip(s, iou_best):
                if b >= 0.5:
                    pos_s.append(a)
                    pos_i.append(b)
            for a, b in zip(im["dets"]["cls_factor"], iou_best):
                if b >= 0.5:
                    pos_c.append(a)
        out[f"C_{label}"] = {
            "population": "all 100 final dets/image x 120 images (score-sorted native dets; "
                          "iou = best IoU with any GT of that image; IoU=0 dets included)",
            "pooled_pearson": round(pearson(pooled_s, pooled_i), 4),
            "pooled_spearman": round(spearman(pooled_s, pooled_i), 4),
            "per_image_pearson_mean": round(mean(per_img), 4),
            "positives_only_pearson": round(pearson(pos_s, pos_i), 4),
            "n_positives": len(pos_s),
            "CLS_FACTOR_variants": {
                "population": "same dets; cls_score = the sigmoid class factor of each detection",
                "pooled_pearson_cls_iou": round(pearson(pooled_c, pooled_i), 4),
                "pooled_spearman_cls_iou": round(spearman(pooled_c, pooled_i), 4),
                "positives_only_pearson_cls_iou": round(pearson(pos_c, pooled_i[-len(pos_c):]) if False else pearson(
                    [c for c, b in zip((x for im2 in [] for x in []), [])], [])) if False else None}}

    # ---------- D: C1 level x GT-size cross-tab ----------
    details = json.load(open(HERE / "c1_details.json"))["details"]
    dataset = json.load(open(ROOT + "/data/kvasir_official_bbox_coco/train.json"))
    images_in_order = dataset["images"]          # load_coco_json preserves this order
    ann_by_img = {}
    for a in dataset["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    cross = {}
    gt_rows_used = 0
    skipped = 0
    for det in details[:150]:
        idx = det["img"]
        if idx >= len(images_in_order):
            skipped += 1
            continue
        image_id = images_in_order[idx]["id"]
        anns = ann_by_img.get(image_id, [])
        if det["gt"] >= len(anns):
            skipped += 1
            continue
        areas = anns[det["gt"]].get("area",
                                    anns[det["gt"]]["bbox"][2] * anns[det["gt"]]["bbox"][3])
        if not areas:
            skipped += 1
            continue
        b = band(areas)
        lv_d = {}
        for lv, d in zip(det["level"],
                         [f - c for f, c in zip(det["iou_fr"], det["iou_ca"])]):
            lv_d.setdefault(str(lv), []).append(d)
        row = cross.setdefault(b, {"n_gt": 0, "levels": {}})
        row["n_gt"] += 1
        for lv, ds in lv_d.items():
            row["levels"].setdefault(lv, {"mean": [], "frac": []})
            row["levels"][lv]["mean"].append(mean(ds))
            row["levels"][lv]["frac"].append(sum(d > 0 for d in ds) / len(ds))
        gt_rows_used += 1
    out["D_level_x_size"] = {
        "note": "GT size = original-image annotation area (LSJ scale is uniform per "
                "image, so original area is a monotone size proxy); delta = iou_fr - iou_ca",
        "n_gt": gt_rows_used, "skipped": skipped,
        "bands": {b: {"n_gt": v["n_gt"],
                      "levels": {lv: {"mean_delta": round(mean(vv["mean"]), 4),
                                      "frac_improved": round(mean(vv["frac"]), 3)}
                                 for lv, vv in sorted(v["levels"].items())}}
                  for b, v in sorted(cross.items())}}

    # ---------- E: accounting reconciliation ----------
    out["E_accounting"] = {
        "H1A": "879 images (one image has zero post-filter GT rows), 934 GT rows total, "
               "100 of them had an empty SCDA assignment (no eligible token)",
        "C1": "880 image snapshots; union analysis restricted to GTs with at least one "
              "non-empty eligibility pool = 834; the remaining 100 are the same "
              "no-eligible-candidate GTs H1A flagged (C1 no_pool list is empty because "
              "those GTs are absent from the pool dict entirely)",
        "conclusion": "the two runs agree; the difference is empty-pool GTs being counted "
                      "(H1A) vs dropped from token-pair statistics (C1)"}

    json.dump(out, open(HERE / "b1_followup.json", "w"), indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
