"""Experiment B1 analyzer: COCO AP identity gates + four-stage aggregates.

Gates (must pass before any conclusion):
  G0  replica==native per-image gates from the collector (all zero)
  G1  COCO AP recomputed from the collector's own native detections equals
      the recorded predict_val metrics AP (3 decimals)
  G1b a second evaluation of the same file reproduces G1 exactly

Then aggregates stages 1-4 for both models and prints a comparison.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = "/home/ouc1/code/work2_polyp_det/work2_polyp_det"
sys.path.insert(0, ROOT)
sys.path.insert(0, str(HERE.parent))
VAL_ANN = "/home/ouc1/code/work2_polyp_det/work2_polyp_det/data/kvasir_official_bbox_coco/val.json"
RECORDED = {"CleanA": {"AP": 66.041, "AP50": 84.082, "AP75": 73.432},
            "FReg": {"AP": 65.571, "AP50": 83.139, "AP75": 74.433}}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


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
    return pearson(ranks(x), ranks(y))


def coco_eval(pred_rows, tag):
    """Returns fraction metrics dict from polypbench.evaluation.evaluate."""
    import tempfile
    from polypbench.evaluation import evaluate
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(pred_rows, f)
        pred_path = f.name
    r = evaluate(VAL_ANN, pred_path)["metrics"]
    Path(pred_path).unlink()
    return r


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def quant(v, ps=(0.1, 0.25, 0.5, 0.75, 0.9)):
    s = sorted(v)
    return {f"p{int(p*100)}": s[min(len(s) - 1, int(p * len(s)))] for p in ps} if s else {}


def main():
    report = {}
    for label in ("CleanA", "FReg"):
        raw = json.load(open(HERE / f"b1_raw_{label}.json"))
        gates = raw["gates"]
        images = raw["images"]
        dets = raw["detections"]

        # ---- gates ----
        ap_rec = coco_eval(dets, "first")
        ap_rec2 = coco_eval(dets, "second")
        rec = RECORDED[label]
        g1_ok = all(abs(round(ap_rec[k] * 100, 3) - rec[k]) < 5e-3 for k in ("AP", "AP50", "AP75"))
        gates_out = {
            "G0_replica_zero": (gates["max_replica_diff"] == 0 and
                                gates["max_det_box_diff"] == 0 and
                                gates["max_det_score_diff"] == 0),
            "G1_ap_identity": {"recomputed": {k: round(ap_rec[k] * 100, 3) for k in
                                              ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")},
                               "recorded": rec, "ok": g1_ok},
            "G1b_repeat_identical": ap_rec == ap_rec2,
        }

        # ---- stage 1: encoder coverage ----
        s1 = {"n_gt": 0, "oracle_best": [], "oracle_rank": [], "oracle_in_top300": 0,
              "score_top": {f"top{K}": [] for K in (1, 10, 50, 100, 300)},
              "recall50_score300": 0, "recall75_score300": 0,
              "best_inside": [], "best_outside": [], "outside_better": 0,
              "lowrank_good": 0, "gt_with_good_outside_only": 0, "levels": {}}
        for im in images:
            for row in im["stage1"]:
                s1["n_gt"] += 1
                s1["oracle_best"].append(row["oracle_best"])
                s1["oracle_rank"].append(row["oracle_best_rank"])
                s1["oracle_in_top300"] += int(row["oracle_best_in_top300"])
                for K in (1, 10, 50, 100, 300):
                    s1["score_top"][f"top{K}"].append(row[f"score_top{K}_best"])
                s1["recall50_score300"] += int(row["recall50_scoretop300"])
                s1["recall75_score300"] += int(row["recall75_scoretop300"])
                s1["best_inside"].append(row["best_inside_top300"])
                s1["best_outside"].append(row["best_outside_top300"])
                s1["outside_better"] += int(row["best_outside_top300"] > row["best_inside_top300"])
                if row["oracle_best"] >= 0.5 and row["oracle_best_rank"] >= 300:
                    s1["lowrank_good"] += 1
                if row["best_inside_top300"] < 0.5 <= row["best_outside_top300"]:
                    s1["gt_with_good_outside_only"] += 1
                lv = row["oracle_best_level"]
                s1["levels"].setdefault(lv, []).append(row["oracle_best"])

        # ---- stage 2: E -> M -> D per query ----
        s2 = {"n_gt": 0, "bestE": [], "bestM": [], "bestD": [],
              "pairedM_E": [], "argmaxE_eq_M": 0, "argmaxM_eq_D": 0,
              "ret": {(t): [0, 0, 0, 0] for t in (0.3, 0.5, 0.7)},
              "D_given_lowM": [], "lowM_n": 0}
        for im in images:
            for g in range(len(im["gt_boxes"])):
                e = im["iouE300"][g]
                m = im["iouM"][g]
                d = im["iouD"][g]
                if not e:
                    continue
                s2["n_gt"] += 1
                be, bm, bd = max(e), max(m), max(d)
                s2["bestE"].append(be)
                s2["bestM"].append(bm)
                s2["bestD"].append(bd)
                s2["pairedM_E"].extend(m[k] - e[k] for k in range(len(e)))
                ae, am, ad = e.index(be), m.index(bm), d.index(bd)
                s2["argmaxE_eq_M"] += int(ae == am)
                s2["argmaxM_eq_D"] += int(am == ad)
                for t, acc in s2["ret"].items():
                    if e[ae] >= t:
                        acc[0] += 1
                        acc[1] += int(m[ae] >= t)
                        acc[2] += int(d[ae] >= t)
                    if m[am] >= t:
                        acc[3] += int(d[am] >= t)
                lows = [k for k, v in enumerate(e) if v < 0.3]
                s2["lowM_n"] += len(lows)
                s2["D_given_lowM"].extend(d[k] for k in lows)

        # ---- stage 3: score decomposition vs IoU ----
        s3 = {"n_det": 0, "score": [], "cls": [], "maskq": [], "iou_best": [],
              "matched_iou": [], "corr_score_iou": None, "corr_cls_iou": None,
              "corr_maskq_iou": None, "corr_cls_maskq": None,
              "quad_good_lowconf": 0, "quad_conf_badloc": 0, "n_pos": 0}
        for im in images:
            gtb = im["gt_boxes"]
            d = im["dets"]
            ious = iou_best_per_det(d["boxes"], gtb)
            for k in range(len(d["scores"])):
                s3["n_det"] += 1
                s3["score"].append(d["scores"][k])
                s3["cls"].append(d["cls_factor"][k])
                s3["maskq"].append(d["mask_q"][k])
                s3["iou_best"].append(ious[k])
            for sc, iou in zip(d["scores"], ious):
                if iou >= 0.5:
                    s3["n_pos"] += 1
                    s3["matched_iou"].append(iou)
                    if sc < 0.3:
                        s3["quad_good_lowconf"] += 1
                if sc >= 0.5 and iou < 0.3:
                    s3["quad_conf_badloc"] += 1
        s3["corr_score_iou"] = pearson(s3["score"], s3["iou_best"])
        s3["corr_cls_iou"] = pearson(s3["cls"], s3["iou_best"])
        s3["corr_maskq_iou"] = pearson(s3["maskq"], s3["iou_best"])
        s3["corr_cls_maskq"] = pearson(s3["cls"], s3["maskq"])
        s3["spearman_score_iou"] = spearman(s3["score"], s3["iou_best"])

        # ---- stage 4: COCO-matched error taxonomy ----
        s4 = {"gt_n": 0, "det50": 0, "det75": 0, "miss50": 0, "miss75": 0,
              "loc_only_fail": 0, "dups50": 0, "by_area": {}}
        for im in images:
            gtb = im["gt_boxes"]
            areas = im["gt_area"]
            d = im["dets"]
            ious = iou_matrix_full(d["boxes"], gtb)   # (G, N)
            if ious is None:
                continue
            for g in range(len(gtb)):
                s4["gt_n"] += 1
                q = area_quart(areas[g])
                slot = s4["by_area"].setdefault(q, {"n": 0, "det50": 0, "det75": 0})
                slot["n"] += 1
            used = set()
            for k in range(len(d["scores"])):  # COCO-like: score order, one det per GT
                for g in iou_argsort_desc(ious[:, k]) if len(gtb) else []:
                    if g in used:
                        continue
                    if ious[g, k] >= 0.5:
                        used.add(g)
                        s4["det50"] += 1
                        s4["by_area"][area_quart(areas[g])]["det50"] += 1
                        if ious[g, k] >= 0.75:
                            s4["det75"] += 1
                            s4["by_area"][area_quart(areas[g])]["det75"] += 1
                        else:
                            s4["loc_only_fail"] += 1
                    break
            hit = set()
            for k in range(len(d["scores"])):
                for g in range(len(gtb)):
                    if ious[g, k] >= 0.5:
                        hit.add(g)
            for g in range(len(gtb)):
                n_match = sum(1 for k in range(len(d["scores"])) if ious[g, k] >= 0.5)
                if n_match >= 2:
                    s4["dups50"] += 1
                if g not in hit:
                    s4["miss50"] += 1
            for g in range(len(gtb)):
                if max((ious[g, k] for k in range(len(d["scores"]))), default=0) < 0.75:
                    s4["miss75"] += 1
        report[label] = {"gates": gates_out,
                         "stage1": {k: (mean(v) if isinstance(v, list) else v)
                                    for k, v in s1.items() if k not in ("score_top", "levels")},
                         "stage1_score_top": {k: mean(v) for k, v in s1["score_top"].items()},
                         "stage1_levels": {k: round(mean(v), 4) for k, v in s1["levels"].items()},
                         "stage2": {"n_gt": s2["n_gt"],
                                    "bestE": mean(s2["bestE"]), "bestM": mean(s2["bestM"]),
                                    "bestD": mean(s2["bestD"]),
                                    "pairedM_E_mean": mean(s2["pairedM_E"]),
                                    "pairedM_E": quant(s2["pairedM_E"]),
                                    "argmaxE_eq_M": s2["argmaxE_eq_M"] / s2["n_gt"],
                                    "argmaxM_eq_D": s2["argmaxM_eq_D"] / s2["n_gt"],
                                    "retention": {str(t): {"P(M>=t|E>=t)": a[1] / max(a[0], 1),
                                                           "P(D>=t|E>=t)": a[2] / max(a[0], 1),
                                                           "P(D>=t|M>=t)": a[3] / max(a[0], 1)}
                                                  for t, a in s2["ret"].items()},
                                    "D_given_lowM_mean": mean(s2["D_given_lowM"]),
                                    "lowM_n": s2["lowM_n"]},
                         "stage3": {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in s3.items()
                                    if k not in ("score", "cls", "maskq", "iou_best", "matched_iou")},
                         "stage3_dist": {"score_p50": quant(s3["score"])["p50"],
                                         "iou_best_p50": quant(s3["iou_best"])["p50"],
                                         "matched_iou_mean": mean(s3["matched_iou"])},
                         "stage4": {k: v for k, v in s4.items() if k != "by_area"},
                         "stage4_by_area": s4["by_area"]}
        json.dump(report[label], open(HERE / f"b1_analysis_{label}.json", "w"), indent=1)
        print(f"===== {label} =====")
        print(json.dumps(report[label], indent=1)[:3500])
    json.dump(report, open(HERE / "b1_analysis.json", "w"), indent=1)
    print("B1_ANALYZE_DONE")


def iou_matrix_full(det_boxes, gt_boxes):
    """(G, N) IoU matrix; None when either side is empty."""
    import torch
    if not gt_boxes or not det_boxes:
        return None
    p = torch.tensor(det_boxes).reshape(-1, 4)
    g = torch.tensor(gt_boxes).reshape(-1, 4)
    lt = torch.maximum(g[:, None, :2], p[None, :, :2])
    rb = torch.minimum(g[:, None, 2:], p[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    ag = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    return (inter / (ag[:, None] + ap[None, :] - inter).clamp(min=1e-9))


def iou_best_per_det(det_boxes, gt_boxes):
    if not gt_boxes:
        return [0.0] * len(det_boxes)
    import torch
    p = torch.tensor(det_boxes).reshape(-1, 4)
    g = torch.tensor(gt_boxes).reshape(-1, 4)
    lt = torch.maximum(g[:, None, :2], p[None, :, :2])
    rb = torch.minimum(g[:, None, 2:], p[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    ag = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    return (inter / (ag[:, None] + ap[None, :] - inter).clamp(min=1e-9)).max(0)[0].tolist()


def iou_argsort_desc(row):
    import torch
    return torch.argsort(torch.tensor(row), descending=True).tolist()


def area_quart(a):
    if a < 96 ** 2:
        return "small"
    if a < 160 ** 2:
        return "medium"
    return "large"


if __name__ == "__main__":
    main()
