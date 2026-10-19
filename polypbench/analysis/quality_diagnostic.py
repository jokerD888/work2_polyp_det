"""Read-only source-val quality diagnosis. All GT-assisted scores are diagnostic.

The main analysis conditions on unique GT compatibility at IoU >= .5 and
meaningful IoU differences >= .05. It does not equate GT-assisted AP with the
attainable benefit of a learned quality head. No model is trained here.
"""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np

from polypbench.io import read_json, write_json
from .coco_cache import official_eval, ap_vector
from .matching import iou_matrix, xyxy, maximum_coverage


THRESHOLDS = (.5, .75, .9, .95)
VARIANTS = ("native", "class_only", "same_gt_permutation", "image_gt_permutation", "direct_gt_iou")
GAP = .05
SCORE_EPS = 1e-12


def owners(matrix, floor=.5):
    """Assign only uniquely compatible predictions; ambiguous boxes stay out."""
    matrix = np.asarray(matrix)
    result = np.full(matrix.shape[1], -1, dtype=int)
    if not matrix.shape[0]:
        return result
    unique = (matrix >= floor).sum(0) == 1
    result[unique] = matrix[:, unique].argmax(0)
    return result


def pair_stats(quality, scores, gap=GAP):
    quality, scores = np.asarray(quality), np.asarray(scores)
    comparable = quality[:, None] - quality[None, :] >= gap - 1e-12
    score_delta = scores[:, None] - scores[None, :]
    total = int(comparable.sum())
    inverted = int((comparable & (score_delta < -SCORE_EPS)).sum())
    tied = int((comparable & (np.abs(score_delta) <= SCORE_EPS)).sum())
    return {"pairs": total, "inversions": inverted, "score_ties": tied,
            "inversion_fraction": inverted / total if total else None}


def same_gt_rows(matrix, scores, annotation_ids, floor=.5):
    assigned = owners(matrix, floor)
    rows = []
    for i, aid in enumerate(annotation_ids):
        ids = np.flatnonzero(assigned == i)
        q, s = matrix[i, ids], scores[ids]
        top = int(np.argmax(s)) if len(ids) else None
        best = float(q.max()) if len(ids) else None
        row = {"annotation_id": aid, "group_candidates": len(ids), **pair_stats(q, s),
               "best_group_iou": best, "top_score_group_iou": float(q[top]) if top is not None else None,
               "top_score_iou_gap": best - float(q[top]) if top is not None else None,
               "top_score_candidate_index": int(ids[top]) if top is not None else None,
               "best_iou_candidate_index": int(ids[int(np.argmax(q))]) if len(ids) else None,
               "top_score_tie_count": int((s == s[top]).sum()) if top is not None else 0}
        for threshold in (.75, .9):
            high = q >= threshold
            low = q < threshold
            comparable = high[:, None] & low[None, :] & ((q[:, None] - q[None, :]) >= GAP - 1e-12)
            row[f"t{threshold:.2f}_cross_pairs"] = int(comparable.sum())
            row[f"t{threshold:.2f}_cross_inversions"] = int((comparable & ((s[:, None] - s[None, :]) < -SCORE_EPS)).sum())
            row[f"t{threshold:.2f}_has_candidate"] = bool(high.any())
            row[f"t{threshold:.2f}_top_score_misses"] = bool(high.any() and q[top] < threshold)
        rows.append(row)
    return rows


def permute_scores(scores, quality, groups=None):
    """Keep the exact score multiset. Quality ties retain native score order."""
    scores = np.asarray(scores)
    result = scores.copy()
    if groups is None:
        groups = [np.arange(len(scores))]
    for ids in groups:
        ids = np.asarray(ids, dtype=int)
        native_order = ids[np.argsort(-scores[ids], kind="stable")]
        order = native_order[np.argsort(-quality[native_order], kind="stable")]
        result[order] = np.sort(scores[ids])[::-1]
    return result


def assisted_scores(matrix, scores, floor=.5):
    assigned = owners(matrix, floor)
    max_iou = matrix.max(0) if matrix.shape[0] else np.zeros(len(scores))
    groups = [np.flatnonzero(assigned == i) for i in range(matrix.shape[0])]
    return {
        "native": scores.copy(),
        "same_gt_permutation": permute_scores(scores, max_iou, groups),
        "image_gt_permutation": permute_scores(scores, max_iou) if matrix.shape[0] else scores.copy(),
        "direct_gt_iou": max_iou,
    }


def aggregate(rows):
    comparable = [r for r in rows if r["pairs"]]
    multi = [r for r in rows if r["group_candidates"] >= 2]
    gaps = [r["top_score_iou_gap"] for r in multi]
    pairs = sum(r["pairs"] for r in rows)
    result = {
        "all_gt": len(rows), "gt_with_group": sum(r["group_candidates"] > 0 for r in rows),
        "gt_with_multiple_candidates": len(multi), "gt_with_comparable_pairs": len(comparable),
        "pairs": pairs, "inversions": sum(r["inversions"] for r in rows),
        "score_ties": sum(r["score_ties"] for r in rows),
        "micro_inversion_fraction": sum(r["inversions"] for r in rows) / pairs if pairs else None,
        "macro_inversion_fraction": float(np.mean([r["inversion_fraction"] for r in comparable])) if comparable else None,
        "gt_with_any_inversion": sum(r["inversions"] > 0 for r in comparable),
        "gt_top_score_gap_ge_005": sum(g >= GAP - 1e-12 for g in gaps),
        "top_score_gap_mean": float(np.mean(gaps)) if gaps else None,
        "top_score_gap_median": float(np.median(gaps)) if gaps else None,
        "top_score_gap_p90": float(np.quantile(gaps, .9)) if gaps else None,
    }
    for threshold in (.75, .9):
        for suffix in ("cross_pairs", "cross_inversions", "has_candidate", "top_score_misses"):
            key = f"t{threshold:.2f}_{suffix}"
            result[key] = sum(r[key] for r in rows)
    return result


def fp_types(ev, gt):
    """Use official greedy COCO matches, then label unmatched predictions."""
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    result = {}
    for threshold in THRESHOLDS:
        tid = int(round((threshold - .5) / .05))
        by_k = {}
        for k in (1, 10, 100):
            counts = dict(tp=0, fp_duplicate_compatible=0, fp_localization=0,
                          fp_low_overlap=0, fp_background=0, ignored=0,
                          ambiguous_compatible_fp=0)
            for e in ev.evalImgs:
                if e is None or e["aRng"] != ev.params.areaRng[0]:
                    continue
                truth = xyxy([a["bbox"] for a in anns[e["image_id"]]])
                for j, did in enumerate(e["dtIds"][:k]):
                    if e["dtIgnore"][tid, j]:
                        counts["ignored"] += 1
                    elif e["dtMatches"][tid, j] > 0:
                        counts["tp"] += 1
                    else:
                        box = xyxy([ev.cocoDt.anns[did]["bbox"]])
                        overlaps = iou_matrix(truth, box).ravel()
                        best = float(overlaps.max()) if len(overlaps) else 0.
                        if best >= threshold:
                            counts["fp_duplicate_compatible"] += 1
                            counts["ambiguous_compatible_fp"] += int((overlaps >= threshold).sum() > 1)
                        elif best >= .5:
                            counts["fp_localization"] += 1
                        elif best >= .1:
                            counts["fp_low_overlap"] += 1
                        else:
                            counts["fp_background"] += 1
            by_k[str(k)] = counts
        result[f"{threshold:.2f}"] = by_k
    return result


def save_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output):
    out = Path(output)
    protocol = read_json(out / "protocol.json")
    gt = read_json(protocol["annotations"])
    if len(gt["categories"]) != 1 or any(a.get("iscrowd", 0) or a.get("ignore", 0) for a in gt["annotations"]):
        raise ValueError("This diagnostic requires a single class and ordinary nonignored GT")
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    images = {i["id"]: i for i in gt["images"]}
    result = {"cohort": {"images": len(images), "gt": len(gt["annotations"]),
                          "negative_images": sum(not anns[i] for i in images),
                          "multi_gt_images": sum(len(anns[i]) > 1 for i in images)},
              "rules": protocol["rules"], "models": {}}
    detail, coverage_rows, examples = [], [], []
    for model, training in protocol["training"].items():
        arrays = np.load(out / "score_replay" / model / "score_components.npz", allow_pickle=False)
        if set(arrays["image_ids"].tolist()) != set(images):
            raise ValueError("Replay cohort differs")
        variants = {k: [] for k in VARIANTS}
        group_rows = {"native": [], "class_only": []}
        sensitivity = []
        ambiguity = {"ambiguous_predictions": 0, "unassigned_predictions": 0}
        cov = {f"{t:.2f}": defaultdict(int) for t in THRESHOLDS}
        for index, iid in enumerate(arrays["image_ids"].tolist()):
            aa = anns[iid]
            boxes = arrays["bbox"][index].astype(float)
            scores = arrays["final_scores"][index].astype(float)
            classes = arrays["class_scores"][index].astype(float)
            query_ids = arrays["query_ids"][index].astype(int)
            all_boxes = arrays["all_query_boxes"][index].astype(float)
            all_classes = arrays["all_query_class_scores"][index].astype(float)
            if boxes.shape != (100, 4) or all_boxes.shape != (300, 4) or len(set(query_ids)) != 100:
                raise ValueError("Unexpected query shapes or identities")
            np.testing.assert_allclose(boxes, all_boxes[query_ids], atol=1e-3, rtol=1e-5)
            np.testing.assert_allclose(classes, all_classes[query_ids], atol=1e-7, rtol=1e-5)
            if all_classes[np.setdiff1d(np.arange(300), query_ids)].max() > classes.min() + 1e-7:
                raise ValueError("Native selection is not classification Top100")
            truth = xyxy([a["bbox"] for a in aa])
            matrix = iou_matrix(truth, xyxy(boxes))
            all_matrix = iou_matrix(truth, xyxy(all_boxes))
            owner = owners(matrix)
            ambiguity["ambiguous_predictions"] += int(((matrix >= .5).sum(0) > 1).sum())
            ambiguity["unassigned_predictions"] += int((owner < 0).sum())
            score_values = assisted_scores(matrix, scores)
            score_values["class_only"] = classes
            # All variants keep every box; there is no threshold filtering/NMS.
            for variant, values in score_values.items():
                for box, score in zip(boxes, values):
                    variants[variant].append({"image_id": iid, "category_id": gt["categories"][0]["id"],
                                               "bbox": box.tolist(), "score": float(score)})
            for scoring, values in (("native", scores), ("class_only", classes)):
                rows = same_gt_rows(matrix, values, [a["id"] for a in aa])
                for j, row in enumerate(rows):
                    row.update(model=model, scoring=scoring, image_id=iid, file_name=images[iid]["file_name"],
                               image_gt_count=len(aa), size="small" if aa[j]["area"] < 1024 else "medium" if aa[j]["area"] < 9216 else "large",
                               best_iou100=float(matrix[j].max()), best_iou300=float(all_matrix[j].max()))
                    group_rows[scoring].append(row)
                    detail.append(row)
                    if scoring == "native":
                        ex = {k: row[k] for k in ("model", "annotation_id", "image_id", "file_name", "top_score_iou_gap", "best_iou300")}
                        ex.update(gt_box=aa[j]["bbox"])
                        for key, field in (("top_score_box", "top_score_candidate_index"), ("best_group_box", "best_iou_candidate_index")):
                            c = row[field]
                            ex[key] = boxes[c].tolist() if c is not None else None
                            ex[key + "_score"] = float(scores[c]) if c is not None else None
                        examples.append(ex)
            sensitivity.extend(same_gt_rows(matrix, scores, [a["id"] for a in aa], floor=.25))
            for threshold in THRESHOLDS:
                counts = {"all300": maximum_coverage(all_matrix, threshold),
                          "selected100": maximum_coverage(matrix, threshold)}
                for scoring in ("native", "class_only", "same_gt_permutation", "image_gt_permutation"):
                    order = np.argsort(-score_values[scoring], kind="stable")
                    for k in (1, 10):
                        counts[f"{scoring}_top{k}"] = maximum_coverage(matrix[:, order[:k]], threshold)
                for k, v in counts.items():
                    cov[f"{threshold:.2f}"][k] += v
                coverage_rows.append({"model": model, "image_id": iid, "gt": len(aa), "threshold": threshold, **counts})
        ap, duplicate = {}, {}
        for variant in VARIANTS:
            ev = official_eval(gt, variants[variant])
            vector = ap_vector(ev)
            ap[variant] = {"AP": float(vector.mean()), "AP50": float(vector[0]), "AP75": float(vector[5]),
                           "AP90": float(vector[8]), "AP95": float(vector[9]), "AP_by_IoU": vector.tolist()}
            duplicate[variant] = fp_types(ev, gt)
            write_json(out / "predictions" / f"{model}_{variant}.json", variants[variant])
        original = read_json(Path(training["run_root"]) / "predict_val/metrics.json")
        for key in ("AP", "AP50", "AP75"):
            if abs(ap["native"][key] - original["metrics"][key]) > 1e-12:
                raise ValueError(f"Original COCO metric mismatch: {model}/{key}")
        total = len(gt["annotations"])
        coverage = {}
        for threshold, counts in cov.items():
            counts = dict(counts)
            coverage[threshold] = {"counts": counts, "fractions": {k: v / total for k, v in counts.items()},
                                   "no_all300_coverage_count": total - counts["all300"],
                                   "lost_at_selection_count": counts["all300"] - counts["selected100"]}
        result["models"][model] = {
            "same_gt": {k: aggregate(v) for k, v in group_rows.items()},
            "same_gt_by_cohort": {scoring: {"single_gt_image": aggregate([r for r in rows if r["image_gt_count"] == 1]),
                                           "multi_gt_image": aggregate([r for r in rows if r["image_gt_count"] > 1]),
                                           **{size: aggregate([r for r in rows if r["size"] == size]) for size in ("small", "medium", "large")}}
                                  for scoring, rows in group_rows.items()},
            "same_gt_floor025_sensitivity": aggregate(sensitivity), "ownership": ambiguity,
            "coverage": coverage, "ap": ap, "official_fp_types": duplicate,
            "original_native_metric_parity": "passed",
        }
    save_csv(out / "per_gt.csv", detail)
    save_csv(out / "per_image_coverage.csv", coverage_rows)
    write_json(out / "diagnostics.json", result)
    write_json(out / "case_candidates.json", examples)
    report(out, result)
    plot(out, result)
    return result


def report(out, result):
    c = result["cohort"]
    lines = ["# Kvasir Val：质量错排、Query筛选与GT辅助排序诊断", "",
             f"只读重放Baseline/MSSL各{c['images']}张图像；{c['gt']}个GT，{c['multi_gt_images']}张多GT图，{c['negative_images']}张纯负图。",
             "源域Val已参与选权，本报告是探索性开发诊断；没有训练、更新模型、目标域分析或Quality Head。", "",
             "## Q1：同一GT的高IoU候选是否被低质量候选压过？", "",
             "主诊断：仅纳入IoU≥0.5且只兼容一个GT的Top100候选；多GT歧义候选不硬分组。",
             "比较IoU差≥0.05的候选对；低质量框分数严格更高才计错排，同分另报。",
             "Micro按候选对加权，Macro按有可比候选对的GT平均。组内最高分IoU差仅条件于存在至少两个候选，不代表全体GT。", "",
             "| 模型/评分 | 可比GT | 可比候选对 | 错排对 | Micro错排 | Macro错排 | 最高分IoU差≥.05的GT/多候选GT |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, d in result["models"].items():
        for scoring, s in d["same_gt"].items():
            pct = lambda x: f"{100*x:.2f}%" if x is not None else "NA"
            lines.append(f"| {model}/{scoring} | {s['gt_with_comparable_pairs']} | {s['pairs']} | {s['inversions']} | {pct(s['micro_inversion_fraction'])} | {pct(s['macro_inversion_fraction'])} | {s['gt_top_score_gap_ge_005']}/{s['gt_with_multiple_candidates']} |")
    lines += ["", "按单/多GT、尺寸分组及IoU≥.25敏感性见diagnostics.json；逐GT分母与边界跨越错排见per_gt.csv。", "",
              "## Q2：Top100筛选前后的损失", "", "覆盖采用IoU阈值约束的最大基数一对一匹配，分母包含全部GT。",
              "全部300和选定100的覆盖差对应可见的筛选损失；Top10覆盖只描述前排检索，不等价于AP损失。", "",
              "| 模型 | IoU | 全300覆盖GT | 原Top100覆盖GT | 筛选丢失 | 全300仍未覆盖 | 原融合Top10覆盖GT |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, d in result["models"].items():
        for threshold, v in d["coverage"].items():
            n = v["counts"]
            lines.append(f"| {model} | {threshold} | {n['all300']} | {n['selected100']} | {v['lost_at_selection_count']} | {v['no_all300_coverage_count']} | {n['native_top10']} |")
    lines += ["", "## 固定Top100的评分参照", "",
              "same_gt_permutation只在上述同GT合理候选组内交换原融合分数，不动歧义/背景候选的分数；仍依赖GT条件分组。",
              "image_gt_permutation在图像内按最大GT IoU分配原分数集合，同时含前景识别和质量排序信息；负图不变。",
              "direct_gt_iou直接使用最大GT IoU，负图全零；包含不可部署的GT信息，不是严格最优AP上界。",
              "所有参照保留全部原框、不加阈值/NMS；COCO照常处理重复检测。不能把AP差值视为可学习头的预期收益。", "",
              "| 模型 | 评分 | AP | AP75 | AP90 | AP95 |", "|---|---|---:|---:|---:|---:|"]
    for model, d in result["models"].items():
        for variant, v in d["ap"].items():
            numbers = " | ".join(f"{100*v[k]:.3f}" for k in ("AP", "AP75", "AP90", "AP95"))
            lines.append(f"| {model} | {variant} | {numbers} |")
    lines += ["", "## 重复检测与其他FP", "",
              "按官方COCO分数贪心一对一匹配计TP/FP；重复兼容FP指未匹配但与至少一个GT达到当前IoU阈值。",
              "该标签表示该阈值下的重复/匹配竞争，不是人工确认同一病灶。下表是各图前10个候选，不使用固定置信度过滤。", "",
              "| 模型/评分 | IoU | TP | 重复兼容FP | 定位FP(.5≤IoU<t) | 低重叠FP(.1≤IoU<.5) | 背景FP(IoU<.1) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for model, d in result["models"].items():
        for scoring in ("native", "same_gt_permutation"):
            for threshold in ("0.75", "0.90"):
                n = d["official_fp_types"][scoring][threshold]["10"]
                lines.append(f"| {model}/{scoring} | {threshold} | {n['tp']} | {n['fp_duplicate_compatible']} | {n['fp_localization']} | {n['fp_low_overlap']} | {n['fp_background']} |")
    lines += ["", "完整Top1/10/100与全部评分参照见diagnostics.json。本数据没有纯负图，不能估计负图误报率。", "",
              "## Q3：是否启动质量对齐研究？", "",
              "应结合主诊断中的GT数量、错排程度、组内重排收益、候选缺失及重复检测解释，由人工复核形成decision.md。",
              "不预设AP门槛，不把最大IoU错排相关性当作因果分解；源域120张无患者分组、单Seed，不输出稳定性或精确置信结论。",
              "若证据不清，允许保持V1、不新增模块。候选缺失本身不能证明低对比，也不能据此直接选局部对比。", "",
              "## 复核入口", "", "- protocol.json：读取数据前固定的规则、输入/源码哈希、模型身份。",
              "- score_replay/*/verification.json：每张图像原生输出与归档预测的核对。",
              "- diagnostics.json、per_gt.csv、per_image_coverage.csv：完整计数、分母和诊断。",
              "- figures/coverage_and_gaps.png：覆盖与组内最高分IoU差。",
              "- predictions/：固定框的各评分参照，供官方COCO复算。", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot(out, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(result["models"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(models))
    for offset, pool, label in ((-.2, "all300", "All 300"), (0, "selected100", "Selected 100"), (.2, "native_top10", "Native top 10")):
        axes[0].bar(x + offset, [100 * result["models"][m]["coverage"]["0.90"]["fractions"][pool] for m in models], .2, label=label)
    axes[0].set_xticks(x, models)
    axes[0].set_ylabel("Coverage at IoU >= .90 (%)")
    axes[0].legend()
    for model in models:
        rows = [r for r in read_json(out / "case_candidates.json") if r["model"] == model and r["top_score_iou_gap"] is not None]
        gaps = np.sort([r["top_score_iou_gap"] for r in rows])
        axes[1].plot(gaps, np.arange(1, len(gaps) + 1) / len(gaps), label=model)
    axes[1].axvline(.05, color="grey", ls="--")
    axes[1].set_xlabel("Best IoU - top-score IoU within unique-GT group")
    axes[1].set_ylabel("Conditional empirical CDF (nonempty groups)")
    axes[1].legend()
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures/coverage_and_gaps.png", dpi=160)
    plt.close(fig)
