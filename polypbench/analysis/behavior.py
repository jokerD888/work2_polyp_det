"""Paired detection behavior on the locked cohort, retaining every GT."""
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from polypbench.io import write_json
from .matching import xyxy,iou_matrix,maximum_coverage,score_order

THRESHOLDS=(.5,.75,.8,.85,.9,.95)
TOPK=(1,5,10,100)


def run(gt,predictions,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    images={x["id"]:x for x in gt["images"]}; anns=defaultdict(list)
    for a in gt["annotations"]: anns[a["image_id"]].append(a)
    grouped={label:defaultdict(list) for label in predictions}
    for label,rows in predictions.items():
        for p in rows: grouped[label][p["image_id"]].append(p)
    image_rows=[]; target_rows=[]; assignments={label:{} for label in predictions}
    for iid,im in sorted(images.items()):
        truth=xyxy([a["bbox"] for a in anns[iid]])
        image_row={"image_id":iid,"center":im["center"],"gt":len(truth),"negative":not len(truth)}
        rows_by_gt=[{"annotation_id":a["id"],"image_id":iid,"center":im["center"],
                     "size":"small" if a["area"]<1024 else "medium" if a["area"]<9216 else "large"} for a in anns[iid]]
        for label in predictions:
            rows=score_order(grouped[label][iid])[:100]; boxes=xyxy([p["bbox"] for p in rows])
            scores=np.array([p["score"] for p in rows]); matrix=iou_matrix(truth,boxes)
            image_row[f"{label}_detections_score_ge_05"]=int((scores>=.5).sum())
            image_row[f"{label}_max_score"]=float(scores.max()) if len(scores) else 0.
            for threshold in THRESHOLDS:
                for k in TOPK:
                    image_row[f"{label}_t{threshold:.2f}_k{k}"]=maximum_coverage(matrix,threshold,k)
                # Given the fixed score threshold, remaining misses and duplicate
                # predictions are reported at a diagnostic operating point only.
                selected=scores>=.5
                count=maximum_coverage(matrix[:,selected],threshold)
                image_row[f"{label}_t{threshold:.2f}_max_tp_score05"]=count
                image_row[f"{label}_t{threshold:.2f}_min_fp_score05"]=int(selected.sum())-count
            assigned={}
            if matrix.size:
                rr,cc=linear_sum_assignment(-matrix); assigned=dict(zip(rr.tolist(),cc.tolist()))
            for i,a in enumerate(anns[iid]):
                x,y,w,h=a["bbox"]; j=assigned.get(i)
                assignments[label][a["id"]]={"iou":float(matrix[i,j]) if j is not None else 0.,
                    "edge_error":((boxes[j]-[x,y,x+w,y+h])/np.maximum([w,h,w,h],4)*[1,1,-1,-1]).tolist() if j is not None else None}
                rows_by_gt[i][f"{label}_best_iou100"]=float(matrix[i].max()) if len(rows) else 0.
                for threshold in THRESHOLDS:
                    ranks=np.flatnonzero(matrix[i]>=threshold)
                    rows_by_gt[i][f"{label}_first_rank_t{threshold:.2f}"]=int(ranks[0]+1) if len(ranks) else None
        target_rows.extend(rows_by_gt); image_rows.append(image_row)
    total_gt=sum(r["gt"] for r in image_rows); summaries={}; pairs={}
    cohorts={"pooled":image_rows,**{c:[r for r in image_rows if r["center"]==c] for c in sorted({r["center"] for r in image_rows})}}
    for cohort,rr in cohorts.items():
        denominator=sum(r["gt"] for r in rr); negatives=[r for r in rr if r["negative"]]
        summaries[cohort]={"images":len(rr),"gt":denominator,"negative_images":len(negatives),"models":{}}
        for label in predictions:
            coverage={f"t{t:.2f}_k{k}":sum(r[f"{label}_t{t:.2f}_k{k}"] for r in rr)/denominator for t in THRESHOLDS for k in TOPK}
            summaries[cohort]["models"][label]={"maximum_cardinality_coverage":coverage,
                "negative_frames_with_detection_score05":sum(r[f"{label}_detections_score_ge_05"]>0 for r in negatives),
                "negative_detections_score05":sum(r[f"{label}_detections_score_ge_05"] for r in negatives),
                "fixed_score05":{f"t{t:.2f}":{
                    "max_tp":sum(r[f"{label}_t{t:.2f}_max_tp_score05"] for r in rr),
                    "min_fp":sum(r[f"{label}_t{t:.2f}_min_fp_score05"] for r in rr),
                    "min_fn":denominator-sum(r[f"{label}_t{t:.2f}_max_tp_score05"] for r in rr)} for t in THRESHOLDS}}
    for control in ("baseline","box_cdf","mask_static"):
        pairs[control]={}
        for t in THRESHOLDS:
            key=f"first_rank_t{t:.2f}"; both=[]; only_mssl=only_control=neither=0
            for row in target_rows:
                a,b=row[f"mssl_{key}"],row[f"{control}_{key}"]
                if a is not None and b is not None: both.append((a,b))
                elif a is not None: only_mssl+=1
                elif b is not None: only_control+=1
                else: neither+=1
            pairs[control][f"t{t:.2f}"]={"both_have_individual_candidate":len(both),
                "only_mssl":only_mssl,"only_control":only_control,"neither":neither,
                "conditional_mean_rank_mssl":float(np.mean([a for a,b in both])) if both else None,
                "conditional_mean_rank_control":float(np.mean([b for a,b in both])) if both else None,
                "mssl_rank_better":sum(a<b for a,b in both),"mssl_rank_worse":sum(a>b for a,b in both),
                "rank_equal":sum(a==b for a,b in both),
                "scope":"individual-GT retrieval ranks, not exclusive matching; condition on both having a candidate; selection is descriptive"}
    common=[a["id"] for a in gt["annotations"] if all(assignments[l][a["id"]]["iou"]>=.5 for l in predictions)]
    edge={l:{name:{"mean_abs":float(np.mean([abs(assignments[l][aid]["edge_error"][i]) for aid in common])),
                  "mean_signed_inward":float(np.mean([assignments[l][aid]["edge_error"][i] for aid in common]))}
              for i,name in enumerate(("left","top","right","bottom"))} for l in predictions}
    result={"protocol":{"coverage":"threshold-constrained maximum-cardinality bipartite matching; GT and predictions used at most once",
                         "candidate_pool":"native class-selected 100 detections, ordered by native fused score; not all 300 decoder queries",
                         "fixed_score05":"maximum TP/minimum FP,FN assignment at score>=0.5, distinct from official score-greedy COCO matching",
                         "normalization":"coverage denominator includes all 1533 supplied GT; negative images retained for AP and false alarms"},
            "gt":total_gt,"cohorts":summaries,"paired_individual_ranks":pairs,
            "edge_errors_common_all_four":{"gt":len(common),"normalization":"max(provided box axis size,4px); positive=inward","models":edge}}
    write_json(out/"behavior.json",result)
    for name,rows in (("per_image",image_rows),("per_gt",target_rows)):
        fields=list(rows[0])
        with (out/f"{name}.csv").open("w",encoding="utf-8-sig",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result
