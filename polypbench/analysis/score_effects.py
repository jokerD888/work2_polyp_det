"""Diagnostic counterfactual rankings on exactly the original selected boxes."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from polypbench.io import write_json
from .coco_cache import CocoCache,metrics
from .matching import xyxy,iou_matrix,maximum_coverage


def run(gt,predictions,replay_root,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    annotations=defaultdict(list)
    for a in gt["annotations"]: annotations[a["image_id"]].append(a)
    result={"scope":"diagnostic only; native deployment unchanged; class-only scores applied to original class-selected100, with original saved boxes; not full300 re-selection",
            "models":{},"pairs":{}}
    for label,rows in predictions.items():
        parts=np.load(Path(replay_root)/label/"score_components.npz")
        ids=parts["image_ids"]; lookup={int(iid):i for i,iid in enumerate(ids)}
        by_image=defaultdict(list)
        for row in rows: by_image[row["image_id"]].append(row)
        class_only=[]; all_coverage={f"{t:.2f}":0 for t in (.8,.9,.95)}
        class_coverage={f"{t:.2f}":{str(k):0 for k in (1,5,10,100)} for t in (.8,.9,.95)}
        for iid in sorted(lookup):
            i=lookup[iid]; original=by_image[iid]
            np.testing.assert_allclose([r["score"] for r in original],parts["final_scores"][i],rtol=1e-5,atol=1e-6)
            class_only.extend({**r,"score":float(score)} for r,score in zip(original,parts["class_scores"][i]))
            truth=xyxy([a["bbox"] for a in annotations[iid]])
            matrix=iou_matrix(truth,xyxy(parts["all_query_boxes"][i]))
            selected=parts["query_ids"][i]
            selected_order=selected[np.argsort(-parts["class_scores"][i],kind="mergesort")]
            for t in (.8,.9,.95):
                all_coverage[f"{t:.2f}"]+=maximum_coverage(matrix,t)
                for k in (1,5,10,100): class_coverage[f"{t:.2f}"][str(k)]+=maximum_coverage(matrix[:,selected_order],t,k)
        write_json(out/f"{label}.class_only.bbox.json",class_only)
        cls_cache=CocoCache(gt,class_only); native=CocoCache(gt,rows)
        n=len(gt["annotations"])
        result["models"][label]={"native_fraction":metrics(native.reference),"class_only_fraction":metrics(cls_cache.reference),
            "class_only_ap_by_iou_fraction":cls_cache.reference.tolist(),
            "all300_maximum_cardinality_coverage":{k:v/n for k,v in all_coverage.items()},
            "native_selected100_class_order_coverage":{t:{k:v/n for k,v in cc.items()} for t,cc in class_coverage.items()},
            "score_decomposition":"native fused=s_cls*s_mask; components recovered at unchanged native inference; same-box counterfactual is not an additive causal decomposition"}
    for control in ("baseline","box_cdf","mask_static"):
        result["pairs"][control]={mode:{metric:100*(result["models"]["mssl"][mode][metric]-result["models"][control][mode][metric])
                                      for metric in ("AP","AP75","AP90","AP95")}
                                      for mode in ("native_fraction","class_only_fraction")}
    write_json(out/"score_effects.json",result)
    return result
