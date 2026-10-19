"""Prediction-only paired diagnostics. Geometry associations are exploratory."""
import argparse
from collections import defaultdict
import contextlib
import csv
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from polypbench.io import read_json, sha256, write_json


def xyxy(boxes):
    a=np.asarray(boxes,dtype=float).reshape(-1,4).copy(); a[:,2:]+=a[:,:2]; return a


def ious(a,b):
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(rb-lt,0).prod(2)
    return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+
                            np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None,:]-inter,1e-12)


def threshold_ap(annotations,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(str(annotations)); dt=gt.loadRes(str(predictions)); ev=COCOeval(gt,dt,"bbox")
        ev.params.imgIds=sorted(gt.imgs); ev.params.catIds=sorted(gt.cats); ev.params.maxDets=[1,10,100]
        ev.evaluate(); ev.accumulate()
    result={}
    for t,thr in enumerate(ev.params.iouThrs):
        p=ev.eval["precision"][t,:,:,0,2]; valid=p[p>=0]
        result[f"{thr:.2f}"]=float(valid.mean()*100) if len(valid) else None
    # TP/FP at fixed thresholds complements AP; no oracle causal decomposition.
    counts={}
    for threshold in (.5,.75):
        t=int(np.argmin(abs(ev.params.iouThrs-threshold))); tp=fp=0
        for image in ev.evalImgs:
            if image is None or image["aRng"]!=ev.params.areaRng[0]: continue
            scores=np.asarray(image["dtScores"]); active=(scores>=.5)&~image["dtIgnore"][t]
            matched=image["dtMatches"][t]>0
            tp+=int((active&matched).sum()); fp+=int((active&~matched).sum())
        counts[f"iou_{threshold}"]={"tp_score_ge_05":tp,"fp_score_ge_05":fp}
    return result,counts


def candidates(gt,predictions):
    by_image=defaultdict(list)
    for p in predictions: by_image[p["image_id"]].append(p)
    grouped=defaultdict(list)
    for a in gt["annotations"]: grouped[a["image_id"]].append(a)
    result={}
    for image_id,anns in grouped.items():
        preds=sorted(by_image[image_id],key=lambda x:-x["score"])[:100]
        truth=xyxy([a["bbox"] for a in anns]); boxes=xyxy([p["bbox"] for p in preds])
        matrix=ious(truth,boxes)
        assigned={}
        if len(preds):
            rows,cols=linear_sum_assignment(-matrix)
            assigned=dict(zip(rows.tolist(),cols.tolist()))
        for i,ann in enumerate(anns):
            j=assigned.get(i)
            ranks=np.flatnonzero(matrix[i]>=.75)
            result[ann["id"]]={"assigned_iou":float(matrix[i,j]) if j is not None else 0.,
                "assigned_box":boxes[j].tolist() if j is not None else None,
                "best_iou_100":float(matrix[i].max()) if len(preds) else 0.,
                "best_iou_10":float(matrix[i,:10].max()) if len(preds) else 0.,
                "top1_iou_single_gt":float(matrix[i,0]) if len(anns)==1 and len(preds) else None,
                "first_rank_iou75":int(ranks[0]+1) if len(ranks) else None}
    return result


def geometry(mask,box):
    x,y,w,h=box; ys,xs=np.nonzero(mask)
    if not len(xs) or min(w,h)<=0: return {}
    tight=np.array([[xs.min(),ys.min(),xs.max()+1,ys.max()+1]],float)
    provided=xyxy([box]); maskbox_iou=float(ious(provided,tight)[0,0])
    xx=np.arange(mask.shape[1])+.5; yy=np.arange(mask.shape[0])+.5
    inside_x=(xx>=x)&(xx<x+w); inside_y=(yy>=y)&(yy<y+h)
    inside=mask[np.ix_(inside_y,inside_x)]
    area=float(mask.sum()); contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    perimeter=sum(cv2.arcLength(c,True) for c in contours)
    densities=[]
    for edge,axis in enumerate((0,1,0,1)):
        # A 0.15*box-axis interior band at the provided GT side, pixel centres.
        axis_pos=xx if axis==0 else yy
        edge_value=(x,y,x+w,y+h)[edge]; size=(w,h)[axis]
        dist=(axis_pos-edge_value)*(1 if edge<2 else -1)
        band=(dist>=0)&(dist<.15*size)
        region=mask[np.ix_(inside_y,band)] if axis==0 else mask[np.ix_(band,inside_x)]
        densities.append(float(region.mean()) if region.size else 0.)
    return {"mask_inside_box_fraction":float(inside.sum()/max(w*h,1)),
        "mask_total_to_box_area":area/max(w*h,1), "compactness_4piA_over_P2":float(4*np.pi*area/perimeter**2) if perimeter else None,
        "maskbox_iou":maskbox_iou,"support_density_min":min(densities),
        "support_density_std":float(np.std(densities)),
        **{f"support_density_{name}":v for name,v in zip(("left","top","right","bottom"),densities)}}


def mean(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def associations(rows,baseline,method):
    features=("mask_inside_box_fraction","compactness_4piA_over_P2","support_density_min",
              "support_density_std","maskbox_iou")
    outcome=f"{method}_minus_{baseline}_assigned_iou"
    result={}
    for feature in features:
        usable=[r for r in rows if r.get(feature) is not None and r.get(outcome) is not None]
        if len(usable)<8: continue
        v=np.array([r[feature] for r in usable]); d=np.array([r[outcome] for r in usable])
        controls=np.array([[r["log_box_area"],r["log_aspect"],r[f"{baseline}_assigned_iou"]] for r in usable])
        matrix=np.column_stack([v,controls]); std=matrix.std(0)
        if std[0]<1e-10: continue
        matrix=(matrix-matrix.mean(0))/np.maximum(std,1e-12)
        design=np.column_stack([np.ones(len(matrix)),matrix])
        coef=np.linalg.lstsq(design,d,rcond=None)[0]
        quartile=np.quantile([r["log_box_area"] for r in usable],[.25,.5,.75])
        strata=[]
        for q in range(4):
            rr=[r for r in usable if int(np.searchsorted(quartile,r["log_box_area"],side="right"))==q]
            strata.append({"area_quartile":q+1,"n":len(rr),"mean_delta_iou":mean([r[outcome] for r in rr]),
                           "spearman":float(spearmanr([r[feature] for r in rr],[r[outcome] for r in rr]).statistic)
                           if len(rr)>3 and np.std([r[feature] for r in rr])>0 and np.std([r[outcome] for r in rr])>0 else None})
        rho=float(spearmanr(v,d).statistic) if np.std(d)>0 else None
        result[feature]={"n":len(usable),"spearman":rho,
            "ols_delta_iou_per_feature_sd":float(coef[1]),
            "controls":"log bbox area, log aspect ratio, baseline assigned IoU; descriptive adjustment, not causality",
            "area_strata":strata}
    return result


def run(annotations,predictions,output,mask_root=None,mask_layout="source"):
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    gt=read_json(annotations); images={im["id"]:im for im in gt["images"]}
    counts=defaultdict(int)
    for ann in gt["annotations"]: counts[ann["image_id"]]+=1
    assigned={}; aps={}; detection_counts={}
    for label,path in predictions.items():
        assigned[label]=candidates(gt,read_json(path))
        aps[label],detection_counts[label]=threshold_ap(annotations,path)
    labels=list(predictions); baseline=labels[0]
    rows=[]; mask_hashes={}
    for ann in gt["annotations"]:
        im=images[ann["image_id"]]; x,y,w,h=ann["bbox"]
        area=ann.get("area",w*h)
        r={"annotation_id":ann["id"],"image_id":im["id"],"center":im.get("center","source"),
           "box_area":w*h,"annotation_area":area,"size":"small" if area<32**2 else "medium" if area<96**2 else "large",
           "log_box_area":float(np.log(max(w*h,1))),"log_aspect":float(np.log(max(w,1)/max(h,1)))}
        if mask_root and counts[im["id"]]==1:
            filename=Path(im["file_name"])
            if mask_layout=="polypgen":
                center=im["center"]
                mask_path=Path(mask_root)/f"data_{center}"/f"masks_{center}"/(filename.stem+"_mask.jpg")
            else:
                mask_path=Path(mask_root)/filename.name
            mask_hashes[str(mask_path)]=sha256(mask_path)
            with Image.open(mask_path) as image:
                mask=np.asarray(image.convert("L"))>127
            if mask.shape!=(im["height"],im["width"]): raise ValueError(f"Mask dimensions differ: {mask_path}")
            r.update(geometry(mask,ann["bbox"]))
        for label in labels:
            r.update({f"{label}_{k}":v for k,v in assigned[label][ann["id"]].items() if k!="assigned_box"})
            box=assigned[label][ann["id"]]["assigned_box"]
            if box is not None:
                error=(np.array(box)-np.array([x,y,x+w,y+h]))/np.maximum([w,h,w,h],4)
                for e,name in enumerate(("left","top","right","bottom")):
                    r[f"{label}_{name}_inward_error"]=float(error[e]*(1 if e<2 else -1))
        for label in labels[1:]:
            r[f"{label}_minus_{baseline}_assigned_iou"]=r[f"{label}_assigned_iou"]-r[f"{baseline}_assigned_iou"]
        rows.append(r)
    summary={"scope":"all fixed GT retained, no confidence filter for candidate analysis; per-image max-total-IoU one-to-one assignment; unmatched=0; oracle assignment is diagnostic only",
             "geometry_scope":"original single-box images only; semantic masks of multi-box images excluded; threshold>127; pixel-centre interior bands at 0.15*axis; raster compactness includes holes; exploratory, no target tuning",
             "geometry_mask_sha256":mask_hashes,
             "annotations_sha256":sha256(annotations),"predictions_sha256":{k:sha256(v) for k,v in predictions.items()},
             "images":len(images),"gt":len(rows),"ap_by_iou_percent":aps,"fixed_score_05_counts":detection_counts,
             "groups":{},"comparisons":{}}
    for group in ["all","small","medium","large"]+sorted({r["center"] for r in rows if r["center"]!="source"}):
        rr=[r for r in rows if group=="all" or r["size"]==group or r["center"]==group]
        summary["groups"][group]={"gt":len(rr),"models":{}}
        for label in labels:
            summary["groups"][group]["models"][label]={"mean_assigned_iou":mean([r[f"{label}_assigned_iou"] for r in rr]),
                "mean_top1_iou_single_gt":mean([r[f"{label}_top1_iou_single_gt"] for r in rr]),
                "coverage":{f"top{k}_iou{thr}":mean([float(r[f"{label}_best_iou_{k}"]>=thr) for r in rr])
                            for k in (10,100) for thr in (.5,.75,.9)}}
    for label in labels[1:]:
        common=[r for r in rows if min(r[f"{baseline}_assigned_iou"],r[f"{label}_assigned_iou"])>=.5]
        summary["comparisons"][label]={"common_iou_ge_05_gt":len(common),"edge_errors_common":{},
            "geometry_associations":associations(rows,baseline,label)}
        for name in ("left","top","right","bottom"):
            summary["comparisons"][label]["edge_errors_common"][name]={l:{
                "mean_abs":mean([abs(r[f"{l}_{name}_inward_error"]) for r in common]),
                "mean_signed_inward":mean([r[f"{l}_{name}_inward_error"] for r in common])}
                for l in (baseline,label)}
    write_json(out/"diagnostics.json",summary)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (out/"per_gt.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines=["# 配对预测诊断","","所有固定 GT 保留；候选匹配不用于替代 COCO AP。形态关联为探索性分析，不作因果结论。","",
           "| IoU | "+" | ".join(labels)+" |","|---|"+"---:|"*len(labels)]
    for threshold in aps[baseline]: lines.append("| "+threshold+" | "+" | ".join(f"{aps[l][threshold]:.3f}" for l in labels)+" |")
    lines += ["","按尺寸/中心统计、共同覆盖目标四边误差、形态关联及控制变量详见 diagnostics.json；逐目标数据见 per_gt.csv。"]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--annotations",required=True); parser.add_argument("--predictions",required=True,help="JSON mapping label to bbox JSON; first is baseline")
    parser.add_argument("--output",required=True); parser.add_argument("--mask-root")
    args=parser.parse_args(); run(args.annotations,read_json(args.predictions),args.output,args.mask_root)
