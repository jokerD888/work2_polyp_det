"""Image-only checkpoint replay: score components and all-query box geometry.

Calls native instance_inference unchanged; adds a read-only observer. No GT,
optimizer, loss or method modification. Every exported native prediction is
checked against the already frozen predictions, not just the first image.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
import types

from polypbench.io import read_json,sha256,write_json


def run(job_path):
    job=read_json(job_path); experiment=read_json(job["experiment"])
    sys.path.insert(0,experiment["project_root"]); sys.path.insert(0,experiment["repo"]["path"])
    import numpy as np
    import torch
    import train_net
    from detectron2.data import DatasetMapper
    from detectron2.modeling import build_model
    from methods.mssl.config import build_cfg
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    out=Path(job["output"]); out.mkdir(parents=True,exist_ok=False)
    cfg=build_cfg(experiment,out)
    model=build_model(cfg)
    if sha256(job["checkpoint"])!=job["checkpoint_sha256"]: raise RuntimeError("Checkpoint drift")
    saved=torch.load(job["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(saved["model"],strict=True); del saved
    model.eval(); mapper=DatasetMapper(cfg,False)
    assert not model.focus_on_box and not model.panoptic_on and model.sem_seg_head.num_classes==1
    expected=defaultdict(list)
    for p in read_json(job["predictions"]): expected[p["image_id"]].append(p)
    images=sorted(read_json(job["annotations"])["images"],key=lambda r:r["id"])
    observed={}; original=model.instance_inference
    def observer(self,mask_cls,mask_pred,mask_box_result):
        classes=mask_cls.sigmoid().flatten()
        selected_scores,indices=classes.topk(self.test_topk_per_image,sorted=False)
        # box_postprocess has already applied the native image/padding scaling.
        query_boxes=mask_box_result.detach().float().cpu().numpy().copy()
        query_boxes[:,2:]-=query_boxes[:,:2]
        observed["all_boxes"]=query_boxes
        observed["all_class_scores"]=classes.detach().float().cpu().numpy()
        result=original(mask_cls,mask_pred,mask_box_result)
        observed["query_ids"]=indices.detach().cpu().numpy()
        observed["class_scores"]=selected_scores.detach().float().cpu().numpy()
        # Native fused score=s_cls*s_mask. Recover the already computed quality
        # without allocating another full-resolution mask tensor.
        observed["mask_scores"]=(result.scores/selected_scores.clamp_min(1e-30)).detach().float().cpu().numpy()
        return result
    model.instance_inference=types.MethodType(observer,model)
    image_ids=[]; bbox=[]; final_scores=[]; class_scores=[]; mask_scores=[]; query_ids=[]; all_boxes=[]; all_class=[]
    max_bbox_error=max_score_error=0.; start=time.monotonic()
    for index,image in enumerate(images,1):
        record={k:v for k,v in image.items() if k in ("file_name","height","width")}
        record["file_name"]=str(Path(job["image_root"])/record["file_name"]); record["image_id"]=image["id"]
        mapped=mapper(record); assert "instances" not in mapped and "annotations" not in mapped
        with torch.no_grad(): inst=model([mapped])[0]["instances"]
        boxes=inst.pred_boxes.tensor.detach().float().cpu().numpy().copy(); boxes[:,2:]-=boxes[:,:2]
        scores=inst.scores.detach().float().cpu().numpy()
        ref=expected[image["id"]]
        if len(ref)!=len(scores): raise AssertionError("Native prediction count mismatch")
        ref_boxes=np.array([r["bbox"] for r in ref]); ref_scores=np.array([r["score"] for r in ref])
        np.testing.assert_allclose(boxes,ref_boxes,rtol=1e-5,atol=1e-3)
        np.testing.assert_allclose(scores,ref_scores,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(scores,observed["class_scores"]*observed["mask_scores"],rtol=1e-5,atol=1e-7)
        max_bbox_error=max(max_bbox_error,float(np.max(abs(boxes-ref_boxes))))
        max_score_error=max(max_score_error,float(np.max(abs(scores-ref_scores))))
        image_ids.append(image["id"]); bbox.append(boxes); final_scores.append(scores)
        class_scores.append(observed["class_scores"]); mask_scores.append(observed["mask_scores"]); query_ids.append(observed["query_ids"])
        all_boxes.append(observed["all_boxes"]); all_class.append(observed["all_class_scores"])
        del inst,mapped
        if index%100==0 or index==len(images):
            elapsed=time.monotonic()-start
            write_json(out/"progress.json",{"images_done":index,"images_total":len(images),"elapsed_seconds":elapsed,"eta_seconds":elapsed/index*(len(images)-index)})
    np.savez_compressed(out/"score_components.npz",image_ids=np.asarray(image_ids),bbox=np.asarray(bbox),
        final_scores=np.asarray(final_scores),class_scores=np.asarray(class_scores),mask_scores=np.asarray(mask_scores),
        query_ids=np.asarray(query_ids),all_query_boxes=np.asarray(all_boxes),all_query_class_scores=np.asarray(all_class))
    write_json(out/"verification.json",{"status":"passed","images":len(images),"detections":sum(map(len,bbox)),
        "checkpoint_sha256":job["checkpoint_sha256"],"predictions_sha256":sha256(job["predictions"]),
        "max_bbox_absolute_error_pixels":max_bbox_error,"max_score_absolute_error":max_score_error,
        "source":"native inference unchanged, strict state_dict load; all-image native prediction parity; no annotations supplied to model",
        "score_note":"mask score recovered from native product; diagnostic only; class-only alternative uses same original selected100 and boxes",
        "elapsed_seconds":time.monotonic()-start})


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--job",required=True); args=p.parse_args(); run(args.job)
