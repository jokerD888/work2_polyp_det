"""Image-only prediction using the frozen trained model and native postprocess.

Export all still frames without evaluating incomplete annotations as negatives.
Evaluation is performed separately on the locked, supplied-label cohorts.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from polypbench.io import read_json,write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",required=True)
    args=parser.parse_args()
    job=read_json(args.job)
    experiment=read_json(job["experiment"])
    sys.path.insert(0,experiment["project_root"])
    sys.path.insert(0,experiment["repo"]["path"])
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]="1"
    import numpy as np
    import torch
    import train_net
    from detectron2.data import DatasetMapper
    from detectron2.modeling import build_model
    from detectron2.engine import default_setup
    from detectron2.evaluation.coco_evaluation import instances_to_coco_json
    from methods.mssl.config import build_cfg
    output=Path(job["output"])
    cfg=build_cfg(experiment,output).clone();cfg.defrost()
    cfg.MODEL.WEIGHTS=job["checkpoint"]
    cfg.freeze()
    default_setup(cfg,args)
    (output/"resolved_inference.yaml").write_text(cfg.dump(),encoding="utf-8")
    model=build_model(cfg)
    saved=torch.load(job["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(saved["model"],strict=True)
    del saved
    model.eval()
    mapper=DatasetMapper(cfg,False)
    def predict(image, image_root):
        record=dict(image)
        record["file_name"]=str(Path(image_root)/image["file_name"])
        record["image_id"]=record.pop("id")
        mapped=mapper(record)
        # No mask/box annotation is provided to the model.
        assert "instances" not in mapped and "annotations" not in mapped
        with torch.no_grad():
            instances=model([mapped])[0]["instances"]
        if instances.has("pred_masks"):
            instances.remove("pred_masks")  # export bbox only, keep native scoring
        rows=instances_to_coco_json(instances.to("cpu"),image["id"])
        for row in rows:
            if row["category_id"]!=0:
                raise RuntimeError("Unexpected class in the one-class model")
            row["category_id"]=1
        return rows
    source=experiment["dataset"]["val"]
    source_images=read_json(source["annotations"])["images"]
    first=source_images[0]
    actual=predict(first,source["image_root"])
    expected=[x for x in read_json(job["source_predictions"]) if x["image_id"]==first["id"]]
    if len(actual)!=len(expected):
        raise AssertionError("Source-validation prediction count mismatch")
    np.testing.assert_allclose([x["bbox"] for x in actual],[x["bbox"] for x in expected],rtol=1e-5,atol=1e-3)
    np.testing.assert_allclose([x["score"] for x in actual],[x["score"] for x in expected],rtol=1e-5,atol=1e-6)
    write_json(output/"source_prediction_parity.json",{"status":"passed","image_id":first["id"],
        "predictions":len(actual),"bbox_atol":.001,"score_atol":.000001,"strict_checkpoint_load":True})
    images=read_json(job["all_images"])["images"]
    rows=[];start=time.monotonic()
    for index,image in enumerate(images,1):
        rows.extend(predict(image,job["image_root"]))
        if index%50==0 or index==len(images):
            elapsed=time.monotonic()-start
            progress={"status":"running","images_done":index,"images_total":len(images),
                      "elapsed_seconds":elapsed,"eta_seconds":elapsed/index*(len(images)-index)}
            write_json(output/"progress.json",progress)
            print(json.dumps(progress),flush=True)
    write_json(output/"predictions.all_stills.bbox.json",rows)
    write_json(output/"progress.json",{"status":"completed","images_done":len(images),
        "images_total":len(images),"elapsed_seconds":time.monotonic()-start})


if __name__=="__main__":
    main()
