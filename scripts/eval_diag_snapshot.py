"""Independent-process evaluation of a diag snapshot on a fixed val subset.

Builds the model from the run's recorded experiment.json, loads the snapshot,
runs plain fp32 inference over the first --max-images validation records
(deterministic subset, same preprocessing as the standard test path), and
writes bbox metrics + prediction statistics. Never touches the training run.
"""
import argparse
from pathlib import Path
import json
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment", required=True, help="run dir experiment.json")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max-images", type=int, default=60)
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/MaskDINO"))
os_ok = __import__("os").environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from polypbench.io import read_json, write_json

experiment = read_json(args.experiment)
sys.path.insert(0, experiment["repo"]["path"])

import numpy as np
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from methods.scda_proposal.precheck import build_cfg
from methods.scda_proposal.integration import register

register(experiment["repo"]["path"])
entry = experiment["dataset"]["val"]
records = load_coco_json(entry["annotations"], entry["image_root"], "diag_eval_val",
                         extra_annotation_keys=["id", "scda_reliable"])
# (image ids required below to subset annotations)
records = records[:args.max_images]
subset = json.loads(Path(entry["annotations"]).read_text())
keep = {r["image_id"] for r in records}
subset["images"] = [im for im in subset["images"] if im["id"] in keep]
subset["annotations"] = [a for a in subset["annotations"] if a["image_id"] in keep]
subset_path = Path(args.output).parent / "diag_eval_subset.json"
subset_path.write_text(json.dumps(subset))
DatasetCatalog.register("diag_eval", lambda: load_coco_json(
    str(subset_path), entry["image_root"], "diag_eval"))
MetadataCatalog.get("diag_eval").set(json_file=str(subset_path),
                                     image_root=entry["image_root"], evaluator_type="coco")

cfg = build_cfg(experiment, True, Path(args.output).parent)
torch.manual_seed(0)
np.random.seed(0)
model = build_model(cfg)
DetectionCheckpointer(model).load(args.checkpoint)
model.eval().cuda()

from detectron2.data import build_detection_test_loader
loader = build_detection_test_loader(cfg, dataset_name="diag_eval")

predictions, top_scores = [], []
with torch.no_grad():
    for inputs in loader:
        outputs = model(inputs)
        for record, out in zip(inputs, outputs):
            inst = out["instances"].to("cpu")
            boxes = inst.pred_boxes.tensor.numpy().tolist()
            scores = inst.scores.numpy().tolist()
            classes = inst.pred_classes.numpy().tolist()
            top_scores.append(max(scores) if scores else 0.0)
            for box, score, cls in zip(boxes, scores, classes):
                predictions.append({"image_id": record["image_id"], "category_id": int(cls) + 1,
                                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                                    "score": float(score)})

pred_path = Path(args.output).parent / "diag_eval_predictions.json"
pred_path.write_text(json.dumps(predictions))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

coco = COCO(str(subset_path))
coco_dt = coco.loadRes(str(pred_path))
e = COCOeval(coco, coco_dt, "bbox")
e.params.imgIds = sorted(keep)
e.evaluate()
e.accumulate()
e.summarize()
metrics = {"AP": float(e.stats[0]), "AP50": float(e.stats[1]), "AP75": float(e.stats[2]),
           "n_images": len(records), "detections_per_image_mean": len(predictions) / len(records),
           "mean_top_score": float(np.mean(top_scores)),
           "frac_top_score_ge_0.3": float(np.mean([s >= 0.3 for s in top_scores])),
           "top_scores": [float(s) for s in top_scores]}
write_json(args.output, metrics)
print({"output": args.output, **{k: v for k, v in metrics.items() if k != "top_scores"}})
