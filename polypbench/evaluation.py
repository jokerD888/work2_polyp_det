import copy
import importlib.metadata
import math

from .data import validate_coco
from .io import read_json, sha256, write_json

METRIC_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def evaluate(annotations, predictions, output=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    validate_coco(annotations, check_images=False)
    gt = COCO(annotations)
    rows = read_json(predictions)
    if not isinstance(rows, list):
        raise ValueError("Predictions must be a COCO detection results list")
    normalized = []
    for index, row in enumerate(rows):
        if row.get("image_id") not in gt.imgs or row.get("category_id") not in gt.cats:
            raise ValueError(f"Prediction {index}: image/category ID not in ground truth")
        box, score = row.get("bbox", []), row.get("score")
        values = [*box, score]
        if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError(f"Prediction {index}: non-finite bbox or score")
        if min(box[2:]) < 0 or not 0 <= score <= 1:
            raise ValueError(f"Prediction {index}: negative box extent or score outside [0,1]")
        normalized.append({key: row[key] for key in ("image_id", "category_id", "bbox", "score")})
    # COCO.loadRes indexes anns[0], so construct an empty result object explicitly.
    if normalized:
        dt = gt.loadRes(normalized)
    else:
        dt = COCO()
        dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]), "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
        dt.createIndex()
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = sorted(gt.imgs)  # includes images with zero predictions
    evaluator.params.catIds = sorted(gt.cats)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    metrics = {name: float(value) if value >= 0 else None for name, value in zip(METRIC_NAMES, evaluator.stats)}
    record = {"metrics": metrics, "units": "fraction_0_to_1", "protocol": {"iou_type": "bbox", "iou_thresholds": [round(float(x), 2) for x in evaluator.params.iouThrs], "max_dets": [1, 10, 100], "all_gt_images": True, "score_filter": None, "pycocotools": importlib.metadata.version("pycocotools")}, "images": len(gt.imgs), "predictions": len(rows), "annotations_sha256": sha256(annotations), "predictions_sha256": sha256(predictions)}
    if output:
        write_json(output, record)
    return record
