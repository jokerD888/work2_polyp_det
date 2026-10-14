import hashlib
import math
from pathlib import Path

from .io import read_json, sha256


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def verify_supplied_boxes(annotations, source_boxes):
    data, provided = read_json(annotations), read_json(source_boxes)
    by_image = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    total = 0
    for image in data["images"]:
        expected = provided[Path(image["file_name"]).stem]["bbox"]
        actual = by_image.get(image["id"], [])
        if len(actual) != len(expected):
            raise ValueError(f"Supplied bbox count changed: {image['file_name']}")
        for index, (annotation, box) in enumerate(zip(actual, expected)):
            coordinates = [box["xmin"], box["ymin"], box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
            if annotation["bbox"] != coordinates or annotation.get("source_bbox_index") != index:
                raise ValueError(f"Supplied bbox coordinates changed: {image['file_name']} #{index}")
        total += len(actual)
    return total


def validate_coco(annotations, image_root=None, require_masks=False, check_images=True, mask_format="polygon"):
    data = read_json(annotations)
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"COCO {key} must be a list: {annotations}")
    images, categories, annotation_ids, files = {}, {}, set(), set()
    image_hashes = hashlib.sha256()
    for category in data["categories"]:
        cid = category["id"]
        if type(cid) is not int or cid in categories or not category.get("name"):
            raise ValueError(f"Invalid/duplicate category: {category}")
        categories[cid] = category["name"]
    if not categories:
        raise ValueError("No categories in annotation file")
    root = Path(image_root).resolve() if image_root else None
    for image in data["images"]:
        iid = image["id"]
        if type(iid) is not int or iid in images:
            raise ValueError(f"Invalid/duplicate image id: {iid}")
        for dimension in ("height", "width"):
            if type(image.get(dimension)) is not int or image[dimension] <= 0:
                raise ValueError(f"Invalid image {dimension}: {iid}")
        filename = image["file_name"]
        if filename in files:
            raise ValueError(f"Duplicate image file_name: {filename}")
        files.add(filename)
        if root:
            file = (root / filename).resolve()
            if not file.is_relative_to(root):
                raise ValueError(f"Image escapes image_root: {filename}")
            if check_images:
                from PIL import Image
                with Image.open(file) as opened:
                    if opened.size != (image["width"], image["height"]):
                        raise ValueError(f"Image size differs from COCO metadata: {file}")
                image_hashes.update((filename + ":" + sha256(file) + "\n").encode())
        images[iid] = image
    if not images:
        raise ValueError("No images in annotation file")
    foreground = 0
    for ann in data["annotations"]:
        aid = ann["id"]
        if type(aid) is not int or aid in annotation_ids:
            raise ValueError(f"Invalid/duplicate annotation id: {aid}")
        annotation_ids.add(aid)
        if ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError(f"Unknown image/category in annotation {aid}")
        box = ann.get("bbox", [])
        if len(box) != 4 or not all(_number(v) for v in box) or min(box[2:]) <= 0:
            raise ValueError(f"Invalid xywh bbox in annotation {aid}")
        if not _number(ann.get("area")) or ann["area"] <= 0:
            raise ValueError(f"Annotation {aid} needs positive COCO area")
        if ann.get("iscrowd", 0) not in (0, 1):
            raise ValueError(f"Invalid iscrowd in annotation {aid}")
        if not ann.get("iscrowd", 0):
            foreground += 1
            if require_masks:
                polygons = ann.get("segmentation")
                if mask_format == "bitmask" and isinstance(polygons, dict):
                    import numpy as np
                    from pycocotools import mask as mask_api
                    image = images[ann["image_id"]]
                    try:
                        rle = polygons
                        if isinstance(rle.get("counts"), list):
                            rle = mask_api.frPyObjects(rle, image["height"], image["width"])
                        decoded = mask_api.decode(rle)
                    except Exception as error:
                        raise ValueError(f"Invalid RLE in annotation {aid}") from error
                    if decoded.shape != (image["height"], image["width"]) or not np.any(decoded):
                        raise ValueError(f"Empty/wrong-size RLE in annotation {aid}")
                    continue
                if not isinstance(polygons, list) or not polygons:
                    raise ValueError(f"Annotation {aid}: native LSJ training requires polygon masks (RLE/bbox-only unsupported)")
                for polygon in polygons:
                    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 or not all(_number(v) for v in polygon):
                        raise ValueError(f"Malformed polygon in annotation {aid}")
                    signed_area = sum(polygon[i] * polygon[(i + 3) % len(polygon)] - polygon[(i + 2) % len(polygon)] * polygon[i + 1] for i in range(0, len(polygon), 2))
                    if abs(signed_area) < 1e-6:
                        raise ValueError(f"Zero-area polygon in annotation {aid}")
    if not foreground:
        raise ValueError("No non-crowd ground-truth objects")
    return {"images": len(images), "annotations": len(annotation_ids), "categories": {str(k): v for k, v in sorted(categories.items())},
            "annotations_sha256": sha256(annotations), "images_sha256": image_hashes.hexdigest() if root and check_images else None, "file_names": sorted(files)}


def validate_dataset(cfg, check_images=True, splits=None):
    result = {}
    if cfg["dataset"].get("bbox_source") == "provided":
        conversion = Path(cfg["dataset"]["train"]["annotations"]).parent / "conversion.json"
        if not conversion.is_file() or read_json(conversion).get("bbox_source_kind") != "provided":
            raise ValueError("Configured supplied bbox source does not match converted dataset provenance")
        provenance = read_json(conversion)
        source_boxes = conversion.parent / provenance["bbox_source_file"]
        if sha256(source_boxes) != provenance["supplied_boxes_sha256"]:
            raise ValueError("Archived supplied bbox JSON differs from its recorded checksum")
        for split in splits or (s for s in ("train", "val", "test") if s in cfg["dataset"]):
            verify_supplied_boxes(cfg["dataset"][split]["annotations"], source_boxes)
    for split in splits or (s for s in ("train", "val", "test") if s in cfg["dataset"]):
        entry = cfg["dataset"][split]
        result[split] = validate_coco(entry["annotations"], entry["image_root"], require_masks=split == "train", check_images=check_images, mask_format=cfg["dataset"].get("mask_format", "polygon"))
        if len(result[split]["categories"]) != cfg["model"]["num_classes"]:
            raise ValueError(f"{split}: category count differs from model.num_classes")
    first = next(iter(result.values()))["categories"]
    if any(item["categories"] != first for item in result.values()):
        raise ValueError("Category IDs/names must match across splits")
    # Identical image paths across splits are definite leakage. Patient/video grouping
    # requires metadata and must be supplied by the dataset owner.
    paths = {}
    for split, item in result.items():
        for filename in item.pop("file_names"):
            file = str((Path(cfg["dataset"][split]["image_root"]) / filename).resolve())
            if file in paths:
                raise ValueError(f"Image shared across {paths[file]} and {split}: {file}")
            paths[file] = split
    budget = cfg.get("training_budget")
    if budget and "train" in result:
        import math
        iterations_per_epoch = math.ceil(result["train"]["images"] / cfg["overrides"].get("SOLVER.IMS_PER_BATCH", 16))
        if budget["train_images"] != result["train"]["images"] or budget["iterations_per_epoch"] != iterations_per_epoch:
            raise ValueError("training_budget does not match train image count/global batch")
        if cfg["overrides"].get("SOLVER.MAX_ITER") != budget["epochs"] * iterations_per_epoch:
            raise ValueError("SOLVER.MAX_ITER does not match the declared epoch budget")
    return result
