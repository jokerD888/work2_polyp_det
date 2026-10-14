import random
import shutil
from pathlib import Path

from .io import read_json, sha256, write_json


def convert_kvasir(source, output, seed=42, limit=None, min_area=5, split_dir=None, bbox_source="provided"):
    """Convert supplied boxes or explicitly requested mask-derived targets.

    Supplied boxes are copied exactly from xmin/ymin/xmax/ymax to COCO xywh.
    Box-restricted semantic masks provide auxiliary supervision, not box labels.
    Original files are read-only. Mask-derived labels require explicit opt-in.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_api

    source, output = Path(source).resolve(), Path(output).resolve()
    if bbox_source not in ("provided", "mask-components"):
        raise ValueError("bbox_source must be provided or mask-components")
    source_boxes = source / "kavsir_bboxes.json"
    provided = read_json(source_boxes) if bbox_source == "provided" else None
    if min_area < 1:
        raise ValueError("min_area must be positive")
    if output.exists():
        raise ValueError(f"Output exists; choose another directory: {output}")
    if output.is_relative_to(source):
        raise ValueError("Write conversion outside the original dataset directory")
    files = sorted(p for p in (source / "images").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    split_files = {}
    if split_dir:
        if limit is not None:
            raise ValueError("--limit cannot be combined with official --split-dir")
        split_dir = Path(split_dir).resolve()
        by_stem = {file.stem: file for file in files}
        if len(by_stem) != len(files):
            raise ValueError("Duplicate image stems prevent unambiguous split mapping")
        splits, used = {}, set()
        for split in ("train", "val"):
            text = split_dir / f"{split}.txt"
            names = [line.strip() for line in text.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not names:
                raise ValueError(f"Empty official split: {text}")
            chosen = []
            for name in names:
                if Path(name).name != name:
                    raise ValueError(f"Split entries must be image names/stems: {name}")
                stem = Path(name).stem
                if stem in used:
                    raise ValueError(f"Duplicate or overlapping official split entry: {stem}")
                if stem not in by_stem:
                    raise ValueError(f"Split references missing image: {stem}")
                used.add(stem)
                chosen.append(by_stem[stem])
            splits[split] = chosen
            split_files[split] = {"path": str(text), "sha256": sha256(text), "entries": len(names)}
        if used != set(by_stem):
            raise ValueError(f"Official splits do not cover all images: {len(set(by_stem) - used)} unassigned")
        files = [file for chosen in splits.values() for file in chosen]
    else:
        random.Random(seed).shuffle(files)
        if limit is not None:
            if limit < 10 or limit > len(files):
                raise ValueError("limit must be between 10 and total image count")
            files = files[:limit]
        if len(files) < 10:
            raise ValueError("At least 10 images needed for 80/10/10 split")
        train_end, val_end = int(len(files) * .8), int(len(files) * .9)
        splits = {"train": files[:train_end], "val": files[train_end:val_end], "test": files[val_end:]}
    # Validate all source pairs before creating an output tree.
    masks = {p.stem: p for p in (source / "masks").iterdir() if p.is_file()}
    for image in files:
        if image.stem not in masks:
            raise ValueError(f"Missing mask: {image.name}")
        if provided is not None and image.stem not in provided:
            raise ValueError(f"Missing supplied bbox entry: {image.name}")
    output.mkdir(parents=True)
    audit = {"source": str(source), "seed": seed, "split_method": "sorted filenames then Python random.Random(seed).shuffle, 80/10/10", "patient_video_grouped": False, "purpose": "local_bringup" if limit else "candidate_split_requires_review", "instance_definition": "8-connected foreground components", "min_component_area_pixels": min_area, "discarded_components": [], "mask_threshold": "grayscale > 127", "mask_storage": "lossless compressed COCO RLE", "bbox_source": "mask component tight box with exclusive xmax/ymax; supplied boxes NOT used", "limit": limit, "splits": {}, "source_files": []}
    if split_files:
        audit.update(split_method="provided train.txt/val.txt; file order preserved; no test split generated", split_files=split_files, purpose="official_train_val_comparison", seed_used_for_split=False)
        for split in split_files:
            shutil.copy2(split_dir / f"{split}.txt", output / f"{split}.txt")
    audit["bbox_source_kind"] = bbox_source
    if provided is not None:
        audit.update(bbox_source="kavsir_bboxes.json; exact xmin,ymin,xmax-xmin,ymax-ymin", instance_definition="one annotation per supplied bbox; no box filtering", min_component_area_pixels=None, area_definition="bbox width * height for bbox evaluation", auxiliary_mask_policy="semantic foreground >127 restricted to each supplied box; overlapping boxes can share foreground pixels", auxiliary_masks_are_instance_ground_truth=False, mask_assignment_audit=[])
    if source_boxes.is_file():
        audit["supplied_boxes_sha256"] = sha256(source_boxes)
        audit["supplied_boxes_count"] = sum(len(v["bbox"]) for v in read_json(source_boxes).values())
        if provided is not None:
            (output / "source").mkdir()
            shutil.copy2(source_boxes, output / "source" / source_boxes.name)
            audit["bbox_source_file"] = "source/kavsir_bboxes.json"
    image_id, annotation_id = 1, 1
    for split, chosen in splits.items():
        directory = output / split
        directory.mkdir()
        images, annotations = [], []
        for file in chosen:
            with Image.open(file) as img:
                width, height = img.size
            with Image.open(masks[file.stem]) as mask_img:
                if mask_img.size != (width, height):
                    raise ValueError(f"Image/mask shape mismatch: {file.name}")
                mask = (np.array(mask_img.convert("L")) > 127).astype(np.uint8)
            images.append({"id": image_id, "file_name": file.name, "width": width, "height": height})
            instances = []
            if provided is not None:
                import math
                metadata = provided[file.stem]
                if metadata["width"] != width or metadata["height"] != height:
                    raise ValueError(f"Supplied bbox metadata/image shape mismatch: {file.name}")
                occupancy = np.zeros_like(mask, dtype=np.uint16)
                for index, box in enumerate(metadata["bbox"]):
                    x0, y0, x1, y1 = (box[key] for key in ("xmin", "ymin", "xmax", "ymax"))
                    if box.get("label") != "polyp" or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                        raise ValueError(f"Invalid supplied bbox (left unchanged): {file.name} #{index}")
                    binary = np.zeros_like(mask)
                    ys, xs = slice(math.floor(y0), math.ceil(y1)), slice(math.floor(x0), math.ceil(x1))
                    binary[ys, xs] = mask[ys, xs]
                    occupancy[ys, xs] += 1
                    if not np.any(binary):
                        raise ValueError(f"Supplied bbox has no mask foreground: {file.name} #{index}; box not removed or fabricated")
                    instances.append(([x0, y0, x1 - x0, y1 - y0], (x1 - x0) * (y1 - y0), binary, index))
                overlapping = int(np.count_nonzero(mask & (occupancy > 1)))
                unassigned = int(np.count_nonzero(mask & (occupancy == 0)))
                if overlapping or unassigned:
                    audit["mask_assignment_audit"].append({"file_name": file.name, "overlapping_foreground_pixels": overlapping, "foreground_outside_boxes_pixels": unassigned})
            else:
                count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if count < 2 or not any(stats[label, 4] >= min_area for label in range(1, count)):
                    raise ValueError(f"Empty foreground mask: {file.name}")
                for label in range(1, count):
                    if stats[label, 4] < min_area:
                        audit["discarded_components"].append({"file_name": file.name, "component": label, "area": int(stats[label, 4])})
                        continue
                    x, y, w, h, area = (int(v) for v in stats[label])
                    instances.append(([x, y, w, h], area, labels == label, None))
            for bbox, area, binary, source_index in instances:
                binary = np.asfortranarray(binary, dtype=np.uint8)
                rle = mask_api.encode(binary)
                rle["counts"] = rle["counts"].decode("ascii")
                annotation = {"id": annotation_id, "image_id": image_id, "category_id": 1, "bbox": bbox, "area": area, "iscrowd": 0, "segmentation": rle}
                if source_index is not None:
                    annotation.update(source_bbox_index=source_index, source_bbox_file="kavsir_bboxes.json")
                annotations.append(annotation)
                annotation_id += 1
            shutil.copy2(file, directory / file.name)
            audit["source_files"].append({"file_name": file.name, "image_sha256": sha256(file), "mask_sha256": sha256(masks[file.stem])})
            image_id += 1
        description = "Kvasir-SEG supplied bbox detection targets" if provided is not None else "Kvasir-SEG mask-derived connected-component detection targets"
        write_json(output / f"{split}.json", {"info": {"description": description}, "images": images, "annotations": annotations, "categories": [{"id": 1, "name": "polyp"}]})
        audit["splits"][split] = {"files": [p.name for p in chosen], "images": len(images), "instances": len(annotations)}
    write_json(output / "conversion.json", audit)
    return audit
