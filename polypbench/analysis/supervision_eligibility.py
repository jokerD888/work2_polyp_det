"""CPU-only audit of native MSSL labels and fixed LSJ supervision eligibility.

No model, CUDA context, optimizer or invented instance masks. Multi-box semantic
support is described but is never promoted to reliable instance supervision.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import random
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from polypbench.io import read_json, sha256, write_json
from .candidate_stage import ReturnObserver
from .local_relation_audit import regions


RULES = {
    "instance_identity": "Only original-single-box raw semantic masks are eligible. Multi-box masks remain identity-ambiguous even after cropping to one box.",
    "regions": "Existing geometric core/ring rule (.05/.20 sqrt foreground area); exclude padding and all semantic foreground from background.",
    "grid": "Area-resample region occupancy at strides 4/8/16/32/64; a cell is usable only with occupancy >=.95 and valid-image occupancy >=.95; at least one foreground and one background cell.",
    "paths": "Current resolved architecture: stride4 is the fused mask feature, candidate multiscale inputs use strides8/16/32/64 (TOTAL_NUM_FEATURE_LEVELS=4). They are not interchangeable attachment paths.",
    "interpretation": "Prospective geometric feasibility screen, not annotation reliability, feature receptive-field purity, or a validated loss design.",
    "small": "Descriptive augmented bbox area ratio <=.03455088874936203; not a training hyperparameter or COCO small-object definition.",
    "acceptance": "No global passing-percentage threshold; report all denominators and shortages by original multiplicity and post-LSJ size.",
}
AREA_REFERENCE = .03455088874936203
STRIDES = (4, 8, 16, 32, 64)


def deterministic_seed(base, image_id):
    return int.from_bytes(hashlib.sha256(f"lsj:{base}:{image_id}".encode()).digest()[:4], "little")


def box_support(mask, boxes):
    """Spatial ownership proxy only; does not generate an instance annotation."""
    h, w = mask.shape
    ownership = np.zeros((h, w), dtype=np.int16)
    rects = []
    for box in boxes:
        x0, y0, x1, y1 = box
        # Pixel centers inside supplied box; no modification of training boxes.
        inside = ((np.arange(w)[None] + .5 >= x0) & (np.arange(w)[None] + .5 < x1)
                & (np.arange(h)[:, None] + .5 >= y0) & (np.arange(h)[:, None] + .5 < y1))
        ownership += inside
        rects.append(inside)
    return [{"semantic_pixels_in_box": int((mask & rect).sum()),
             "semantic_pixels_unique_box": int((mask & rect & (ownership == 1)).sum()),
             "semantic_pixels_ambiguous_boxes": int((mask & rect & (ownership > 1)).sum())}
            for rect in rects]


def region_eligibility(mask, valid, reliable_identity, strides=STRIDES):
    mask = np.asarray(mask, dtype=bool) & valid
    core, ring = regions(mask)
    core &= valid
    ring &= valid & ~mask
    result = {"semantic_foreground_pixels": int(mask.sum()), "core_pixels": int(core.sum()),
              "ring_pixels": int(ring.sum()), "pixel_eligible": bool(reliable_identity and core.any() and ring.any()),
              "grids": {}}
    h, w = mask.shape
    for stride in strides:
        gh, gw = (h+stride-1)//stride, (w+stride-1)//stride
        occupancy = lambda region: cv2.resize(region.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        good = occupancy(valid) >= .95
        fg = int(((occupancy(core) >= .95) & good).sum())
        bg = int(((occupancy(ring) >= .95) & good).sum())
        result["grids"][str(stride)] = {"foreground_cells": fg, "background_cells": bg,
            "valid_cells": int(good.sum()), "eligible": bool(reliable_identity and fg and bg)}
    return result


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        for group in ("all", "original=" + r["original_multiplicity"],
                      "phase=" + r["phase"],
                      f"phase={r['phase']}/base_seed={r['augmentation_base_seed']}",
                      f"phase={r['phase']}/original={r['original_multiplicity']}/small={r['small']}"):
            groups[group].append(r)
    result = {}
    for name, items in groups.items():
        result[name] = {"gt_observations": len(items), "retained_gt": sum(r["retained"] for r in items),
            "reliable_instance_identity": sum(r["reliable_identity"] and r["retained"] for r in items),
            "pixel_eligible": sum(r["pixel_eligible"] for r in items),
            "shortage_counts_nonexclusive": dict(Counter(reason for r in items for reason in r["shortages"])),
            "grid_eligible": {str(s): sum(r[f"stride{s}_eligible"] for r in items) for s in STRIDES}}
    return result


def make_rows(record, original_anns, semantic, valid, transformed_anns, retained,
              native_masks, phase, seed):
    from detectron2.structures import BoxMode
    h, w = valid.shape
    boxes = [np.asarray(BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS)).tolist() for a in transformed_anns]
    supports = box_support(semantic, boxes)
    reliable = len(original_anns) == 1
    eligibility = region_eligibility(semantic, valid, reliable)
    rows = []
    native_index = 0
    for i, (ann, kept, box, support) in enumerate(zip(transformed_anns, retained, boxes, supports)):
        reasons = []
        if not kept:
            reasons.append("native_box_filtered")
        if not reliable:
            reasons.append("instance_identity_unavailable")
        if not eligibility["semantic_foreground_pixels"]:
            reasons.append("semantic_foreground_erased")
        if not eligibility["core_pixels"]:
            reasons.append("core_empty")
        if not eligibility["ring_pixels"]:
            reasons.append("ring_empty")
        if support["semantic_pixels_in_box"] == 0:
            reasons.append("no_semantic_support_inside_box")
        ratio = max(0, box[2]-box[0])*max(0, box[3]-box[1])/(h*w)
        native_pixels = None
        if kept:
            native_pixels = int(native_masks[native_index].sum())
            if native_pixels == 0:
                reasons.append("native_instance_mask_empty")
            native_index += 1
        row = {"image_id": record["image_id"], "annotation_id": original_anns[i]["id"],
            "phase": phase, "augmentation_base_seed": seed, "original_multiplicity": "single" if reliable else "multi",
            "original_gt_count": len(original_anns), "retained_image_gt_count": int(sum(retained)),
            "retained": bool(kept), "reliable_identity": reliable, "box_area_ratio": ratio,
            "small": ratio <= AREA_REFERENCE, "padding_pixels": int((~valid).sum()),
            "semantic_foreground_pixels": eligibility["semantic_foreground_pixels"],
            "core_pixels": eligibility["core_pixels"], "ring_pixels": eligibility["ring_pixels"],
            "native_instance_mask_pixels": native_pixels, **support,
            "pixel_eligible": bool(kept and eligibility["pixel_eligible"]), "shortages": reasons}
        for stride, grid in eligibility["grids"].items():
            row[f"stride{stride}_foreground_cells"] = grid["foreground_cells"] if reliable else None
            row[f"stride{stride}_background_cells"] = grid["background_cells"] if reliable else None
            row[f"stride{stride}_eligible"] = bool(kept and grid["eligible"])
        rows.append(row)
    assert native_index == len(native_masks)
    return rows


def run(experiment_path, output, workspace, seeds=(42, 142, 242), limit=None,
        annotations=None, image_root=None, raw_mask_root=None):
    import torch
    # No torch.cuda.* calls, model imports or model configuration construction.
    torch.set_num_threads(1); cv2.setNumThreads(1)
    root = Path(workspace).resolve()
    sys.path.insert(0, str(root / "third_party/MaskDINO"))
    from methods.mssl.data import load_records, MSSLMapper
    from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
    from detectron2.structures import BoxMode, Boxes
    experiment = read_json(experiment_path)
    settings = experiment["overrides"]
    if experiment["dataset"]["training_boxes"] != "annotations":
        raise ValueError("Audit requires the current provided-box training protocol")
    cfg = SimpleNamespace(INPUT=SimpleNamespace(IMAGE_SIZE=settings["INPUT.IMAGE_SIZE"],
        MIN_SCALE=settings["INPUT.MIN_SCALE"], MAX_SCALE=settings["INPUT.MAX_SCALE"],
        RANDOM_FLIP=settings.get("INPUT.RANDOM_FLIP", "horizontal"), FORMAT="RGB"))
    annotations = Path(annotations or experiment["dataset"]["train"]["annotations"])
    image_root = Path(image_root or experiment["dataset"]["train"]["image_root"])
    raw_mask_root = Path(raw_mask_root or experiment["dataset"]["raw_mask_root"])
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    coco = read_json(annotations)
    groups = defaultdict(list)
    for a in coco["annotations"]:
        groups[a["image_id"]].append(a)
    records = sorted(load_records(str(annotations), str(image_root), None, str(raw_mask_root)), key=lambda r: r["image_id"])
    if limit is not None:
        records = records[:limit]
    files = [Path(experiment_path), annotations, Path(__file__), root / "methods/mssl/data.py",
             root / "polypbench/adapters/bitmask_mapper.py", root / "polypbench/adapters/instance_lsj.py",
             root / "polypbench/analysis/local_relation_audit.py"]
    for record in records:
        files.extend((Path(record["file_name"]), raw_mask_root / Path(record["file_name"]).name))
    hashes = {str(p.resolve()): sha256(p) for p in files}
    write_json(out / "protocol.json", {"rules": RULES, "base_seeds": list(seeds),
        "per_image_rng": "SHA256(lsj:base_seed:image_id) first four bytes little endian",
        "mapper_settings": vars(cfg.INPUT), "source_sha256": hashes, "device": "cpu", "model_forward": False,
        "images": len(records), "limited_engineering_run": limit is not None,
        "feature_levels_note": "Stride grids match current architecture resolutions; occupancy is geometric, not learned feature purity."})
    mapper = MSSLMapper(cfg, "annotations")
    rows = []; start = time.monotonic()
    write_json(out / "state.json", {"status": "running"})
    try:
        for number, record in enumerate(records, 1):
            original_anns = groups[record["image_id"]]
            assert len(original_anns) == len(record["annotations"])
            for ann, original in zip(record["annotations"], original_anns):
                np.testing.assert_allclose(BoxMode.convert(ann["bbox"], ann["bbox_mode"], BoxMode.XYWH_ABS), original["bbox"], rtol=0, atol=0)
            raw = np.asarray(Image.open(raw_mask_root / Path(record["file_name"]).name).convert("L")) > 127
            assert raw.shape == (record["height"], record["width"])
            # Original labels are descriptive; native multi-box segmentation is
            # preserved, never substituted by our spatial ownership proxy.
            from detectron2.data import detection_utils as utils
            original_instances = utils.annotations_to_instances(record["annotations"], raw.shape, mask_format="bitmask")
            original_masks = original_instances.gt_masks.tensor.numpy()
            rows.extend(make_rows(record, original_anns, raw, np.ones_like(raw), record["annotations"],
                                  np.ones(len(original_anns), dtype=bool), original_masks, "original", None))
            for base in seeds:
                seed = deterministic_seed(base, record["image_id"])
                random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
                observed = {}
                def capture(loc, result):
                    assert loc["self"] is mapper
                    observed.update(transforms=loc["transforms"], annotations=loc["annotations"])
                with ReturnObserver({BitmaskLSJMapper.__call__.__code__: capture}):
                    mapped = mapper(record)
                instances = mapped["instances"]
                valid = ~mapped["padding_mask"].numpy()
                semantic = observed["transforms"].apply_segmentation(raw.astype(np.uint8)).astype(bool)
                assert not semantic[~valid].any()
                transformed = observed["annotations"]
                assert len(transformed) == len(original_anns)
                b = torch.from_numpy(np.asarray([BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS) for a in transformed], dtype=np.float32))
                kept = Boxes(b).nonempty(threshold=1e-5).numpy()
                np.testing.assert_allclose(instances.gt_boxes.tensor.numpy(), b.numpy()[kept], rtol=0, atol=0)
                native_masks = instances.gt_masks.numpy()
                assert not native_masks[:, ~valid].any()
                np.testing.assert_array_equal(instances.gt_mssl_eligible.numpy(), np.full(int(kept.sum()), len(original_anns)==1))
                if len(original_anns) == 1 and kept[0]:
                    np.testing.assert_array_equal(native_masks[0], semantic)
                rows.extend(make_rows(record, original_anns, semantic, valid, transformed, kept,
                                      native_masks, "lsj", base))
                del mapped, observed, semantic, native_masks, instances
            if number % 20 == 0 or number == len(records):
                write_json(out / "progress.json", {"images_done": number, "images_total": len(records),
                    "elapsed_seconds": time.monotonic()-start})
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise RuntimeError(f"Input changed: {path}")
        write_json(out / "summary.json", {"groups": summarize(rows), "rules": RULES,
            "images": len(records), "base_seeds": list(seeds), "inputs_unchanged": True,
            "model_forward": False, "cpu_only": True,
            "limits": "Repeated augmented views are not independent instances or a complete training distribution. Multi-box spatial ownership is not instance truth."})
        with (out / "per_gt.jsonl").open("w", encoding="utf-8") as stream:
            import json
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        flat = [{**r, "shortages": "|".join(r["shortages"])} for r in rows]
        with (out / "per_gt.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        write_json(out / "state.json", {"status": "completed", "elapsed_seconds": time.monotonic()-start})
    except BaseException as error:
        write_json(out / "state.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 142, 242])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--annotations")
    parser.add_argument("--image-root")
    parser.add_argument("--raw-mask-root")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate seeds would double-count identical augmented views")
    run(args.experiment, args.output, args.workspace, args.seeds, args.limit,
        args.annotations, args.image_root, args.raw_mask_root)
