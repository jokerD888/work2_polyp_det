"""Read-only native MaskDINO stage tracing. GPU entry point is remote-only.

Observe Python return frames, without replacing forward, TopK or mask2box.
The observer is pinned to the checked native revision and fails on drift.
GT is used only after image-only inference has returned.
"""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from polypbench.io import read_json, sha256, write_json
from .matching import iou_matrix, maximum_coverage, xyxy


NATIVE_COMMIT = "3831d8514a3728535ace8d4ecc7d28044c42dd14"
THRESHOLDS = (.5, .75, .9)


class ReturnObserver:
    """Scoped return-frame inspection; never changes arguments or return values."""
    def __init__(self, handlers):
        self.handlers = handlers

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError("Refuse to replace an existing Python profiler")
        sys.setprofile(self._observe)
        return self

    def _observe(self, frame, event, result):
        if event == "return":
            handler = self.handlers.get(frame.f_code)
            if handler is not None:
                handler(frame.f_locals, result)

    def __exit__(self, *exc):
        sys.setprofile(None)


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def to_original(boxes, padded_hw, mapped_hw, original_hw):
    """Native box_postprocess geometry, including its padding scale; no clipping."""
    a = np.asarray(boxes, dtype=np.float32)
    result = np.concatenate((a[:, :2] - a[:, 2:] / 2,
                             a[:, :2] + a[:, 2:] / 2), axis=1)
    sy = padded_hw[0] / mapped_hw[0] * original_hw[0]
    sx = padded_hw[1] / mapped_hw[1] * original_hw[1]
    return result * np.asarray([sx, sy, sx, sy], dtype=np.float32)


def stage_metrics(boxes, gt_boxes, query_ids=None):
    boxes = np.asarray(boxes).reshape(-1, 4)
    # Retain invalid boxes in raw exports; they cannot provide geometric support.
    valid = np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    matrix = np.zeros((len(gt_boxes), len(boxes)), dtype=np.float64)
    if len(gt_boxes) and valid.any():
        matrix[:, valid] = iou_matrix(gt_boxes, boxes[valid])
    best = matrix.max(1) if len(boxes) else np.zeros(len(gt_boxes))
    argmax = matrix.argmax(1) if len(boxes) else np.full(len(gt_boxes), -1)
    ids = np.arange(len(boxes)) if query_ids is None else np.asarray(query_ids)
    best_ids = [int(ids[j]) if j >= 0 and best[i] > 0 else None
                for i, j in enumerate(argmax)]
    return matrix, {
        "candidates": len(boxes), "invalid_or_degenerate_boxes": int((~valid).sum()),
        "max_iou": best.tolist(), "best_query_or_token_id": best_ids,
        "thresholds": {str(t): {
            "one_to_one_coverage": maximum_coverage(matrix, t),
            "marginal_gt_covered": int((best >= t).sum())} for t in THRESHOLDS}}


def mask_statistics(binary, valid_fraction):
    """Per-query native mask geometry; valid_fraction is cell image occupancy."""
    binary = np.asarray(binary, dtype=bool)
    count = binary.sum((1, 2))
    effective = (binary * valid_fraction[None]).sum((1, 2))
    rows = binary.any(2).sum(1)
    cols = binary.any(1).sum(1)
    return {"foreground_pixels": count, "foreground_fraction": count / np.prod(binary.shape[1:]),
            "valid_image_foreground_equivalent_pixels": effective,
            "padding_foreground_fraction": np.divide(count - effective, count,
                out=np.zeros_like(effective), where=count > 0),
            "empty": count == 0, "one_row": rows == 1, "one_column": cols == 1}


def valid_cell_fraction(shape, padded_hw, mapped_hw):
    h, w = shape
    y = np.clip(mapped_hw[0] / padded_hw[0] * h - np.arange(h), 0, 1)
    x = np.clip(mapped_hw[1] / padded_hw[1] * w - np.arange(w), 0, 1)
    return y[:, None] * x[None, :]


class NativeCapture:
    def __init__(self, model):
        from maskdino.utils import box_ops
        self.model = model
        self.decoder = model.sem_seg_head.predictor
        self.data = {}
        self.handlers = {
            self.decoder.forward.__func__.__code__: self.decoder_return,
            box_ops.masks_to_boxes.__code__: self.mask2box_return,
            model.instance_inference.__func__.__code__: self.instance_return,
            model.forward.__func__.__code__: self.model_return,
        }

    def mask2box_return(self, loc, result):
        if "raw_mask2box_xyxy" in self.data:
            raise AssertionError("Unexpected repeated native mask2box")
        self.data["raw_mask2box_xyxy"] = cpu(result)

    def decoder_return(self, loc, result):
        if loc["self"] is not self.decoder or "dense_boxes" in self.data:
            raise AssertionError("Unexpected decoder return")
        out, mask_dict = result
        assert mask_dict is None and not self.decoder.training
        assert self.decoder.two_stage and self.decoder.initial_pred
        assert self.decoder.initialize_box_type == "mask2box"
        assert not self.decoder.learn_tgt
        assert len(out["aux_outputs"]) == 9 and self.decoder.num_queries == 300
        assert out["pred_logits"].shape == (1, 300, 1)
        top = loc["topk_proposals"][0]
        dense = loc["enc_outputs_coord_unselected"].sigmoid()[0]
        selected = out["interm_outputs"]["pred_boxes"][0]
        import torch
        torch.testing.assert_close(dense[top], selected, rtol=0, atol=0)
        init = loc["refpoint_embed"].sigmoid()[0]
        torch.testing.assert_close(init, out["aux_outputs"][0]["pred_boxes"][0], rtol=0, atol=0)
        data = self.data
        data["dense_boxes"] = cpu(dense)
        data["dense_class_scores"] = cpu(loc["enc_outputs_class_unselected"][0].sigmoid().max(-1).values)
        data["native_legal"] = cpu(loc["output_proposals"][0].isfinite().all(-1) & ~loc["mask_flatten"][0])
        data["native_padding_mask"] = cpu(loc["mask_flatten"][0])
        data["spatial_shapes"] = cpu(loc["spatial_shapes"])
        data["top300_token_ids"] = cpu(top)
        data["encoder_top300"] = cpu(selected)
        data["decoder_init"] = cpu(init)
        for index, aux in enumerate(out["aux_outputs"][1:], 1):
            data[f"decoder_{index:02d}"] = cpu(aux["pred_boxes"][0])
        data["decoder_09"] = cpu(out["pred_boxes"][0])
        data["final_class_scores"] = cpu(out["pred_logits"][0].sigmoid().flatten())
        # Binary masks only, held on CPU for this image; no full floating masks exported.
        data["init_binary_masks"] = cpu(loc["flaten_mask"] > 0)

    def instance_return(self, loc, result):
        assert loc["self"] is self.model
        self.data["final_top100_query_ids"] = cpu(loc["topk_indices"])
        # Native locals have already sliced mask_box_result to Top100.
        self.data["native_top100_boxes"] = cpu(result.pred_boxes.tensor)
        self.data["native_top100_scores"] = cpu(result.scores)
        self.data["native_top100_class_scores"] = cpu(loc["scores_per_image"])
        self.data["native_top100_mask_scores"] = cpu(loc["mask_scores_per_image"])

    def model_return(self, loc, result):
        assert loc["self"] is self.model
        images = loc["images"]
        self.data["padded_hw"] = tuple(images.tensor.shape[-2:])
        self.data["mapped_hw"] = tuple(images.image_sizes[0])


def summarize_image(data, image, annotations, directory):
    """All GT-dependent work happens here, after model forward has completed."""
    padded, mapped = data["padded_hw"], data["mapped_hw"]
    original = (image["height"], image["width"])
    convert = lambda b: to_original(b, padded, mapped, original)
    top = data["top300_token_ids"]
    legal = data["native_legal"]
    cell_valid = np.concatenate([valid_cell_fraction(s, padded, mapped).reshape(-1) > .5
                                 for s in data["spatial_shapes"]])
    # This is a secondary diagnostic filter, never used to select native queries.
    stages = {
        "dense_native_legal": (convert(data["dense_boxes"][legal]), np.flatnonzero(legal)),
        "dense_valid_image_centers": (convert(data["dense_boxes"][legal & cell_valid]), np.flatnonzero(legal & cell_valid)),
        "encoder_top300": (convert(data["encoder_top300"]), np.arange(300)),
        "decoder_init": (convert(data["decoder_init"]), np.arange(300)),
    }
    for i in range(1, 10):
        stages[f"decoder_{i:02d}"] = (convert(data[f"decoder_{i:02d}"]), np.arange(300))
    stages["final_top100"] = (data["native_top100_boxes"], data["final_top100_query_ids"])
    np.testing.assert_allclose(stages["decoder_09"][0][data["final_top100_query_ids"]],
                               data["native_top100_boxes"], rtol=1e-6, atol=1e-3)
    gt = xyxy([a["bbox"] for a in annotations])
    report = {"image_id": image["id"], "file_name": image["file_name"],
              "center": image["center"], "sampling_stratum": image["sampling_stratum"],
              "gt": len(gt), "annotation_ids": [a["id"] for a in annotations],
              "padded_hw": padded, "mapped_hw": mapped, "stages": {}}
    arrays = {"gt_xyxy": gt, "top300_token_ids": top, "dense_native_legal": legal,
              "dense_valid_image_centers": cell_valid, "spatial_shapes": data["spatial_shapes"],
              "dense_class_scores": data["dense_class_scores"],
              "dense_all_boxes_xyxy": convert(data["dense_boxes"]),
              "final_class_scores": data["final_class_scores"],
              "final_top100_query_ids": data["final_top100_query_ids"],
              "final_top100_scores": data["native_top100_scores"],
              "final_top100_class_scores": data["native_top100_class_scores"],
              "final_top100_mask_scores": data["native_top100_mask_scores"],
              "mask2box_raw_lowres_xyxy": data["raw_mask2box_xyxy"]}
    matrices = {}
    for name, (boxes, ids) in stages.items():
        matrix, metrics = stage_metrics(boxes, gt, ids)
        report["stages"][name] = metrics
        matrices[name] = matrix
        arrays[name + "_boxes_xyxy"] = boxes
        arrays[name + "_gt_iou"] = matrix
    raw = data["raw_mask2box_xyxy"].astype(np.float32)
    mh, mw = data["init_binary_masks"].shape[-2:]
    raw_norm = raw / np.array([mw, mh, mw, mh], dtype=np.float32)
    raw_cxwh = np.concatenate(((raw_norm[:, :2] + raw_norm[:, 2:]) / 2,
                              raw_norm[:, 2:] - raw_norm[:, :2]), axis=1)
    arrays["mask2box_raw_original_xyxy"] = convert(raw_cxwh)
    arrays["mask2box_raw_gt_iou"], report["raw_mask2box_geometry"] = stage_metrics(arrays["mask2box_raw_original_xyxy"], gt)
    mask_stats = mask_statistics(data.pop("init_binary_masks"), valid_cell_fraction((mh, mw), padded, mapped))
    for key, value in mask_stats.items():
        arrays["mask_" + key] = value
    report["mask_summary"] = {key: int(value.sum()) for key, value in mask_stats.items()
                              if key in ("empty", "one_row", "one_column")}
    report["proposal_summary"] = {"dense_tokens": len(legal), "native_legal": int(legal.sum()),
        "native_masked_tokens": int(data["native_padding_mask"].sum()),
        "native_top300_not_legal": int((~legal[top]).sum()),
        "native_top300_outside_valid_image_centers": int((~cell_valid[top]).sum())}
    e, initial = stages["encoder_top300"][0], stages["decoder_init"][0]
    # Paired IoU without allocating a 300x300 matrix.
    lo, hi = np.maximum(e[:, :2], initial[:, :2]), np.minimum(e[:, 2:], initial[:, 2:])
    inter = np.maximum(hi-lo, 0).prod(1)
    union = np.maximum(e[:, 2:]-e[:, :2], 0).prod(1) + np.maximum(initial[:, 2:]-initial[:, :2], 0).prod(1)-inter
    arrays["encoder_init_same_query_iou"] = inter / np.maximum(union, 1e-12)
    sequence = ["dense_native_legal", "encoder_top300", "decoder_init"] + [f"decoder_{i:02d}" for i in range(1, 10)] + ["final_top100"]
    report["transitions"] = {}
    for a, b in zip(sequence, sequence[1:]):
        av, bv = np.array(report["stages"][a]["max_iou"]), np.array(report["stages"][b]["max_iou"])
        report["transitions"][a + "->" + b] = {"max_iou_delta": (bv-av).tolist(),
            "thresholds": {str(t): {"marginal_gained_ids": [annotations[i]["id"] for i in np.flatnonzero((av<t)&(bv>=t))],
                                    "marginal_lost_ids": [annotations[i]["id"] for i in np.flatnonzero((av>=t)&(bv<t))]}
                           for t in THRESHOLDS}}
    np.savez_compressed(directory / f"{image['id']}.npz", **arrays)
    write_json(directory / f"{image['id']}.json", report)
    return report


def remap_experiment(experiment, workspace):
    """Only relocate paths; preserve the hashed original configuration separately."""
    import copy
    from dataclasses import asdict
    from methods.mssl.config import MSSLOptions
    e = copy.deepcopy(experiment)
    oldrepo = Path(e["repo"]["path"])
    config_rel = Path(e["model"]["config"]).relative_to(oldrepo)
    e["repo"]["path"] = str(workspace / "third_party/MaskDINO")
    e["project_root"] = str(workspace)
    e["model"]["config"] = str(Path(e["repo"]["path"]) / config_rel)
    e["mssl"] = asdict(MSSLOptions(**e["mssl"]).validate())
    return e


def run(job_path):
    import torch
    job_path = Path(job_path).resolve()
    job = read_json(job_path)
    location = job.get("runtime_location")
    if location not in ("training_machine", "local_after_queue") or job.get("device") != "cuda":
        raise ValueError("Stage forward requires an explicit GPU execution location")
    if location == "local_after_queue":
        from .candidate_stage_local import authorize_child
        authorize_child(job["local_controller_config"])
    root = Path(job["workspace"]).resolve()
    expected = job["expected_sha256"]
    for key, digest in expected.items():
        if sha256(job[key]) != digest:
            raise RuntimeError(f"Input drift: {key}")
    native = root / "third_party/MaskDINO"
    commit = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    if commit != NATIVE_COMMIT:
        raise RuntimeError("Native revision differs from frozen protocol")
    for relative, digest in job["native_source_sha256"].items():
        if sha256(native / relative) != digest:
            raise RuntimeError(f"Native source drift: {relative}")
    for relative, digest in job["observer_source_sha256"].items():
        if sha256(root / relative) != digest:
            raise RuntimeError(f"Observer source drift: {relative}")
    out = Path(job["output"]); out.mkdir(parents=True, exist_ok=False)
    write_json(out / "state.json", {"status": "running", "job": str(job_path)})
    sys.path.insert(0, str(native))
    import train_net  # noqa: F401, native registry dependencies
    from detectron2.data import DatasetMapper
    from detectron2.modeling import build_model
    from methods.mssl.config import build_cfg
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    experiment = remap_experiment(read_json(job["experiment"]), root)
    cfg = build_cfg(experiment, out, device="cuda")
    (out / "resolved_cfg.yaml").write_text(cfg.dump(), encoding="utf-8")
    model = build_model(cfg)
    state = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True); del state
    model.eval()
    assert model.sem_seg_head.num_classes == 1 and not model.focus_on_box and not model.panoptic_on
    versions = {k: v._version for k, v in model.state_dict(keep_vars=True).items()}
    mapper = DatasetMapper(cfg, False)
    sample = read_json(job["sample"])["images"]
    gt = read_json(job["annotations"])
    anns = defaultdict(list)
    image_map = {im["id"]: im for im in gt["images"]}
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    references = np.load(job["reference"])
    positions = {int(i): n for n, i in enumerate(references["image_ids"])}
    reports = []; start = time.monotonic()
    directory = out / "images"; directory.mkdir()
    try:
        for image in sample:
            for key in ("file_name", "height", "width"):
                assert image[key] == image_map[image["id"]][key]
            path = Path(job["image_root"]) / image["file_name"]
            image_hash = sha256(path)
            if image_hash != job["image_sha256"][str(image["id"])]:
                raise RuntimeError(f"Image differs from historical reference: {image['id']}")
            record = {k: image[k] for k in ("height", "width")}
            record["file_name"] = str(path)
            mapped = mapper(record)
            assert set(mapped).isdisjoint(("instances", "annotations", "sem_seg", "targets"))
            with torch.no_grad():
                plain = model([mapped])[0]["instances"]
                plain_boxes, plain_scores = cpu(plain.pred_boxes.tensor), cpu(plain.scores)
                del plain
                capture = NativeCapture(model)
                with ReturnObserver(capture.handlers):
                    observed = model([mapped])[0]["instances"]
                np.testing.assert_array_equal(cpu(observed.pred_boxes.tensor), plain_boxes)
                np.testing.assert_array_equal(cpu(observed.scores), plain_scores)
                del observed
            data = capture.data
            np.savez_compressed(out / "last_parity_actual.npz", image_id=image["id"],
                boxes=plain_boxes, scores=plain_scores, query_ids=data["final_top100_query_ids"],
                all300_normalized=data["decoder_09"], all_class_scores=data["final_class_scores"])
            n = positions[image["id"]]
            np.testing.assert_array_equal(data["final_top100_query_ids"], references["query_ids"][n])
            np.testing.assert_allclose(plain_boxes, xyxy(references["bbox"][n]), rtol=1e-5, atol=1e-3)
            np.testing.assert_allclose(plain_scores, references["final_scores"][n], rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(data["final_class_scores"], references["all_query_class_scores"][n], rtol=1e-5, atol=1e-6)
            all_boxes = to_original(data["decoder_09"], data["padded_hw"], data["mapped_hw"], (image["height"], image["width"]))
            np.testing.assert_allclose(all_boxes, xyxy(references["all_query_boxes"][n]), rtol=1e-5, atol=1e-3)
            report = summarize_image(data, image, anns[image["id"]], directory)
            if sha256(path) != image_hash:
                raise RuntimeError("Image changed during trace")
            report["image_sha256"] = image_hash
            write_json(directory / f"{image['id']}.json", report)
            reports.append(report)
            write_json(out / "progress.json", {"images_done": len(reports), "images_total": len(sample),
                       "elapsed_seconds": time.monotonic()-start})
            del data, capture, mapped
        if versions != {k: v._version for k, v in model.state_dict(keep_vars=True).items()}:
            raise RuntimeError("Model tensors changed during read-only inference")
        for key, digest in expected.items():
            if sha256(job[key]) != digest:
                raise RuntimeError(f"Input changed: {key}")
        aggregate_reports(reports, out)
        write_json(out / "verification.json", {"status": "passed", "images": len(reports),
            "checkpoint_sha256": expected["checkpoint"], "native_commit": commit,
            "observed_vs_plain": "bitwise boxes and fused scores on every image",
            "frozen_reference": "all300 boxes/class scores and Top100 boxes/scores/query IDs on every image",
            "model_tensor_versions_unchanged": True, "gt_supplied_to_model": False,
            "elapsed_seconds": time.monotonic()-start, "sample_population_claim": False,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
        write_json(out / "state.json", {"status": "completed"})
    except BaseException as error:
        write_json(out / "state.json", {"status": "failed", "images_completed": len(reports),
                                        "error": repr(error)})
        raise


def aggregate_reports(reports, out):
    groups = defaultdict(list)
    for r in reports:
        for key in ("all", "center=" + r["center"], "stratum=" + r["sampling_stratum"]):
            groups[key].append(r)
    summary = {}
    for key, items in groups.items():
        stages = {}
        for name in items[0]["stages"]:
            values = [v for r in items for v in r["stages"][name]["max_iou"]]
            stages[name] = {"max_iou_mean": float(np.mean(values)) if values else None,
                "max_iou_median": float(np.median(values)) if values else None,
                "thresholds": {str(t): {metric: sum(r["stages"][name]["thresholds"][str(t)][metric] for r in items)
                    for metric in ("one_to_one_coverage", "marginal_gt_covered")} for t in THRESHOLDS}}
        summary[key] = {"images": len(items), "gt": sum(r["gt"] for r in items), "stages": stages}
    write_json(out / "summary.json", {"groups": summary, "population_prevalence_estimate": False,
        "limits": "Balanced mechanism sample; stage geometry is not a causal decomposition or AP attribution."})
    rows = []
    for r in reports:
        for j, aid in enumerate(r["annotation_ids"]):
            for name, stage in r["stages"].items():
                rows.append({"image_id": r["image_id"], "annotation_id": aid, "center": r["center"],
                    "stratum": r["sampling_stratum"], "stage": name,
                    "max_iou": stage["max_iou"][j], "best_query_or_token_id": stage["best_query_or_token_id"][j]})
    with (out / "per_gt.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["image_id"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    run(parser.parse_args().job)
