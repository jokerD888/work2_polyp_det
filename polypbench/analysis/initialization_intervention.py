"""Frozen-weight initialization intervention, pinned to the native forward body.

Compile a private copy of forward with ONE assignment inserted before tgt_mask.
No installed native source, TopK, content query, mask conversion or scoring is edited.
The native condition returns the exact original reference tensor (no round trip).
"""
import argparse
import ast
from collections import defaultdict
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import types

import numpy as np

from polypbench.io import read_json, sha256, write_json
from .candidate_stage import (NativeCapture, ReturnObserver, NATIVE_COMMIT,
    aggregate_reports, cpu, remap_experiment, summarize_image)

CONDITIONS = ("native", "encoder", "blend05")


class InitializationIntervention:
    def __init__(self, decoder):
        self.decoder = decoder
        self.condition = "native"
        self.calls = 0
        self.data = {}

    def replace(self, encoder_logits, mask_logits, top_ids, content, masks):
        import torch
        from maskdino.utils.utils import inverse_sigmoid
        assert not self.decoder.training and self.decoder.two_stage
        assert self.decoder.initialize_box_type == "mask2box" and self.decoder.initial_pred
        assert not self.decoder.learn_tgt and encoder_logits.shape == mask_logits.shape == (1, 300, 4)
        encoder, mask = encoder_logits.sigmoid(), mask_logits.sigmoid()
        # Native dense proposals may contain +/-inf logits; the native decoder
        # applies sigmoid first, yielding finite boundary references. Preserve it.
        if not torch.isfinite(encoder).all() or not torch.isfinite(mask).all():
            raise RuntimeError("Non-finite normalized reference; do not invent a fallback")
        if self.condition == "native":
            result = mask_logits  # bitwise identity, including native numerical edge handling
        elif self.condition == "encoder":
            result = encoder_logits  # exactly native INITIALIZE_BOX_TYPE=no reference
        elif self.condition == "blend05":
            result = inverse_sigmoid(.5 * encoder + .5 * mask)
        else:
            raise ValueError(self.condition)
        self.calls += 1
        self.data = {"top300_token_ids": cpu(top_ids[0]), "content": cpu(content),
            "encoder": cpu(encoder[0]), "mask": cpu(mask[0]),
            "actual": cpu(result.sigmoid()[0]), "mask_logits": cpu(masks),
            "encoder_logits": cpu(encoder_logits), "mask_reference_logits": cpu(mask_logits)}
        return result

    def __enter__(self):
        original = self.decoder.forward
        self.original = original
        self.had_instance_forward = "forward" in self.decoder.__dict__
        tree = ast.parse(textwrap.dedent(inspect.getsource(original.__func__)))
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef) or function.decorator_list:
            raise RuntimeError("Unexpected native forward structure")
        indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "tgt_mask" for t in node.targets)]
        if len(indices) != 1:
            raise RuntimeError("Native intervention point drift")
        insertion = ast.parse("refpoint_embed = _initialization_intervention(refpoint_embed_undetach.detach(), refpoint_embed, topk_proposals, tgt, outputs_mask)").body[0]
        function.body.insert(indices[0], insertion)
        ast.fix_missing_locations(tree)
        namespace = dict(original.__func__.__globals__)
        namespace["_initialization_intervention"] = self.replace
        exec(compile(tree, "<frozen-init-intervention>", "exec"), namespace)
        self.decoder.forward = types.MethodType(namespace[function.name], self.decoder)
        return self

    def __exit__(self, *exc):
        if self.had_instance_forward:
            self.decoder.forward = self.original
        else:
            del self.decoder.forward


def assert_same_upstream(native, other):
    for key in ("top300_token_ids", "content", "encoder", "mask", "mask_logits",
                "encoder_logits", "mask_reference_logits"):
        np.testing.assert_array_equal(native[key], other[key], err_msg="Upstream changed: " + key)


def compare_capture(a, b):
    if a.keys() != b.keys():
        raise AssertionError("Capture fields differ")
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg="Native identity: " + key)


def coco_rows(image_id, data, category_id):
    boxes = data["native_top100_boxes"].copy()
    boxes[:, 2:] -= boxes[:, :2]
    return [{"image_id": image_id, "category_id": category_id,
        "bbox": box.tolist(), "score": float(score)}
        for box, score in zip(boxes, data["native_top100_scores"])]


def signed_events(base, alternative, threshold):
    base, alternative = np.asarray(base), np.asarray(alternative)
    gained = (base < threshold) & (alternative >= threshold)
    lost = (base >= threshold) & (alternative < threshold)
    return {"recovered": int(gained.sum()), "newly_lost": int(lost.sum()),
        "net": int(gained.sum() - lost.sum())}


def evaluate_condition(gt, predictions, reports, output):
    from .coco_cache import official_eval, ap_vector
    from .quality_diagnostic import fp_types
    ev = official_eval(gt, predictions)
    vector = ap_vector(ev)
    record = {"AP": float(vector.mean()), "AP50": float(vector[0]),
        "AP75": float(vector[5]), "AP90": float(vector[8]), "AP95": float(vector[9]),
        "units": "fraction_0_to_1", "ap_by_iou": vector.tolist(),
        "official_fp_types": fp_types(ev, gt),
        "negative_images": sum(r["gt"] == 0 for r in reports),
        "negative_images_with_score_ge_05": sum(r["gt"] == 0 and r["native_top100_score_ge_05"] > 0 for r in reports),
        "negative_detections_score_ge_05": sum(r["native_top100_score_ge_05"] for r in reports if r["gt"] == 0)}
    write_json(output / "predictions.bbox.json", predictions)
    write_json(output / "evaluation.json", record)
    return record


def comparison(reports, output):
    """Compare FINAL results against native, never just the historical lost group."""
    groups = defaultdict(lambda: defaultdict(list))
    rows = []
    for condition in CONDITIONS[1:]:
        for base, other in zip(reports["native"], reports[condition]):
            assert base["image_id"] == other["image_id"] and base["annotation_ids"] == other["annotation_ids"]
            for j, aid in enumerate(base["annotation_ids"]):
                row = {"condition": condition, "image_id": base["image_id"], "annotation_id": aid,
                    "center": base["center"], "multiplicity": "single" if base["gt"] == 1 else "multi",
                    **base["gt_metadata"][j], "stages": {}}
                for stage in ("decoder_init", "decoder_09", "final_top100"):
                    row["stages"][stage] = {"native": base["stages"][stage]["max_iou"][j],
                        "alternative": other["stages"][stage]["max_iou"][j]}
                row["historical_native_init_lost_05"] = (base["stages"]["encoder_top300"]["max_iou"][j] >= .5
                    and base["stages"]["decoder_init"]["max_iou"][j] < .5)
                rows.append(row)
                for group in ("all", "center="+row["center"], row["multiplicity"],
                    "area="+str(row["relative_area_quartile"]),
                    "area="+str(row["relative_area_quartile"])+"/"+row["multiplicity"]):
                    groups[condition][group].append(row)
    result = {}
    for condition, subsets in groups.items():
        result[condition] = {}
        for group, items in subsets.items():
            result[condition][group] = {"gt": len(items), "stages": {}}
            for stage in ("decoder_init", "decoder_09", "final_top100"):
                a = [r["stages"][stage]["native"] for r in items]
                b = [r["stages"][stage]["alternative"] for r in items]
                result[condition][group]["stages"][stage] = {str(t): signed_events(a,b,t) for t in (.5,.75,.9)}
            lost = [r for r in items if r["historical_native_init_lost_05"]]
            result[condition][group]["historical_init_lost"] = {"gt": len(lost),
                "native_final300_covered": sum(r["stages"]["decoder_09"]["native"] >= .5 for r in lost),
                "alternative_final300_covered": sum(r["stages"]["decoder_09"]["alternative"] >= .5 for r in lost)}
    write_json(output / "per_gt_comparison.json", rows)
    write_json(output / "comparison.json", result)
    return result


def run(job_path):
    import torch
    job = read_json(job_path)
    from .initialization_queue import authorize_child
    authorize_child(job["controller_config"])
    root, out = Path(job["workspace"]), Path(job["output"])
    for path, digest in job["inputs_sha256"].items():
        if sha256(path) != digest:
            raise RuntimeError("Input drift: " + path)
    native = root / "third_party/MaskDINO"
    commit = subprocess.check_output(["git", "-C", str(native), "rev-parse", "HEAD"], text=True).strip()
    assert commit == NATIVE_COMMIT
    for relative, digest in job["native_source_sha256"].items():
        if sha256(native / relative) != digest:
            raise RuntimeError("Native source drift: " + relative)
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(native))
    import train_net  # noqa: F401
    from detectron2.data import DatasetMapper
    from detectron2.modeling import build_model
    from methods.mssl.config import build_cfg
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    cfg = build_cfg(remap_experiment(read_json(job["experiment"]), root), out, device="cuda")
    model = build_model(cfg)
    state = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True); del state
    model.eval()
    versions = {k: v._version for k,v in model.state_dict(keep_vars=True).items()}
    mapper = DatasetMapper(cfg, False)
    gt = read_json(job["annotations"])
    sample = read_json(job["sample"])["images"]
    assert len(sample) == job["expected_images"] and len({im["id"] for im in sample}) == len(sample)
    annotations = defaultdict(list)
    for ann in gt["annotations"]:
        annotations[ann["image_id"]].append(ann)
    reports = {c: [] for c in CONDITIONS}; predictions = {c: [] for c in CONDITIONS}
    for condition in CONDITIONS:
        (out / condition / "images").mkdir(parents=True)
    old_predictions = defaultdict(list)
    if job.get("historical_predictions"):
        for p in read_json(job["historical_predictions"]):
            old_predictions[p["image_id"]].append(p)
    start = time.monotonic()
    try:
        for image in sample:
            path = Path(job["image_root"]) / image["file_name"]
            assert sha256(path) == job["image_sha256"][str(image["id"])]
            mapped = mapper({"file_name": str(path), "height": image["height"], "width": image["width"]})
            assert set(mapped).isdisjoint(("instances", "annotations", "sem_seg", "targets"))
            plain_data = None
            if job["smoke"]:
                capture = NativeCapture(model)
                with torch.no_grad(), ReturnObserver(capture.handlers):
                    model([mapped])
                plain_data = capture.data
            upstream = None
            with InitializationIntervention(model.sem_seg_head.predictor) as intervention:
                for condition in CONDITIONS:
                    intervention.condition = condition
                    capture = NativeCapture(model)
                    with torch.no_grad(), ReturnObserver(capture.handlers):
                        model([mapped])
                    data = capture.data
                    if condition == "native":
                        upstream = intervention.data
                        if plain_data is not None:
                            compare_capture(plain_data, data)
                    else:
                        assert_same_upstream(upstream, intervention.data)
                    np.testing.assert_array_equal(data["decoder_init"], intervention.data["actual"])
                    report = summarize_image(data, image, annotations[image["id"]],
                        out / condition / "images", job["relative_area_quartile_cuts"])
                    report["reference_numerics"] = {
                        "encoder_infinite_logit_coordinates": int(np.isinf(upstream["encoder_logits"]).sum()),
                        "encoder_normalized_boundary_coordinates": int(((upstream["encoder"] == 0) | (upstream["encoder"] == 1)).sum()),
                        "actual_normalized_boundary_coordinates": int(((intervention.data["actual"] == 0) | (intervention.data["actual"] == 1)).sum()),
                        "extra_fallback": False}
                    write_json(out / condition / "images" / f"{image['id']}.json", report)
                    archive = out / condition / "images" / f"{image['id']}.npz"
                    if condition == "native" and job.get("historical_stages"):
                        historical_path = Path(job["historical_stages"])/"images"/archive.name
                        assert sha256(historical_path) == job["historical_stage_sha256"][str(image["id"])]
                        with np.load(historical_path) as old, np.load(archive) as new:
                            for key in old.files:
                                np.testing.assert_array_equal(old[key], new[key], err_msg="Historical native: "+key)
                    # Dense upstream is invariant and already preserved in the historical census.
                    # Keep all Query/Decoder boxes, scores and per-GT trajectories for each condition.
                    with np.load(archive) as saved:
                        compact = {k: saved[k] for k in saved.files if k not in
                            ("dense_all_boxes_xyxy", "dense_class_scores", "dense_native_legal", "dense_valid_image_centers")}
                    compact.update(init_encoder_normalized=upstream["encoder"], init_mask_normalized=upstream["mask"])
                    np.savez_compressed(archive, **compact)
                    row = coco_rows(image["id"], data, gt["categories"][0]["id"])
                    if condition == "native" and old_predictions:
                        historical = old_predictions[image["id"]]
                        np.testing.assert_allclose([p["bbox"] for p in row], [p["bbox"] for p in historical], rtol=1e-5, atol=1e-3)
                        np.testing.assert_allclose([p["score"] for p in row], [p["score"] for p in historical], rtol=1e-5, atol=1e-6)
                    reports[condition].append(report); predictions[condition].extend(row)
                    write_json(out / "progress.json", {"images_done": len(reports["blend05"]),
                        "images_total": len(sample), "condition": condition, "image_id": image["id"],
                        "elapsed_seconds": time.monotonic()-start})
                    del data, capture
                assert intervention.calls == 3
            assert sha256(path) == job["image_sha256"][str(image["id"])]
        assert versions == {k: v._version for k,v in model.state_dict(keep_vars=True).items()}
        assert sum(r["gt"] for r in reports["native"]) == job["expected_gt"]
        evaluation = {}
        (out / "resolved_cfg.yaml").write_text(cfg.dump(), encoding="utf-8")
        for condition in CONDITIONS:
            aggregate_reports(reports[condition], out / condition, job["cohort"])
            evaluation[condition] = evaluate_condition(gt if not job["smoke"] else
                {**gt, "images": sample, "annotations": [a for a in gt["annotations"] if a["image_id"] in {i["id"] for i in sample}]},
                predictions[condition], reports[condition], out / condition)
        changes = comparison(reports, out)
        for path, digest in job["inputs_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError("Input changed during inference: " + path)
        write_json(out / "verification.json", {"status": "passed", "images": len(sample),
            "gt": job["expected_gt"], "training": False, "gt_supplied_to_model": False,
            "upstream_bitwise_identity_all_conditions": True, "model_state_versions_unchanged": True,
            "native_vs_unmodified_capture_bitwise": job["smoke"],
            "historical_stage_arrays_bitwise": bool(job.get("historical_stages")),
            "elapsed_seconds": time.monotonic()-start})
        write_json(out / "result.json", {"evaluation": evaluation, "changes": changes})
    except BaseException as error:
        write_json(out / "failure.json", {"error": repr(error), "images_done": len(reports["blend05"])}); raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    run(parser.parse_args().job)
