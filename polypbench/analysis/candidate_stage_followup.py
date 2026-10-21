"""CPU follow-up of frozen stage traces: recovery, precision tradeoffs and masks.

No model inference, new thresholds, learned gates or AP experiments. Stage
events are observational and nonexclusive, not a causal partition of errors.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from polypbench.io import read_json, sha256, write_json


def run(traces, output):
    root, out = Path(traces), Path(output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []; summaries = {}; source_hashes = {}
    for model in ("baseline", "mssl"):
        directory = root / model
        for path in [directory / "verification.json", directory / "summary.json", *sorted((directory / "images").glob("*.json"))]:
            source_hashes[str(path)] = sha256(path)
        verification = read_json(directory / "verification.json")
        if verification["status"] != "passed" or verification["images"] != 72:
            raise RuntimeError("Require the fixed, fully verified 72-image trace")
        reports = [read_json(p) for p in sorted((directory / "images").glob("*.json"))]
        model_rows = []; masks = defaultdict(list)
        for report in reports:
            path = directory / "images" / f"{report['image_id']}.npz"
            source_hashes[str(path)] = sha256(path)
            with np.load(path) as a:
                for name in ("empty", "one_row", "one_column", "padding_foreground_fraction"):
                    masks[name].extend(a["mask_" + name].tolist())
                for j, aid in enumerate(report["annotation_ids"]):
                    best_q = int(a["encoder_top300_gt_iou"][j].argmax())
                    e = a["encoder_top300_boxes_xyxy"][best_q]
                    m = a["decoder_init_boxes_xyxy"][best_q]
                    ew, eh = np.maximum(e[2:] - e[:2], 1e-9)
                    mw, mh = np.maximum(m[2:] - m[:2], 0)
                    norm_center_shift = float(np.linalg.norm(((m[:2]+m[2:])-(e[:2]+e[2:]))/2/[ew,eh]))
                    row = {"model": model, "image_id": report["image_id"], "annotation_id": aid,
                        "center": report["center"], "stratum": report["sampling_stratum"],
                        "multiplicity": "single" if report["gt"] == 1 else "multi",
                        "best_encoder_query": best_q, "same_query_encoder_iou": float(a["encoder_top300_gt_iou"][j,best_q]),
                        "same_query_init_iou": float(a["decoder_init_gt_iou"][j,best_q]),
                        "same_query_final_iou": float(a["decoder_09_gt_iou"][j,best_q]),
                        "best_query_mask_empty": bool(a["mask_empty"][best_q]),
                        "best_query_mask_one_row_or_column": bool(a["mask_one_row"][best_q] or a["mask_one_column"][best_q]),
                        "best_query_padding_foreground_fraction": float(a["mask_padding_foreground_fraction"][best_q]),
                        "best_query_init_encoder_area_ratio": float(mw*mh/(ew*eh)),
                        "best_query_init_encoder_center_shift": norm_center_shift,
                        "best_query_encoder_init_box_iou": float(a["encoder_init_same_query_iou"][best_q])}
                    for stage in ("dense_native_legal", "encoder_top300", "decoder_init", "decoder_09", "final_top100"):
                        row[stage] = report["stages"][stage]["max_iou"][j]
                    rows.append(row); model_rows.append(row)
        groups = {"all": model_rows}
        groups.update({key: [r for r in model_rows if r["multiplicity"] == key] for key in ("single", "multi")})
        groups.update({"center="+c: [r for r in model_rows if r["center"] == c] for c in sorted({r["center"] for r in model_rows})})
        groups_out = {}
        for name, items in groups.items():
            thresholds = {}
            for t in (.5, .75, .9):
                lost = [r for r in items if r["encoder_top300"] >= t and r["decoder_init"] < t]
                gained = [r for r in items if r["encoder_top300"] < t and r["decoder_init"] >= t]
                select_lost = [r for r in items if r["dense_native_legal"] >= t and r["encoder_top300"] < t]
                thresholds[str(t)] = {
                    "marginal_covered": {stage: sum(r[stage] >= t for r in items) for stage in (
                        "dense_native_legal", "encoder_top300", "decoder_init", "decoder_09", "final_top100")},
                    "init_lost": len(lost), "init_gained": len(gained),
                    "init_lost_recovered_by_final300": sum(r["decoder_09"] >= t for r in lost),
                    "init_lost_recovered_by_top100": sum(r["final_top100"] >= t for r in lost),
                    "init_gained_retained_by_final300": sum(r["decoder_09"] >= t for r in gained),
                    "selection_lost": len(select_lost),
                    "selection_lost_recovered_by_final300": sum(r["decoder_09"] >= t for r in select_lost),
                    "init_lost_best_encoder_query_empty": sum(r["best_query_mask_empty"] for r in lost),
                    "init_lost_best_encoder_query_degenerate": sum(r["best_query_mask_one_row_or_column"] for r in lost),
                    "init_lost_median_area_ratio": float(np.median([r["best_query_init_encoder_area_ratio"] for r in lost])) if lost else None,
                    "init_lost_median_center_shift": float(np.median([r["best_query_init_encoder_center_shift"] for r in lost])) if lost else None,
                    "init_lost_median_padding_fraction": float(np.median([r["best_query_padding_foreground_fraction"] for r in lost])) if lost else None,
                    "init_lost_gt_ids": [r["annotation_id"] for r in lost]}
            groups_out[name] = {"gt": len(items), "thresholds": thresholds}
        summaries[model] = {"groups": groups_out, "all_selected_masks": {
            "count": len(masks["empty"]), "empty": sum(masks["empty"]),
            "one_row": sum(masks["one_row"]), "one_column": sum(masks["one_column"]),
            "padding_foreground_positive": sum(v > 0 for v in masks["padding_foreground_fraction"])}}
        original = read_json(directory / "summary.json")["groups"]["all"]["stages"]
        for threshold, values in groups_out["all"]["thresholds"].items():
            for stage, number in values["marginal_covered"].items():
                if number != original[stage]["thresholds"][threshold]["marginal_gt_covered"]:
                    raise RuntimeError("Follow-up counts disagree with native stage report")
    write_json(out / "per_gt.json", rows)
    write_json(out / "summary.json", {"models": summaries,
        "limits": "Balanced mechanism sample, marginal events overlap. Empty/area/center proxies are observational, not causal or calibrated reliability labels.",
        "source_npz_sha256": source_hashes})
    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise RuntimeError("Frozen trace changed")
    print({label: value["groups"]["all"] for label, value in summaries.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True); parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))
