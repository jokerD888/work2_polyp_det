"""CPU report from completed, parity-verified stage observations; no inference."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from polypbench.io import read_json, write_json
from .candidate_stage import THRESHOLDS


SEQUENCE = ["dense_native_legal", "encoder_top300", "decoder_init"] + [f"decoder_{i:02d}" for i in range(1,10)] + ["final_top100"]


def report(queue_root, image_root, output):
    root = Path(queue_root); out = Path(output); out.mkdir(parents=True, exist_ok=False)
    model_reports = {}; results = {}; verifications = {}
    for model in ("baseline", "mssl"):
        verification = read_json(root / model / "verification.json")
        if verification["status"] != "passed":
            raise RuntimeError("Only parity-verified traces can enter the report")
        verifications[model] = verification
        model_reports[model] = {r["image_id"]: r for r in
            (read_json(path) for path in sorted((root / model / "images").glob("*.json")))}
    assert set(model_reports["baseline"]) == set(model_reports["mssl"])
    for iid in model_reports["baseline"]:
        assert model_reports["baseline"][iid]["annotation_ids"] == model_reports["mssl"][iid]["annotation_ids"]
    full = verifications["baseline"].get("cohort_kind") == "full_polypgen"
    if verifications["baseline"].get("cohort_kind") != verifications["mssl"].get("cohort_kind"):
        raise RuntimeError("Model cohorts differ")
    scope = (f"Complete fixed PolypGen cohort: {len(model_reports['baseline'])} images, including negatives. Frequencies describe this benchmark; they are not causal effects or clinical population prevalence."
             if full else "Balanced mechanism sample. Counts are not population prevalence, AP attribution or causal effects.")
    lines = ["# Candidate-stage mechanism report", "", scope, "",
        "Initial aux[0] boxes equal actual initialization; Decoder 1 starts at aux[1]. Same-query IoU trajectories are in each image NPZ.", ""]
    examples = []
    for model, reports in model_reports.items():
        transitions = {}; lines.extend([f"## {model}", "", "| Transition | GT | Mean max-IoU delta | Marginal lost/gained @.5 |", "|---|---:|---:|---:|"])
        for a, b in zip(SEQUENCE, SEQUENCE[1:]):
            delta = []; events = {str(t): {"gained": 0, "lost": 0} for t in THRESHOLDS}
            for r in reports.values():
                values = r["transitions"][a+"->"+b]
                delta.extend(values["max_iou_delta"])
                for t in THRESHOLDS:
                    v = values["thresholds"][str(t)]
                    events[str(t)]["gained"] += len(v["marginal_gained_ids"])
                    events[str(t)]["lost"] += len(v["marginal_lost_ids"])
            transitions[a+"->"+b] = {"gt": len(delta), "mean_max_iou_delta": float(np.mean(delta)),
                "median_max_iou_delta": float(np.median(delta)), "marginal_threshold_events": events}
            v = transitions[a+"->"+b]
            lines.append(f"| {a} → {b} | {len(delta)} | {v['mean_max_iou_delta']:+.4f} | {events['0.5']['lost']}/{events['0.5']['gained']} |")
        results[model] = {"transitions": transitions, "coverage": read_json(root / model / "summary.json")["groups"]}
        lines.extend(["", "One-to-one counts, marginal maxima and center/stratum denominators are separately available in summary.json and per_gt.csv.", ""])
        # Show both deterioration and improvement, including later recovery.
        # Selected after tracing for illustration, never for cohort selection.
        for name, a, b in (("init_decline", "encoder_top300", "decoder_init"),
                           ("decoder_decline", "decoder_init", "decoder_09"),
                           ("decoder_improve", "decoder_init", "decoder_09")):
            candidates = []
            for r in reports.values():
                for j, aid in enumerate(r["annotation_ids"]):
                    va, vb = r["stages"][a]["max_iou"][j], r["stages"][b]["max_iou"][j]
                    if (name.endswith("improve") and vb > va) or (not name.endswith("improve") and vb < va):
                        candidates.append((vb-va, r["image_id"], j, aid))
            if candidates:
                chosen = sorted(candidates, reverse=name.endswith("improve"))[0]
                delta, iid, index, aid = chosen
                examples.append({"model": model, "kind": name, "image_id": iid, "annotation_id": aid,
                                 "gt_index": index, "delta": delta})
        # One selection example when any marginal coverage is lost.
        for a, b, name in (("dense_native_legal", "encoder_top300", "encoder_selection_loss"),
                            ("decoder_09", "final_top100", "final_selection_loss")):
            candidates = []
            for r in reports.values():
                for j, aid in enumerate(r["annotation_ids"]):
                    av, bv = r["stages"][a]["max_iou"][j], r["stages"][b]["max_iou"][j]
                    if av >= .5 and bv < .5:
                        candidates.append((bv-av, r["image_id"], j, aid))
            if candidates:
                delta, iid, index, aid = sorted(candidates)[0]
                examples.append({"model": model, "kind": name, "image_id": iid, "annotation_id": aid,
                                 "gt_index": index, "delta": delta})
    write_json(out / "comparison.json", {"models": results, "illustrations": examples,
        "population_prevalence_estimate": False, "unique_dominant_cause_claim": False})
    render_examples(examples, model_reports, root, Path(image_root), out)
    lines.extend(["## Interpretation boundary", "",
        "Multiple stage failures may coexist. Dense geometric deficits do not establish poor semantic separability; a mask anomaly does not establish causation.",
        "", "The next mechanism requires inspection of the exported examples and the independent Kvasir LSJ eligibility report. No new module or threshold sweep is selected automatically.",
        "", "![Selected mechanism examples](examples.jpg)" if examples else "No examples matched the predefined illustrative categories.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def render_examples(examples, reports, root, image_root, out):
    if not examples:
        return
    stages = ["dense_native_legal", "encoder_top300", "decoder_init", "decoder_09", "final_top100"]
    canvas = Image.new("RGB", (1500, len(examples)*250), "white")
    draw = ImageDraw.Draw(canvas)
    for n, ex in enumerate(examples):
        r = reports[ex["model"]][ex["image_id"]]
        with np.load(root / ex["model"] / "images" / f"{ex['image_id']}.npz") as data:
            for col, stage in enumerate(stages):
                image = Image.open(image_root / r["file_name"]).convert("RGB")
                pen = ImageDraw.Draw(image)
                gt = data["gt_xyxy"][ex["gt_index"]]
                pen.rectangle(tuple(gt), outline="yellow", width=4)
                matrix = data[stage + "_gt_iou"]
                if matrix.shape[1] and matrix[ex["gt_index"]].max() > 0:
                    box = data[stage + "_boxes_xyxy"][matrix[ex["gt_index"]].argmax()]
                    if np.isfinite(box).all() and (box[2:] >= box[:2]).all():
                        pen.rectangle(tuple(box), outline="cyan", width=4)
                image.thumbnail((294,185)); canvas.paste(image, (col*300+3,n*250+60))
                value = r["stages"][stage]["max_iou"][ex["gt_index"]]
                draw.text((col*300+3,n*250+35), f"{stage} IoU={value:.3f}", fill="black")
        draw.text((3,n*250+5), f"{ex['model']} id={ex['image_id']} GT={ex['annotation_id']} {ex['kind']} delta={ex['delta']:+.3f}; yellow GT/cyan stage best (query may change)", fill="black")
    canvas.save(out / "examples.jpg", quality=92)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    report(**vars(parser.parse_args()))
