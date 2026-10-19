"""Freeze and execute a finite source-validation study; no training or tuning."""
import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from polypbench.io import read_json, sha256, write_json


RUNS = {"baseline": "maskdino-mssl-control-s42-20261018T025946", "mssl": "mssl-official50-s42-20261018T025946"}


def prepare(output):
    root = Path.cwd().resolve()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = out / "source"
    shutil.copytree(root / "polypbench", source / "polypbench", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "tests").mkdir()
    shutil.copyfile(root / "tests/test_quality_diagnostic.py", source / "tests/test_quality_diagnostic.py")
    release = read_json(root / "repro/mssl_v1_20261020/manifest.json")
    training, inputs = {}, {}
    image_digest = None
    for label, run_id in RUNS.items():
        run = root / "runs" / run_id
        experiment = read_json(run / "experiment.json")
        expected = release["models"][label]["checkpoint_sha256"]
        checkpoint = run / "train/model_best.pth"
        if sha256(checkpoint) != expected:
            raise RuntimeError("Frozen checkpoint changed")
        native = run / "predict_val/predictions.bbox.json"
        metrics = read_json(run / "predict_val/metrics.json")
        if sha256(native) != metrics["predictions_sha256"] or metrics["checkpoint_sha256"] != expected:
            raise RuntimeError("Saved native prediction identity changed")
        for path in (run / "experiment.json", checkpoint, native, run / "predict_val/metrics.json"):
            inputs[str(path)] = sha256(path)
        adapter = read_json(run / "adapter_sources.json")
        for rel, digest in adapter.items():
            historical = Path(experiment["project_root"]) / rel
            if sha256(historical) != digest:
                raise RuntimeError(f"Original training source changed: {historical}")
            inputs[str(historical)] = digest
        annotations = Path(experiment["dataset"]["val"]["annotations"])
        image_root = Path(experiment["dataset"]["val"]["image_root"])
        gt = read_json(annotations)
        frozen = read_json(run / "manifest.json")["dataset"]["val"]
        if sha256(annotations) != frozen["annotations_sha256"]:
            raise RuntimeError("Validation GT changed")
        digest = hashlib.sha256()
        for image in gt["images"]:
            path = image_root / image["file_name"]
            value = sha256(path)
            digest.update((image["file_name"] + ":" + value + "\n").encode())
            inputs[str(path)] = value
        if digest.hexdigest() != frozen["images_sha256"]:
            raise RuntimeError("Validation image fingerprint changed")
        if image_digest is not None and image_digest != digest.hexdigest():
            raise RuntimeError("Model validation cohorts differ")
        image_digest = digest.hexdigest()
        inputs[str(annotations)] = sha256(annotations)
        training[label] = {"experiment": str(run / "experiment.json"), "run_root": str(run),
                           "checkpoint": str(checkpoint), "checkpoint_sha256": expected,
                           "predictions": str(native)}
    rules = {
        "new_training": False, "model_updates": False, "new_quality_head": False,
        "cohort": "Kvasir Val, source checkpoint-selection/development cohort only; no PolypGen or new datasets",
        "same_gt_primary": "selected100; IoU>=.5 compatible with exactly one GT; no assignment of ambiguous boxes",
        "pair_difference_min": .05, "score_tie_absolute_tolerance": 1e-12,
        "primary_scores": ["native class*mask-quality", "class-only, same fixed100"],
        "inversions": "higher-IoU candidate scores strictly lower; report pair-micro and GT-macro denominators and score ties",
        "sensitivity": "IoU>=.25 unique-GT ownership, fixed in advance; descriptive only",
        "threshold_crossing": [.75, .9], "coverage_thresholds": [.5, .75, .9, .95],
        "coverage": "threshold-constrained maximum-cardinality one-to-one matching; all300 vs selected100 vs top1/10",
        "same_gt_permutation": "exchange original fused score multiset only within unique-GT IoU>=.5 groups; other scores unchanged",
        "image_gt_permutation": "exchange image's original fused score multiset in max-GT-IoU order; quality ties retain native score order; negative images unchanged",
        "direct_gt_iou": "all selected100 boxes retained; score=maxIoU or 0 on negatives; GT-assisted reference, not an optimal AP bound",
        "duplicates": "official greedy COCO matching, classify unmatched predictions; report top1/10/100 without confidence cut",
        "evaluation": "official COCOeval bbox; maxDets=[1,10,100], no filtering/NMS/TTA; no AP admission threshold or bootstrap",
        "interpretation": "source120 is small, no independent test inference or patient/seed stability claims; allow no-new-module decision",
    }
    protocol = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "workspace": str(root),
                "source_root": str(source), "annotations": str(annotations), "image_root": str(image_root),
                "input_sha256": inputs, "training": training, "rules": rules,
                "source_sha256": {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}}
    write_json(out / "protocol.json", protocol)
    write_json(out / "state.json", {"status": "prepared", "stage": "frozen", "completed": []})
    print(f"Frozen read-only study: {out}", flush=True)


def execute(output):
    out = Path(output).resolve()
    protocol = read_json(out / "protocol.json")
    source = Path(protocol["source_root"])
    state = {"status": "running", "stage": "verification", "completed": [], "pid": os.getpid()}

    def save():
        state["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(out / "state.json", state)

    def verify():
        for rel, digest in protocol["source_sha256"].items():
            if sha256(source / rel) != digest:
                raise RuntimeError(f"Study source changed: {rel}")
        for path, digest in protocol["input_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError(f"Input changed: {path}")

    def command(stage, argv):
        verify()
        state["stage"] = stage
        save()
        env = os.environ.copy()
        env.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        with (out / f"{stage}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen([sys.executable, *argv], cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = child.pid
            save()
            code = child.wait()
        if code:
            raise RuntimeError(f"{stage} failed ({code}); see {stage}.log")
        state["completed"].append(stage)
        save()

    save()
    try:
        command("unit_checks", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_quality_diagnostic.py", "-v"])
        for model, training in protocol["training"].items():
            job = {**training, "annotations": protocol["annotations"], "image_root": protocol["image_root"],
                   "output": str(out / "score_replay" / model)}
            path = out / f"{model}_job.json"
            write_json(path, job)
            command(f"{model}_replay", ["-m", "polypbench.analysis.score_replay", "--job", str(path)])
        command("diagnostics", ["-m", "polypbench.analysis.quality_study", "analyze", "--output", str(out)])
        verify()
        state.update(status="completed", stage="completed")
        save()
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run", "analyze"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.output)
    elif args.mode == "run":
        execute(args.output)
    else:
        from .quality_diagnostic import run
        run(args.output)
