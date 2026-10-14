import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .config import MASKDINO_COMMIT, load_config
from .data import validate_dataset
from .evaluation import evaluate
from .io import read_json, write_json
from .runner import command, environment, run, verify_repo
from .toy import create_toy


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native detector experiments and unified bbox COCO metrics")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare-repo")
    prepare.add_argument("--path", default="third_party/MaskDINO")
    toy = sub.add_parser("make-toy")
    toy.add_argument("--output", default="data/toy")
    conversion = sub.add_parser("convert-kvasir")
    conversion.add_argument("--source", required=True)
    conversion.add_argument("--output", required=True)
    conversion.add_argument("--seed", type=int, default=42)
    conversion.add_argument("--limit", type=int)
    conversion.add_argument("--min-area", type=int, default=5, help="Only for --bbox-source mask-components; supplied boxes are never filtered")
    conversion.add_argument("--split-dir", help="Use provided train.txt/val.txt and preserve the two-way split")
    conversion.add_argument("--bbox-source", choices=["provided", "mask-components"], default="provided", help="Default: use kavsir_bboxes.json exactly; mask-derived labels require explicit opt-in")
    for name in ("plan", "doctor", "validate", "train", "predict"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
        if name == "validate":
            item.add_argument("--skip-images", action="store_true")
        if name in ("train", "predict"):
            item.add_argument("--run-id", required=name == "predict")
        if name == "predict":
            item.add_argument("--split", choices=["val", "test"], default="val")
            item.add_argument("--checkpoint")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--annotations", required=True)
    evaluation.add_argument("--predictions", required=True)
    evaluation.add_argument("--output", required=True)
    visualization = sub.add_parser("preview")
    visualization.add_argument("--annotations", required=True)
    visualization.add_argument("--predictions", required=True)
    visualization.add_argument("--image-root", required=True)
    visualization.add_argument("--output", required=True)
    visualization.add_argument("--limit", type=int, default=8)
    visualization.add_argument("--max-predictions", type=int, default=5)
    summary = sub.add_parser("summarize")
    summary.add_argument("--runs", default="runs")
    summary.add_argument("--output", default="runs/results.csv")
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare-repo":
            path = Path(args.path).resolve()
            if path.exists():
                head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
                if head != MASKDINO_COMMIT:
                    raise ValueError(f"Existing checkout is {head}, expected {MASKDINO_COMMIT}; left untouched")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "https://github.com/IDEA-Research/MaskDINO.git", str(path)], check=True)
                subprocess.run(["git", "-C", str(path), "checkout", "--detach", MASKDINO_COMMIT], check=True)
            print(path)
        elif args.action == "make-toy":
            create_toy(args.output)
            print(f"Synthetic test data created: {args.output}")
        elif args.action == "convert-kvasir":
            from .kvasir import convert_kvasir
            audit = convert_kvasir(args.source, args.output, args.seed, args.limit, args.min_area, args.split_dir, args.bbox_source)
            print(json.dumps({"output": args.output, "bbox_source": audit["bbox_source"], "splits": {k: {"images": v["images"], "instances": v["instances"]} for k, v in audit["splits"].items()}}, indent=2))
        elif args.action == "evaluate":
            evaluate(args.annotations, args.predictions, args.output)
        elif args.action == "preview":
            from .preview import preview
            print(json.dumps(preview(args.annotations, args.predictions, args.image_root, args.output, args.limit, args.max_predictions), indent=2))
        elif args.action == "summarize":
            rows = []
            for file in sorted(Path(args.runs).glob("*/predict_*/metrics.json")):
                record = read_json(file)
                state = read_json(file.parent / "state.json")
                if state["status"] != "completed":
                    continue
                row = {key: record[key] for key in ("run_id", "name", "seed", "split", "supervision", "purpose", "annotations_sha256", "checkpoint_sha256")}
                row["checkpoint_selection"] = record.get("checkpoint_selection", "final")
                row.update({key: None if value is None else 100 * value for key, value in record["metrics"].items()})
                rows.append(row)
            if not rows:
                raise ValueError("No completed prediction metrics found")
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {len(rows)} runs to {target}; AP/AR units = percent")
        else:
            cfg = load_config(args.config)
            if args.action in ("doctor", "train", "predict") and os.name == "nt":
                raise ValueError("Model execution requires Linux/WSL2. On Windows use ./scripts/run_wsl.ps1 with the same command arguments")
            if args.action == "validate":
                print(json.dumps(validate_dataset(cfg, check_images=not args.skip_images), indent=2))
            elif args.action == "plan":
                print(json.dumps({"experiment": cfg, "note": "Plan only: no data/environment checks or training performed", "argv": command(cfg, "<run>/experiment.json", "<run>/train", "train")}, indent=2))
            elif args.action == "doctor":
                verify_repo(cfg)
                with tempfile.TemporaryDirectory() as temp:
                    snapshot = Path(temp) / "experiment.json"
                    write_json(snapshot, cfg)
                    subprocess.run(command(cfg, snapshot, "", "doctor"), cwd=cfg["repo"]["path"], env=environment(cfg), check=True)
            else:
                if args.action == "predict" and args.split not in cfg["dataset"]:
                    raise ValueError(f"Dataset split not configured: {args.split}")
                run(cfg, args.action, args.run_id, getattr(args, "split", "val"), getattr(args, "checkpoint", None))
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0
