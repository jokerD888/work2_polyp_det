"""Independent F-v1 runner, adapted from the current polypbench runner.

It reuses protocol/data/evaluation utilities, but owns its worker command and
source snapshot. The baseline runner and its ongoing pipeline are not modified.
"""
import datetime as dt
import os
import re
import subprocess
from pathlib import Path

from polypbench.config import fingerprint
from polypbench.data import validate_dataset
from polypbench.evaluation import evaluate
from polypbench.io import read_json, sha256, write_json


def git_info(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, encoding="utf-8").strip()
    return {"commit": git("rev-parse", "HEAD"), "remote": git("remote", "get-url", "origin"), "status": git("status", "--porcelain"), "diff": git("diff", "HEAD")}


def verify_repo(cfg):
    if not Path(cfg["model"]["config"]).is_file():
        raise ValueError("Native model config missing. Run prepare-repo first")
    info = git_info(cfg["repo"]["path"])
    if info["commit"] != cfg["repo"]["commit"]:
        raise ValueError(f"MaskDINO commit mismatch: {info['commit']} != {cfg['repo']['commit']}")
    return info


def adapter_sources(cfg):
    root = Path(cfg["project_root"])
    files = ["polypbench/config.py", "polypbench/data.py", "polypbench/evaluation.py",
             "polypbench/io.py", "polypbench/adapters/bitmask_mapper.py",
             "polypbench/adapters/instance_lsj.py", "methods/__init__.py"]
    files.extend(path.relative_to(root).as_posix()
                 for path in sorted((root / "methods" / "scda_proposal").rglob("*"))
                 if path.is_file() and path.suffix in (".py", ".yaml"))
    return {file: sha256(root / file) for file in files}


def command(cfg, snapshot, output, mode, split="val", checkpoint=None):
    result = [cfg["runtime"]["python"], "-m", "methods.scda_proposal.worker", "--experiment", str(snapshot), "--output", str(output), "--mode", mode, "--split", split]
    if checkpoint:
        result += ["--checkpoint", str(checkpoint)]
    return result


def environment(cfg):
    env = os.environ.copy()
    env["PYTHONPATH"] = cfg["project_root"] + os.pathsep + env.get("PYTHONPATH", "")
    if "visible_gpus" in cfg["runtime"]:
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["runtime"]["visible_gpus"])
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_process(args, cfg, output):
    output = Path(output)
    write_json(output / "command.json", {"argv": args, "cwd": cfg["repo"]["path"], "visible_gpus": cfg["runtime"].get("visible_gpus")})
    with (output / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(args, cwd=cfg["repo"]["path"], env=environment(cfg), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Model process exited with code {code}; see {output / 'console.log'}")


def run(cfg, mode, run_id=None, split="val", checkpoint=None):
    repo = verify_repo(cfg)
    data = validate_dataset(cfg)  # category mapping and split overlap checked together
    run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{cfg['name']}-s{cfg['seed']}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run-id")
    root = Path(cfg["output_root"]) / run_id
    checkpoint_rule = "explicit" if checkpoint else cfg.get("checkpoint_selection", "final")
    if mode == "train":
        root.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": run_id, "name": cfg["name"], "adapter": cfg["adapter"], "seed": cfg["seed"], "supervision": cfg["dataset"]["supervision"], "purpose": cfg.get("purpose", "research"), "config_fingerprint": fingerprint(cfg), "dataset": data, "repo": repo, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(root / "manifest.json", manifest)
        write_json(root / "experiment.json", cfg)
        sources = adapter_sources(cfg)
        write_json(root / "adapter_sources.json", sources)
        for file in sources:
            source = Path(cfg["project_root"]) / file
            target = root / "source" / file
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        conversion = Path(cfg["dataset"]["train"]["annotations"]).parent / "conversion.json"
        if conversion.exists():
            write_json(root / "data_conversion.json", read_json(conversion))
    else:
        if not root.is_dir():
            raise ValueError("predict needs an existing training run-id")
        manifest = read_json(root / "manifest.json")
        if manifest["config_fingerprint"] != fingerprint(cfg) or manifest["dataset"] != data:
            raise ValueError("Experiment or dataset changed since training; use the original configuration/data")
        if manifest["repo"] != repo:
            raise ValueError("Native repository changed since training; restore the recorded code before prediction")
        if (root / "adapter_sources.json").exists() and read_json(root / "adapter_sources.json") != adapter_sources(cfg):
            raise ValueError("Adapter code changed since training; restore the recorded adapter before prediction")
        filename = "model_best.pth" if checkpoint_rule == "best_val_ap" else "model_final.pth"
        checkpoint = Path(checkpoint).resolve() if checkpoint else root / "train" / filename
        if not checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint}")
    stage = root / ("train" if mode == "train" else f"predict_{split}")
    stage.mkdir(exist_ok=False)
    state = {"status": "running", "mode": mode, "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(stage / "state.json", state)
    try:
        if checkpoint:
            write_json(stage / "checkpoint.json", {"path": str(checkpoint), "sha256": sha256(checkpoint), "selection": checkpoint_rule})
        args = command(cfg, root / "experiment.json", stage, mode, split, checkpoint)
        run_process(args, cfg, stage)
        if mode == "train":
            final = stage / "model_final.pth"
            if not final.is_file():
                raise RuntimeError("Training exited without model_final.pth")
            write_json(stage / "checkpoint.json", {"path": str(final), "sha256": sha256(final), "selection": "final"})
            if cfg.get("checkpoint_selection") == "best_val_ap":
                best = stage / "model_best.pth"
                if not best.is_file() or not (stage / "best_checkpoint.json").is_file():
                    raise RuntimeError("Training finished without a best validation checkpoint")
                write_json(stage / "selected_checkpoint.json", {"path": str(best), "sha256": sha256(best), "selection": "best_val_ap", "validation": read_json(stage / "best_checkpoint.json")})
        else:
            native = read_json(stage / "inference" / "coco_instances_results.json")
            predictions = [{k: row[k] for k in ("image_id", "category_id", "bbox", "score")} for row in native]
            write_json(stage / "predictions.bbox.json", predictions)
            record = evaluate(cfg["dataset"][split]["annotations"], stage / "predictions.bbox.json")
            record.update({"run_id": run_id, "name": cfg["name"], "seed": cfg["seed"], "split": split, "supervision": manifest["supervision"], "purpose": manifest["purpose"], "checkpoint_sha256": sha256(checkpoint), "checkpoint_selection": checkpoint_rule})
            write_json(stage / "metrics.json", record)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(stage / "state.json", state)
    print(f"Run: {root}")
    return root


def main():
    import argparse
    from .experiment import load_experiment
    parser = argparse.ArgumentParser(description="Isolated F-v1 experiment entry point")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "kvasir.yaml"))
    parser.add_argument("--mode", required=True, choices=("train", "predict"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_experiment(args.config)
    if args.split not in cfg["dataset"]:
        raise ValueError(f"No {args.split} split in the confirmed baseline dataset")
    run(cfg, args.mode, run_id=args.run_id, split=args.split, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
