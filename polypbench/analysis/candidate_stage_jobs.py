"""Freeze portable references locally; prepare explicit remote GPU jobs offline."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from polypbench.io import read_json, sha256, write_json
from .candidate_stage import NATIVE_COMMIT


def frozen_image_hashes(root, sample):
    hashes = read_json(root / "runs/polypgen-local-relation-primary-20261021/protocol.json")["inputs_sha256"]
    result = {}
    for im in sample["images"]:
        matches = [digest for path, digest in hashes.items() if path.endswith("/" + im["file_name"])]
        if len(matches) != 1:
            raise RuntimeError(f"Missing or ambiguous historical image hash: {im['id']}")
        result[str(im["id"])] = matches[0]
    return result


def freeze(workspace, output):
    root = Path(workspace).resolve(); out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = root / "docs/protocols/candidate_stage_20261021"
    sample = read_json(protocol / "sample.images.json")
    ids = [im["id"] for im in sample["images"]]
    models = read_json(protocol / "models.json")
    manifest = {"native_commit": NATIVE_COMMIT, "models": {}, "files_sha256": {},
        "annotations_sha256": sha256(root / "data/polypgen_zeroshot_stills_v2/provided_boxes.coco.json"),
        "image_sha256": frozen_image_hashes(root, sample),
        "native_source_sha256": {}, "stage_status": "prepared_not_gpu_executed"}
    for relative in ("sample.images.json", "protocol.json"):
        shutil.copyfile(protocol / relative, out / relative)
    native = root / "third_party/MaskDINO"
    for path in sorted(native.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml") and ".git" not in path.parts:
            manifest["native_source_sha256"][path.relative_to(native).as_posix()] = sha256(path)
    for label, model in models.items():
        # Source paths were frozen under WSL. Relocate only the known workspace prefix.
        experiment = root / Path(model["experiment"].split("/work2_polypy_det/", 1)[1])
        if sha256(experiment) != model["experiment_sha256"]:
            raise RuntimeError(f"Frozen experiment drift: {label}")
        shutil.copyfile(experiment, out / f"{label}.experiment.json")
        replay = root / "runs/mssl-polypgen-analysis-20261019/score_replay" / label
        verification = read_json(replay / "verification.json")
        if verification["status"] != "passed" or verification["checkpoint_sha256"] != model["checkpoint_sha256"]:
            raise RuntimeError("Unverified historical reference")
        with np.load(replay / "score_components.npz") as arrays:
            positions = {int(i): n for n, i in enumerate(arrays["image_ids"])}
            order = [positions[i] for i in ids]
            np.savez_compressed(out / f"{label}.reference.npz", **{k: arrays[k][order] for k in arrays.files})
        manifest["models"][label] = {"checkpoint_sha256": model["checkpoint_sha256"],
            "experiment_sha256": model["experiment_sha256"],
            "original_run_id": experiment.parent.name,
            "reference_source_sha256": sha256(replay / "score_components.npz"),
            "original_verification": verification}
    for path in out.iterdir():
        if path.is_file():
            manifest["files_sha256"][path.name] = sha256(path)
    write_json(out / "manifest.json", manifest)
    print(f"Prepared portable CPU references for {len(ids)} images: {out}")


def prepare(bundle, workspace, assets_root, image_root, annotations, output):
    bundle, root = Path(bundle).resolve(), Path(workspace).resolve()
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = read_json(bundle / "manifest.json")
    for relative, digest in manifest["files_sha256"].items():
        if sha256(bundle / relative) != digest:
            raise RuntimeError(f"Bundle drift: {relative}")
    if sha256(annotations) != manifest["annotations_sha256"]:
        raise RuntimeError("PolypGen provided-box annotations differ from frozen source")
    for relative, digest in manifest["native_source_sha256"].items():
        if sha256(root / "third_party/MaskDINO" / relative) != digest:
            raise RuntimeError(f"Native source drift: {relative}")
    sample = read_json(bundle / "sample.images.json")["images"]
    missing = [str(Path(image_root) / im["file_name"]) for im in sample if not (Path(image_root) / im["file_name"]).is_file()]
    checkpoints = {label: Path(assets_root) / m["original_run_id"] / "train/model_best.pth"
                   for label, m in manifest["models"].items()}
    missing.extend(str(p) for p in checkpoints.values() if not p.is_file())
    if missing:
        raise FileNotFoundError("Missing assets; no GPU job prepared: " + "\n".join(missing))
    for im in sample:
        if sha256(Path(image_root) / im["file_name"]) != manifest["image_sha256"][str(im["id"])]:
            raise RuntimeError(f"Image drift: {im['id']}")
    for label, path in checkpoints.items():
        if sha256(path) != manifest["models"][label]["checkpoint_sha256"]:
            raise RuntimeError(f"Wrong checkpoint: {label}")
    out.mkdir(parents=True)
    for label in ("baseline", "mssl"):
        job = {"workspace": str(root), "runtime_location": "training_machine", "device": "cuda",
            "model_label": label, "checkpoint": str(checkpoints[label]),
            "experiment": str(bundle / f"{label}.experiment.json"),
            "reference": str(bundle / f"{label}.reference.npz"),
            "sample": str(bundle / "sample.images.json"), "annotations": str(Path(annotations).resolve()),
            "image_root": str(Path(image_root).resolve()), "output": str(out / label),
            "native_source_sha256": manifest["native_source_sha256"]}
        job["image_sha256"] = manifest["image_sha256"]
        job["observer_source_sha256"] = {relative: sha256(root / relative) for relative in (
            "polypbench/analysis/candidate_stage.py", "polypbench/analysis/matching.py",
            "methods/mssl/config.py", "methods/mssl/model.py", "methods/mssl/loss.py")}
        job["expected_sha256"] = {key: sha256(job[key]) for key in ("checkpoint", "experiment", "reference", "sample", "annotations")}
        write_json(out / f"{label}.job.json", job)
    write_json(out / "queue.json", {"status": "prepared", "tasks": [
        {"model": label, "command": [sys.executable, "-m", "polypbench.analysis.candidate_stage", "--job", str(out / f"{label}.job.json")]}
        for label in ("baseline", "mssl")], "training": False})
    print(f"All assets verified. Sequential GPU jobs prepared: {out}")


def run_queue(queue_root):
    """Finite sequential inference queue; no training and no automatic retries."""
    root = Path(queue_root).resolve()
    queue_path = root / "queue.json"
    queue = read_json(queue_path)
    if queue["status"] != "prepared":
        raise RuntimeError("Preserve prior results; prepare a new output directory instead of overwriting a queue")
    queue["status"] = "running"
    write_json(queue_path, queue)
    try:
        for task in queue["tasks"]:
            label = task["model"]
            if label not in ("baseline", "mssl"):
                raise ValueError(label)
            command = [sys.executable, "-m", "polypbench.analysis.candidate_stage", "--job", str(root / f"{label}.job.json")]
            task.update(status="running", started_at=time.time(), command=command)
            write_json(queue_path, queue)
            with (root / f"{label}.log").open("w", encoding="utf-8") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            task.update(status="completed", ended_at=time.time())
            write_json(queue_path, queue)
        from .candidate_stage_report import report
        image_root = read_json(root / "baseline.job.json")["image_root"]
        report(root, image_root, root / "report")
        queue["status"] = "completed"
        write_json(queue_path, queue)
    except BaseException as error:
        queue.update(status="failed", error=repr(error))
        if "task" in locals() and task.get("status") == "running":
            task["status"] = "failed"
        write_json(queue_path, queue)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--workspace", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("prepare")
    for key in ("bundle", "workspace", "assets-root", "image-root", "annotations", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("run"); p.add_argument("--queue-root", required=True)
    args = vars(parser.parse_args()); action = args.pop("action")
    {"freeze": freeze, "prepare": prepare, "run": run_queue}[action](**args)
