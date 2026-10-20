"""Finite local continuation: wait for all E/F tasks, then run native diagnosis.

Does not edit, pause, stop or append into the running frozen E/F campaign.
This is a dependency-linked continuation with its own frozen source and state.
"""
import argparse
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from polypbench.io import read_json, write_json, sha256


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_identity(pid):
    """Linux start ticks avoid treating an unrelated reused PID as predecessor."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == "Z" else fields[19]


def dependency_decision(state, queue, expected_order, expected_pid, expected_start, actual_start):
    if state.get("pipeline_pid") != expected_pid:
        raise RuntimeError("Predecessor controller identity changed; preserve and review continuation")
    if state["status"] in ("failed", "paused", "stopped", "cancelled"):
        raise RuntimeError(f"Predecessor stopped: {state['status']}; diagnosis will not start")
    if actual_start is not None and actual_start != expected_start:
        raise RuntimeError("Predecessor PID was reused")
    if state["status"] == "completed":
        if queue["status"] == "running" and actual_start is not None:
            # The predecessor writes state, then queue, in its final save.
            return "waiting_completion_write"
        if (queue["status"] != "completed" or state["completed"] != expected_order
                or [t["name"] for t in queue["tasks"]] != expected_order
                or any(t["status"] != "completed" for t in queue["tasks"])):
            raise RuntimeError("Predecessor completion evidence is incomplete")
        return "waiting_predecessor_exit" if actual_start is not None else "ready"
    if state["status"] not in ("running", "starting") or actual_start is None:
        raise RuntimeError("Predecessor is not running and has no verified successful completion")
    return "waiting_predecessor"


def predecessor_status(config):
    root = Path(config["predecessor"])
    if sha256(root / "protocol.json") != config["predecessor_protocol_sha256"]:
        raise RuntimeError("Predecessor protocol changed")
    return dependency_decision(read_json(root / "state.json"), read_json(root / "queue.json"),
        config["predecessor_order"], config["predecessor_pid"], config["predecessor_start_ticks"],
        process_identity(config["predecessor_pid"]))


def gpu_available(gpu):
    output = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
    memory, utilization = [int(v.strip()) for v in output.split(",")]
    return memory <= 2048 and utilization <= 20, {"memory_used_mib": memory, "utilization_percent": utilization}


def authorize_child(config_path):
    config = read_json(config_path)
    state = read_json(Path(config_path).parent / "state.json")
    if state.get("status") != "running" or state.get("stage") != "native_diagnostics":
        raise RuntimeError("Local diagnosis has not been released by its continuation controller")
    if os.getppid() != state.get("controller_pid"):
        raise RuntimeError("Local GPU job must be a child of its authorized continuation controller")
    if predecessor_status(config) != "ready":
        raise RuntimeError("E/F predecessor has not fully exited after successful completion")


def launch(workspace, predecessor, output, image_root, annotations, gpu="0"):
    from .candidate_stage_jobs import prepare
    root, pred, out = map(lambda p: Path(p).resolve(), (workspace, predecessor, output))
    if out.exists():
        raise FileExistsError(f"A continuation already exists; do not duplicate it: {out}")
    protocol = read_json(pred / "protocol.json")
    pred_state = read_json(pred / "state.json")
    start = process_identity(pred_state["pipeline_pid"])
    if pred_state["status"] != "running" or start is None:
        raise RuntimeError("Expected the currently running E/F campaign")
    out.mkdir(parents=True)
    source = out / "source"; source.mkdir()
    for name in ("polypbench", "methods"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    bundle_rel = "docs/protocols/candidate_stage_20261021/bundle"
    shutil.copytree(root / bundle_rel, source / bundle_rel)
    (source / "third_party").mkdir()
    os.symlink(root / "third_party/MaskDINO", source / "third_party/MaskDINO", target_is_directory=True)
    config = {"created_at": now(), "workspace": str(root), "source": str(source), "output": str(out),
        "predecessor": str(pred), "predecessor_pid": pred_state["pipeline_pid"],
        "predecessor_start_ticks": start, "predecessor_order": protocol["order"],
        "predecessor_protocol_sha256": sha256(pred / "protocol.json"), "gpu": str(gpu),
        "gpu_idle_rule": "Predecessor successful and exited; memory<=2048MiB and utilization<=20% twice 30 seconds apart",
        "order": ["wait_for_all_ef_tasks_and_controller_exit", "wait_for_gpu_release", "baseline_stage_trace", "mssl_stage_trace", "comparison_report"],
        "training": False, "source_sha256": {p.relative_to(source).as_posix(): sha256(p)
            for p in source.rglob("*") if p.is_file() and "third_party" not in p.relative_to(source).parts}}
    prepare(source / bundle_rel, source, root / "runs", image_root, annotations, out / "diagnostics")
    for label in ("baseline", "mssl"):
        path = out / "diagnostics" / f"{label}.job.json"
        job = read_json(path)
        job.update(runtime_location="local_after_queue", local_controller_config=str(out / "config.json"))
        write_json(path, job)
    write_json(out / "config.json", config)
    write_json(out / "state.json", {"status": "waiting", "stage": "waiting_predecessor", "updated_at": now()})
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=str(gpu))
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    with (out / "controller.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "polypbench.analysis.candidate_stage_local",
            "run", "--output", str(out)], cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out / "launcher.json", {"pid": child.pid, "start_ticks": process_identity(child.pid),
        "detached": True, "output": str(out)})
    print(f"Queued local diagnosis after all E/F tasks; controller PID={child.pid}: {out}", flush=True)


def run(output):
    from .candidate_stage_jobs import run_queue
    out = Path(output); config = read_json(out / "config.json")
    state = {"status": "waiting", "controller_pid": os.getpid(), "stage": "waiting_predecessor"}
    def save():
        state["updated_at"] = now(); write_json(out / "state.json", state)
    try:
        for relative, digest in config["source_sha256"].items():
            if sha256(Path(config["source"]) / relative) != digest:
                raise RuntimeError(f"Frozen continuation source changed: {relative}")
        consecutive_idle = 0
        while True:
            if (out / "cancel.request").exists():
                raise RuntimeError("Continuation cancelled by request")
            status = predecessor_status(config)
            state["stage"] = status
            if status == "ready":
                available, usage = gpu_available(config["gpu"])
                state.update(stage="waiting_gpu_release", gpu=usage)
                consecutive_idle = consecutive_idle + 1 if available else 0
                if consecutive_idle >= 2:
                    break
            else:
                consecutive_idle = 0
            save(); time.sleep(30)
        state.update(status="running", stage="native_diagnostics"); save()
        run_queue(out / "diagnostics")
        state.update(status="completed", stage="report_complete")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("launch")
    for key in ("workspace", "predecessor", "output", "image-root", "annotations"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("run"); p.add_argument("--output", required=True)
    args = vars(parser.parse_args()); action = args.pop("action")
    {"launch": launch, "run": run}[action](**args)
