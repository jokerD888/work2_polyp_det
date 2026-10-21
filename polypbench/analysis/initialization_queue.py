"""Finite local queue: CPU acceptance -> both GPU smoke tests -> full inference."""
import argparse
from collections import defaultdict
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from polypbench.io import read_json, sha256, write_json
from .candidate_stage_jobs import full_cohort_sample
from .candidate_stage_local import gpu_available, process_identity


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def authorize_child(config_path):
    state = read_json(Path(config_path).parent / "state.json")
    if state.get("status") != "running" or os.getppid() != state.get("controller_pid"):
        raise RuntimeError("GPU intervention must be a direct child of its active finite controller")


def launch(workspace, output, cpu_receipt):
    root, out = Path(workspace).resolve(), Path(output).resolve()
    receipt = read_json(cpu_receipt)
    if receipt["status"] != "passed":
        raise RuntimeError("CPU acceptance not passed")
    for relative, digest in receipt["source_sha256"].items():
        if sha256(root / relative) != digest:
            raise RuntimeError("Tested source changed: " + relative)
    previous = root / "runs/polypgen-candidate-stages-full-20261021"
    assert read_json(previous / "state.json")["status"] == "completed"
    if shutil.disk_usage(root).free < 15*1024**3:
        raise RuntimeError("Need at least 15 GiB for preserved outputs")
    out.mkdir(parents=True, exist_ok=False)
    source = out / "source"; source.mkdir()
    for name in ("polypbench", "methods", "tests"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "third_party").mkdir()
    os.symlink(root / "third_party/MaskDINO", source / "third_party/MaskDINO", target_is_directory=True)
    shutil.copyfile(cpu_receipt, out / "cpu_checks.json")
    manifest = read_json(previous / "bundle/manifest.json")
    all_jobs = []
    for cohort in ("polypgen", "kvasir_val"):
        for label in ("baseline", "mssl"):
            base = read_json(previous / "diagnostics" / f"{label}.job.json")
            experiment = read_json(base["experiment"])
            if cohort == "polypgen":
                annotations, image_root = base["annotations"], base["image_root"]
                sample = read_json(base["sample"])
                cuts = base["relative_area_quartile_cuts"]
                historical_stages = str(previous / "diagnostics" / label)
                historical_stage_sha256 = {p.stem: sha256(p) for p in (Path(historical_stages)/"images").glob("*.npz")}
                assert len(historical_stage_sha256) == 1473
                historical_predictions = None
                hashes = base["image_sha256"]
            else:
                annotations = experiment["dataset"]["val"]["annotations"]
                image_root = experiment["dataset"]["val"]["image_root"]
                gt = read_json(annotations)
                sample, cuts = full_cohort_sample(gt)
                for im in sample["images"]:
                    im["center"] = "Kvasir"
                assert len(sample["images"]) == 120 and len(gt["annotations"]) == 130
                historical_stages = None
                historical_stage_sha256 = {}
                historical_predictions = str(Path(base["checkpoint"]).parents[1] / "predict_val/predictions.bbox.json")
                # Original validation cohort image fingerprint was frozen in this prior study.
                prior = read_json(root / "runs/mssl-quality-source-20261020/protocol.json")
                hashes = {str(im["id"]): sha256(Path(image_root)/im["file_name"]) for im in sample["images"]}
                for im in sample["images"]:
                    assert hashes[str(im["id"])] == prior["input_sha256"][str(Path(image_root)/im["file_name"])]
                assert sha256(annotations) == prior["input_sha256"][annotations]
                assert sha256(historical_predictions) == prior["input_sha256"][historical_predictions]
            gt = read_json(annotations)
            for im in sample["images"]:
                assert sha256(Path(image_root)/im["file_name"]) == hashes[str(im["id"])]
            for smoke in ((True, False) if cohort == "polypgen" else (False,)):
                name = ("smoke" if smoke else cohort) + "_" + label
                chosen = sample
                if smoke:
                    # One deterministic negative, single and multi, plus C5; no prediction-based sampling.
                    by_id = sorted(sample["images"], key=lambda x: x["id"])
                    selected = [next(im for im in by_id if im["gt_count"] == n) for n in (0,1,2)]
                    selected.append(next(im for im in by_id if im["center"] == "C5" and im["id"] not in {i["id"] for i in selected}))
                    chosen = {"images": selected}
                sample_path = out / f"{name}.sample.json"
                write_json(sample_path, chosen)
                inputs = {base["checkpoint"]: base["expected_sha256"]["checkpoint"],
                    base["experiment"]: base["expected_sha256"]["experiment"],
                    annotations: sha256(annotations), str(sample_path): sha256(sample_path)}
                if historical_predictions:
                    inputs[historical_predictions] = sha256(historical_predictions)
                job = {"name": name, "workspace": str(source), "cohort": cohort,
                    "checkpoint": base["checkpoint"], "experiment": base["experiment"],
                    "annotations": annotations, "image_root": image_root, "sample": str(sample_path),
                    "expected_images": len(chosen["images"]),
                    "expected_gt": sum(im["gt_count"] for im in chosen["images"]),
                    "smoke": smoke, "output": str(out / name), "controller_config": str(out / "config.json"),
                    "inputs_sha256": inputs, "image_sha256": hashes,
                    "relative_area_quartile_cuts": cuts, "native_source_sha256": manifest["native_source_sha256"],
                    "historical_stages": historical_stages, "historical_stage_sha256": historical_stage_sha256,
                    "historical_predictions": historical_predictions}
                for path, digest in inputs.items():
                    assert sha256(path) == digest
                job_path = out / f"{name}.job.json"; write_json(job_path, job)
                all_jobs.append({"name": name, "job": str(job_path), "job_sha256": sha256(job_path), "status": "prepared"})
    order = ["smoke_baseline", "smoke_mssl", "polypgen_baseline", "polypgen_mssl", "kvasir_val_baseline", "kvasir_val_mssl"]
    all_jobs.sort(key=lambda j: order.index(j["name"]))
    config = {"created_at": now(), "workspace": str(root), "source": str(source), "gpu": "0",
        "training": False, "conditions": ["native", "encoder", "blend05"],
        "tasks": all_jobs, "order": order,
        "source_sha256": {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*")
            if p.is_file() and "third_party" not in p.relative_to(source).parts},
        "fixed_rules": {"blend": "0.5 normalized cxcywh from Encoder sigmoid and native actual Mask reference sigmoid; native inverse_sigmoid afterwards",
            "abnormal_masks": "retain native mask2box sentinel conversion and clamp; no extra fallback; nonfinite actual references abort",
            "scoring": "recompute Decoder/classes/masks/top100/fused scores for each condition; no NMS or score filtering for AP",
            "selection": "Encoder Top300 IDs, content, encoder boxes and predicted initial mask logits bitwise identical on EVERY image",
            "decision": "joint AP/coverage net gains/FP and inference-observable mechanism; frozen-weight failure does not disprove retraining",
            "reserved": ["PolypsSet", "PolypDB"]}}
    write_json(out / "config.json", config)
    write_json(out / "state.json", {"status": "waiting", "stage": "gpu_release", "updated_at": now()})
    env = os.environ.copy(); env.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env["PATH"] = str(Path(sys.executable).parent)+os.pathsep+env["PATH"]
    with (out / "controller.log").open("w", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, "-m", "polypbench.analysis.initialization_queue", "run", "--output", str(out)],
            cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out / "launcher.json", {"pid": child.pid, "start_ticks": process_identity(child.pid), "detached": True})
    print(f"Initialization intervention queued: PID={child.pid}, {out}", flush=True)


def report(out):
    result = {}
    lines = ["# 冻结权重：三条件初始化干预", "", "PolypGen为主要开发基准；Kvasir Val为辅助复核。本轮无训练、无系数搜索。", "",
        "所有AP以百分数表示；覆盖得失是逐GT边际事件，与一对一覆盖分别报告。", ""]
    for cohort in ("polypgen", "kvasir_val"):
        result[cohort] = {}
        lines += [f"## {cohort}", "", "| 模型/条件 | AP | AP50 | AP75 | AP90 | AP95 |", "|---|---:|---:|---:|---:|---:|"]
        for label in ("baseline", "mssl"):
            directory = out / f"{cohort}_{label}"
            r = read_json(directory / "result.json")
            result[cohort][label] = r
            for c, metrics in r["evaluation"].items():
                lines.append(f"| {label}/{c} | "+" | ".join(f"{100*metrics[k]:.3f}" for k in ("AP","AP50","AP75","AP90","AP95"))+" |")
        lines += ["", "| 模型/干预，相对原生最终300框 | 阈值 | 恢复 | 新增丢失 | 净变化 |", "|---|---:|---:|---:|---:|"]
        for label in ("baseline", "mssl"):
            for c, change in result[cohort][label]["changes"].items():
                for t, events in change["all"]["stages"]["decoder_09"].items():
                    lines.append(f"| {label}/{c} | {t} | {events['recovered']} | {events['newly_lost']} | {events['net']} |")
        lines.append("")
    lines += ["## 解释边界", "", "固定权重初始化干预存在训练与推理分布变化，负结果不直接否定匹配初始化重训；局部恢复也不自动支持追加训练。GT仅用于前向后评估，未用于选择框或训练门控。", "",
        "原生框、分类分数、Mask质量、阶段轨迹及各分支上游一致性验收见各任务verification.json。分中心/尺寸/单多目标净得失、原183组恢复、负图误报、官方匹配下重复兼容FP见各任务result.json及per_gt_comparison.json。", "",
        "固定0.5融合属于工程对照，不能自动视为第二创新；PolypGen已参与开发，不作为盲测。"]
    write_json(out / "comparison.json", result)
    (out / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run(output):
    out = Path(output); config = read_json(out / "config.json")
    state = {"status": "waiting", "stage": "gpu_release", "controller_pid": os.getpid(), "completed": []}
    def save():
        state["updated_at"] = now(); write_json(out / "state.json", state)
        write_json(out / "queue.json", {"status": state["status"], "tasks": config["tasks"], "training": False})
    def verify_source():
        for relative, digest in config["source_sha256"].items():
            if sha256(Path(config["source"])/relative) != digest:
                raise RuntimeError("Frozen source changed: " + relative)
    try:
        verify_source()
        count = 0
        while count < 2:
            if (out / "cancel.request").exists():
                raise RuntimeError("Cancelled before task release")
            ready, usage = gpu_available(config["gpu"])
            state["gpu"] = usage; count = count+1 if ready else 0; save()
            if count < 2:
                time.sleep(30)
        state["status"] = "running"
        for task in config["tasks"]:
            if (out / "cancel.request").exists():
                raise RuntimeError("Cancelled before next task")
            verify_source()
            assert sha256(task["job"]) == task["job_sha256"]
            state["stage"] = task["name"]; task.update(status="running", started_at=now()); save()
            with (out / (task["name"]+".log")).open("w", encoding="utf-8") as stream:
                subprocess.run([sys.executable, "-m", "polypbench.analysis.initialization_intervention", "--job", task["job"]],
                    check=True, stdout=stream, stderr=subprocess.STDOUT)
            assert read_json(out / task["name"] / "verification.json")["status"] == "passed"
            task.update(status="completed", ended_at=now()); state["completed"].append(task["name"]); save()
        state["stage"] = "report"; save(); report(out)
        state.update(status="completed", stage="report_complete")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        for task in config["tasks"]:
            if task["status"] == "running":
                task.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("launch")
    for key in ("workspace", "output", "cpu-receipt"):
        p.add_argument("--"+key, required=True)
    p = sub.add_parser("run"); p.add_argument("--output", required=True)
    args = vars(parser.parse_args()); action = args.pop("action")
    {"launch": launch, "run": run}[action](**args)
