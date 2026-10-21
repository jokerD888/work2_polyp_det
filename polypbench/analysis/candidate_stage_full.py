"""Launch a finite, frozen local stage census of the full PolypGen benchmark."""
import argparse
from collections import defaultdict
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from polypbench.io import read_json, write_json, sha256
from .candidate_stage_local import gpu_available, process_identity


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def authorize_child(config_path):
    out = Path(config_path).parent
    config, state = read_json(config_path), read_json(out / "state.json")
    if config.get("scope") != "full_polypgen" or state.get("status") != "running" or state.get("stage") != "native_diagnostics":
        raise RuntimeError("Full-cohort GPU execution has not been released")
    if os.getppid() != state.get("controller_pid"):
        raise RuntimeError("Full-cohort GPU job must be a direct child of its controller")


def launch(workspace, output):
    from .candidate_stage_jobs import freeze, prepare
    root, out = Path(workspace).resolve(), Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    if read_json(root / "runs/mssl-strength-scope-20261020-r2/state.json")["status"] != "completed":
        raise RuntimeError("E/F campaign has not completed")
    if read_json(root / "runs/candidate-stage-local-after-ef-20261021/state.json")["status"] != "completed":
        raise RuntimeError("Prior fixed-panel diagnosis has not completed")
    if shutil.disk_usage(root).free < 15*1024**3:
        raise RuntimeError("Need at least 15 GiB free before preserving all full-cohort stage exports")
    out.mkdir(parents=True)
    source = out / "source"; source.mkdir()
    for name in ("polypbench", "methods"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (source / "third_party").mkdir()
    os.symlink(root / "third_party/MaskDINO", source / "third_party/MaskDINO", target_is_directory=True)
    freeze(root, out / "bundle", full_cohort=True)
    previous = read_json(root / "runs/mssl-polypgen-analysis-20261019/protocol.json")
    prepare(out / "bundle", source, root / "runs", previous["image_root"], previous["inputs"]["annotations"], out / "diagnostics")
    for label in ("baseline", "mssl"):
        path = out / "diagnostics" / f"{label}.job.json"
        job = read_json(path)
        job.update(runtime_location="local_full_cohort", local_controller_config=str(out / "config.json"))
        write_json(path, job)
    config = {"created_at": now(), "scope": "full_polypgen", "workspace": str(root), "source": str(source),
        "images": 1473, "gt": 1533, "gpu": "0", "training": False,
        "panel_reference": str(root / "runs/candidate-stage-local-after-ef-20261021/diagnostics"),
        "source_sha256": {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob("*")
            if p.is_file() and "third_party" not in p.relative_to(source).parts},
        "order": ["gpu_release", "baseline_full_stage_trace", "mssl_full_stage_trace", "panel_parity", "full_recovery_analysis", "full_report"]}
    write_json(out / "config.json", config)
    write_json(out / "state.json", {"status": "waiting", "stage": "gpu_release", "updated_at": now()})
    env = os.environ.copy(); env.update(PYTHONPATH=str(source), PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env["PATH"] = str(Path(sys.executable).parent)+os.pathsep+env["PATH"]
    with (out / "controller.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, "-m", "polypbench.analysis.candidate_stage_full", "run", "--output", str(out)],
            cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(out / "launcher.json", {"pid": child.pid, "start_ticks": process_identity(child.pid), "detached": True})
    print(f"Full PolypGen stage census queued: PID={child.pid}, {out}", flush=True)


def verify_panel(full_root, panel_root):
    checked = 0
    for model in ("baseline", "mssl"):
        for old_path in sorted((Path(panel_root)/model/"images").glob("*.npz")):
            new_path = Path(full_root)/model/"images"/old_path.name
            with np.load(old_path) as old, np.load(new_path) as new:
                for key in old.files:
                    if old[key].dtype.kind in "biu":
                        np.testing.assert_array_equal(old[key], new[key])
                    else:
                        np.testing.assert_allclose(old[key], new[key], rtol=1e-5, atol=1e-3 if "box" in key else 1e-6)
            checked += 1
    if checked != 144:
        raise RuntimeError("Historical 72-image panel is incomplete")
    return {"status": "passed", "model_images_checked": checked,
            "note": "Historical panel arrays agree within the same numerical tolerances; cohort annotations and native stages unchanged."}


def full_report(out):
    summary = read_json(out / "followup/summary.json")
    results = {"scope": "full_polypgen", "images": 1473, "gt": 1533, "models": summary["models"],
        "limits": summary["limits"], "panel_parity": read_json(out / "panel_parity.json")}
    lines = ["# PolypGen全量候选阶段诊断", "", "固定1473张图、1533个GT，六中心全量枚举，包含126张负图。仅Baseline与MSSL V1，无新增训练。", "",
        "各阶段统一原图坐标；逐GT是否存在候选与一对一最大基数覆盖分开。下面尺寸分层采用GT几何定义，不作为训练或模型输入。", ""]
    for model, value in results["models"].items():
        counts = read_json(out / "diagnostics" / model / "summary.json")
        if value["images"] != 1473 or value["gt"] != 1533:
            raise RuntimeError("Full census denominators differ")
        lines += [f"## {model}", "", "| 阶段 | ≥.5一对一覆盖 | ≥.75一对一覆盖 | ≥.9一对一覆盖 | ≥.5边际覆盖 | 最大IoU均值/中位数 |", "|---|---:|---:|---:|---:|---:|"]
        for stage in ("dense_native_legal", "encoder_top300", "decoder_init", "decoder_09", "final_top100"):
            metrics = counts["groups"]["all"]["stages"][stage]["thresholds"]
            numbers = [metrics[str(t)]["one_to_one_coverage"] for t in (.5,.75,.9)] + [metrics["0.5"]["marginal_gt_covered"]]
            continuous = counts["groups"]["all"]["stages"][stage]
            lines.append("| "+stage+" | "+" | ".join(map(str,numbers))+f" | {continuous['max_iou_mean']:.4f}/{continuous['max_iou_median']:.4f} |")
        lines += ["", "| 子集 | GT | 初始化丢失≥.5 | 初始化新增≥.5 | 丢失后最终300恢复 | 丢失后Top100恢复 |", "|---|---:|---:|---:|---:|---:|"]
        for group, data in value["groups"].items():
            m = data["thresholds"]["0.5"]
            lines.append(f"| {group} | {data['gt']} | {m['init_lost']} | {m['init_gained']} | {m['init_lost_recovered_by_final300']} | {m['init_lost_recovered_by_top100']} |")
        lines += ["", "各阈值的筛选丢失/后续恢复、最高IoU Encoder Query对应Mask的空/退化/范围变化，以及连续最大IoU均保存在JSON与逐GT文件中。", ""]
        negative = counts["groups"]["all"]
        lines += [f"负图：{negative['negative_images']}张，其中原生分数≥.5检测非空{negative['negative_images_with_score_ge_05']}张；这是固定分数的描述，不是临床或校准误报率。", ""]
    lines += ["## 解释边界", "", "这些频率覆盖完整固定PolypGen基准，但阶段观察不是因果分解，事件不能直接相加为AP收益上限。它们不证明跨seed或患者泛化，也不恢复已被开发过程观察过的盲测资格。", "",
        "高IoU收益与低阈值覆盖损失同时报告，不根据单一阈值宣布关闭Mask-to-Box或增加新模块。初始化干预、新损失和新训练均未执行。", "",
        "可视化实例：diagnostics/report/examples.jpg；完整阶段对照：diagnostics/report/report.md；恢复与尺寸分层：followup/。", ""]
    write_json(out / "full_comparison.json", results)
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(output):
    from .candidate_stage_jobs import run_queue
    from .candidate_stage_followup import run as recovery
    out = Path(output); config = read_json(out / "config.json")
    state = {"status": "waiting", "controller_pid": os.getpid(), "stage": "gpu_release"}
    def save():
        state["updated_at"] = now(); write_json(out / "state.json", state)
    try:
        for relative, digest in config["source_sha256"].items():
            if sha256(Path(config["source"])/relative) != digest:
                raise RuntimeError(f"Frozen source changed: {relative}")
        consecutive = 0
        while consecutive < 2:
            if (out / "cancel.request").exists():
                raise RuntimeError("Cancelled before GPU execution")
            available, usage = gpu_available(config["gpu"])
            state["gpu"] = usage
            consecutive = consecutive+1 if available else 0
            save()
            if consecutive < 2:
                time.sleep(30)
        state.update(status="running", stage="native_diagnostics"); save()
        run_queue(out / "diagnostics")
        state["stage"] = "panel_parity"; save()
        write_json(out / "panel_parity.json", verify_panel(out / "diagnostics", config["panel_reference"]))
        state["stage"] = "full_recovery_analysis"; save()
        recovery(out / "diagnostics", out / "followup")
        state["stage"] = "full_report"; save(); full_report(out)
        state.update(status="completed", stage="report_complete")
    except BaseException as error:
        state.update(status="failed", error=repr(error)); raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("launch"); p.add_argument("--workspace", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("run"); p.add_argument("--output", required=True)
    args = vars(parser.parse_args()); action = args.pop("action")
    {"launch": launch, "run": run}[action](**args)
