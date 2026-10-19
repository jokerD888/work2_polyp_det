"""Archive existing MSSL V1 evidence; never train, infer or load checkpoints.

Build once into a new directory, then verify with the Python standard library.
The archive contains exact historical source bundles and small evidence files.
Large checkpoints, predictions and data stay at their original hashed paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "repro/mssl_v1_20261020"
COMPETITION = "runs/ciou-competition-20261019"
CAMPAIGNS = {
    "initial": "runs/mssl-implementation-20261018/campaign-r2/campaign.json",
    "attribution": "runs/mssl-attribution-20261018-r2/protocol.json",
    "ciou": f"{COMPETITION}/protocol.json",
}
METRICS = ("AP", "AP50", "AP75", "AP90", "AP95")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def local(path):
    prefix = "/mnt/d/code/work2_polypy_det/"
    if path.startswith(prefix):
        return ROOT / path[len(prefix):]
    candidate = Path(path)
    if candidate.is_absolute():
        candidate.resolve().relative_to(ROOT)
        return candidate
    candidate = ROOT / candidate
    candidate.resolve().relative_to(ROOT)
    return candidate


def check(condition, message):
    if not condition:
        raise ValueError(message)


def build(output):
    check(not output.exists(), f"Refusing to overwrite archive: {output}")
    output.mkdir(parents=True)
    external = {}
    origins = {}

    def record(path, expected=None):
        path = local(str(path))
        key = relative(path)
        if key not in external:
            external[key] = {"sha256": digest(path), "bytes": path.stat().st_size}
        if expected is not None:
            check(external[key]["sha256"] == expected, f"Historical hash mismatch: {key}")
        return key

    def copy(path, dest):
        key = record(path)
        target = output / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / key, target)
        origins[dest] = key
        return dest

    bundles = {}
    for name, protocol_path in CAMPAIGNS.items():
        protocol = read(ROOT / protocol_path)
        source = local(protocol["source_root"])
        hashes = protocol["source_sha256"]
        archive = f"sources/{name}.zip"
        (output / "sources").mkdir(exist_ok=True)
        with zipfile.ZipFile(output / archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for rel, expected in sorted(hashes.items()):
                path = source / rel
                record(path, expected)
                info = zipfile.ZipInfo(rel, date_time=(2026, 10, 20, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes())
        copy(protocol_path, f"protocols/{name}.json")
        bundles[name] = {"archive": archive, "source_root": relative(source), "files": hashes}

    protocol = read(ROOT / COMPETITION / "protocol.json")
    comparison_path = f"{COMPETITION}/five_group_comparison/comparison.json"
    comparison = read(ROOT / comparison_path)
    targets = {
        "baseline": "runs/polypgen-zeroshot-20261018/baseline",
        "mssl": "runs/polypgen-zeroshot-20261018/mssl",
        "box_cdf": "runs/mssl-attribution-20261018-r2/external/box_cdf",
        "mask_static": "runs/mssl-attribution-20261018-r2/external/mask_static",
        "ciou_replace": f"{COMPETITION}/external",
    }
    models = {}
    for name, run in protocol["runs"].items():
        print(f"Checking {name}: checkpoint, source, configs and predictions", flush=True)
        run_root = ROOT / "runs" / run
        selection = read(run_root / "train/selected_checkpoint.json")
        weight = record(run_root / "train/model_best.pth", selection["sha256"])
        check(local(selection["path"]).resolve() == (ROOT / weight).resolve(), f"Checkpoint path: {name}")
        check(read(run_root / "train/state.json")["status"] == "completed", f"Incomplete training: {name}")
        bundle_name = "initial" if name in ("baseline", "mssl") else "ciou" if name == "ciou_replace" else "attribution"
        adapters = read(run_root / "adapter_sources.json")
        for rel, expected in adapters.items():
            record(run_root / "source" / rel, expected)
            check(bundles[bundle_name]["files"].get(rel) == expected, f"Run/campaign source mismatch: {name}/{rel}")
        target = ROOT / targets[name]
        source_metrics = read(run_root / "predict_val/metrics.json")
        target_metrics = read(target / "pooled/metrics.json")
        experiment = read(run_root / "experiment.json")
        run_manifest = read(run_root / "manifest.json")
        check(source_metrics["checkpoint_sha256"] == selection["sha256"], f"Prediction checkpoint: {name}")
        check(source_metrics["images"] == 120 and target_metrics["images"] == 1473, f"Image cohort: {name}")
        for domain, values in (("source_fraction", source_metrics), ("polypgen_fraction", target_metrics)):
            for metric in ("AP", "AP50", "AP75"):
                check(abs(values["metrics"][metric] - comparison[domain][name][metric]) < 1e-12, f"Result mismatch: {name}/{domain}/{metric}")
        source_predictions = record(run_root / "predict_val/predictions.bbox.json", source_metrics["predictions_sha256"])
        target_predictions = record(target / "pooled/predictions.bbox.json", target_metrics["predictions_sha256"])
        for split in ("train", "val"):
            record(experiment["dataset"][split]["annotations"], run_manifest["dataset"][split]["annotations_sha256"])
        check(source_metrics["annotations_sha256"] == run_manifest["dataset"]["val"]["annotations_sha256"], f"Source annotation identity: {name}")
        record(protocol["annotations"], target_metrics["annotations_sha256"])
        for rel in [
            "experiment.json", "manifest.json", "adapter_sources.json",
            "train/selected_checkpoint.json", "train/best_checkpoint.json", "train/config.yaml",
            "train/environment.json", "train/initialization.json", "train/method.json", "train/state.json",
            "predict_val/config.yaml", "predict_val/metrics.json", "predict_val/checkpoint.json",
        ]:
            copy(run_root / rel, f"models/{name}/{rel}")
        for rel in ["pooled/metrics.json", "config.yaml", "resolved_inference.yaml", "source_prediction_parity.json", "job.json"]:
            copy(target / rel, f"models/{name}/polypgen/{rel}")
        last = None
        with (run_root / "train/metrics.json").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if "optimizer/successful_updates" in row:
                        last = row
        check(last is not None, f"Missing AMP accounting: {name}")
        accounting = {k: v for k, v in last.items() if k == "iteration" or "amp" in k.lower() or "update" in k.lower()}
        record(run_root / "train/metrics.json")
        models[name] = {
            "run_id": run, "source_bundle": bundle_name,
            "checkpoint": weight, "checkpoint_sha256": selection["sha256"],
            "selected_iteration": selection["validation"]["iteration"],
            "selection_rule": selection["validation"]["rule"],
            "source_predictions": source_predictions, "polypgen_predictions": target_predictions,
            "source_fraction": comparison["source_fraction"][name],
            "polypgen_fraction": comparison["polypgen_fraction"][name],
            "training_accounting": accounting,
        }

    evidence = {
        "comparison.json": comparison_path,
        "uncertainty.json": f"{COMPETITION}/five_group_comparison/uncertainty/uncertainty.json",
        "old_four_group_parity.json": f"{COMPETITION}/five_group_comparison/old_four_group_parity.json",
        "dependencies.json": f"{COMPETITION}/dependencies.json",
        "ciou_initial_precheck.json": f"{COMPETITION}/precheck/report.json",
        "ciou_checkpoint_replay.json": f"{COMPETITION}/checkpoint_replay/report.json",
        "ciou_config_parity.json": f"{COMPETITION}/resolved_config_parity.json",
        "initialization_parity.json": f"{COMPETITION}/initialization_parity.json",
        "ciou_state.json": f"{COMPETITION}/state.json",
        "mssl_state.json": "runs/mssl-implementation-20261018/campaign-r2/state.json",
        "attribution_state.json": "runs/mssl-attribution-20261018-r2/state.json",
        "behavior_state.json": "runs/mssl-polypgen-analysis-20261019/state.json",
        "mssl_v1_contract.json": "runs/mssl-polypgen-analysis-20261019/mssl_v1_contract.json",
        "math_verification.json": "runs/mssl-polypgen-analysis-20261019/math_verification.json",
        "group_audit.json": "runs/mssl-polypgen-analysis-20261019/group_audit.json",
        "mssl_gpu_precheck.json": "runs/mssl-implementation-20261018/precheck-3/report.json",
        "behavior.json": "runs/mssl-polypgen-analysis-20261019/behavior/behavior.json",
    }
    for name, path in evidence.items():
        copy(path, f"evidence/{name}")
    for key in ("ciou", "mssl", "attribution", "behavior"):
        check(read(output / f"evidence/{key}_state.json")["status"] == "completed", f"Incomplete campaign: {key}")
    for key in ("ciou_initial_precheck", "ciou_checkpoint_replay", "mssl_gpu_precheck", "old_four_group_parity"):
        check(read(output / f"evidence/{key}.json")["status"] == "passed", f"Failed acceptance: {key}")
    for path, expected in protocol["reference_input_sha256"].items():
        record(path, expected)
    for name in ["bootstrap_ap_vectors.npz", "image_multiplicities.npy"]:
        record(f"{COMPETITION}/five_group_comparison/uncertainty/{name}")
    for path in ["data/kvasir_official_bbox_coco/train.json", "data/kvasir_official_bbox_coco/val.json"]:
        if (ROOT / path).exists():
            record(path)

    manifest = {
        "schema_version": 1, "release": "MSSL V1 / Seed42 / existing five-model evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Retrospective evidence freeze; no training, inference, tuning or new statistical claims.",
        "units": "AP fractions in JSON; multiply by 100 for AP points.",
        "source_bundles": bundles, "models": models,
        "copied_from": origins, "external_files": external,
        "limitations": ["single Seed42", "source validation selects checkpoints", "PolypGen already observed",
                        "image-level intervals; unknown patient correlation; no multiple-comparison correction",
                        "MSSL-minus-Box-CDF AP interval includes zero", "independent external evaluation pending",
                        "checkpoint/prediction/data bytes remain local; this is not an off-machine backup"],
    }
    dump(output / "manifest.json", manifest)
    write_summary(output, models)
    payload = {relative_to_output.as_posix(): {"sha256": digest(p), "bytes": p.stat().st_size}
               for p in sorted(output.rglob("*")) if p.is_file()
               for relative_to_output in [p.relative_to(output)]}
    dump(output / "checksums.json", payload)
    print(json.dumps({"archive": relative(output), "payload_files": len(payload), "external_files": len(external)}, indent=2))


def write_summary(output, models):
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "model", "run_id", "units", *METRICS])
        for domain in ("source_fraction", "polypgen_fraction"):
            for name, model in models.items():
                writer.writerow([domain.removesuffix("_fraction"), name, model["run_id"], "AP_points_0_to_100",
                                 *[100 * model[domain][k] for k in METRICS]])
    lines = ["# MSSL V1：已完成五组实验的证据归档", "",
             "归档日期：2026-10-20。只整理既有结果，没有训练、推理或修改方法。", "",
             "统一说明见 [发布索引](../../docs/mssl_v1_release_20261020.md)。", "",
             "## 内容", "",
             "- `manifest.json`：五组 run-id、权重路径/哈希、历史源码清单、训练更新数和外部文件指纹。",
             "- `sources/`：三个经过原协议逐文件 SHA256 核对的历史源码 ZIP；不以工作区新版代码代替旧训练版本。",
             "- `models/`：五组实际实验配置、解析配置、选权记录、环境与源域/PolypGen指标。",
             "- `protocols/`：三个原始执行协议的逐字节副本。",
             "- `evidence/`：方法契约、五组结果、配对区间、行为分析、数值/GPU验收与完成状态。",
             "- `results.csv`：两份数据上的五组 AP/AP50/AP75/AP90/AP95；单位为0–100 AP点。",
             "- `checksums.json`：归档内容校验清单；该文件自身由Git版本记录保护。", "",
             "## 校验", "", "在项目根目录执行（Python 3.10+，仅标准库）：", "", "```powershell",
             "python scripts/archive_mssl_v1.py verify",
             "python scripts/archive_mssl_v1.py verify --local", "```", "",
             "第一条校验已归档文件及ZIP内部源码；第二条还重新读取原位置的权重、预测和证据文件。", "",
             "权重、完整预测、Bootstrap抽样向量和数据未复制进Git，仍在原来的 `runs/`、`data/` 路径。",
             "保留这些本地目录；仅克隆Git仓库不包含模型权重，也不构成异机备份。", "",
             "## 源码与模型对应", "", "| 模型 | 源码包 | 选定迭代 | 成功更新数 |", "|---|---|---:|---:|"]
    for name, model in models.items():
        lines.append(f"| {name} | {model['source_bundle']}.zip | {model['selected_iteration']} | {model['training_accounting']['optimizer/successful_updates']:.0f} |")
    lines += ["", "所有组按Kvasir Val最大bbox AP50:95选权重，同分取最早；均为Seed42。",
              "源码包包含项目自有实现，官方MaskDINO和Detectron2依赖需按固定版本另行准备。",
              "真实环境、第三方版本和初始化哈希保存在 `models/*/train/`、`evidence/dependencies.json`。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify(output, include_local):
    checksums = read(output / "checksums.json")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - {"checksums.json"}
    check(actual == set(checksums), "Archive has missing or unlisted files")
    for rel, expected in checksums.items():
        path = output / rel
        check(path.resolve().is_relative_to(output.resolve()), f"Unsafe payload path: {rel}")
        check(digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"], f"Archive mismatch: {rel}")
    manifest = read(output / "manifest.json")
    total = 0
    for bundle in manifest["source_bundles"].values():
        with zipfile.ZipFile(output / bundle["archive"]) as source:
            check(set(source.namelist()) == set(bundle["files"]), "Source bundle file list mismatch")
            for name, expected in bundle["files"].items():
                check(hashlib.sha256(source.read(name)).hexdigest() == expected, f"Source mismatch: {name}")
                total += 1
    if include_local:
        for rel, expected in manifest["external_files"].items():
            path = local(rel)
            check(digest(path) == expected["sha256"] and path.stat().st_size == expected["bytes"], f"Local evidence mismatch: {rel}")
    print(json.dumps({"status": "passed", "payload_files": len(checksums), "source_entries": total,
                      "local_files_checked": len(manifest["external_files"]) if include_local else 0}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--local", action="store_true", help="Also hash original local checkpoints/predictions/evidence")
    args = parser.parse_args()
    if args.mode == "build":
        build(args.output.resolve())
    else:
        verify(args.output.resolve(), args.local)
