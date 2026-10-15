"""Compare completed unified bbox evaluations under the same training protocol."""
import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--baseline", default="maskdino-official-s42")
    parser.add_argument("--method", default="scda-proposal-v2-kvasir-s42")
    parser.add_argument("--output", default="runs/maskdino_vs_scda_v2")
    args = parser.parse_args()
    roots = [Path(args.runs) / name for name in (args.baseline, args.method)]
    configs, metrics = [], []
    rows = []
    for root in roots:
        for phase in ("train", "predict_val"):
            if read(root / phase / "state.json")["status"] != "completed":
                raise ValueError(f"Incomplete {root.name}/{phase}; no final comparison")
        cfg = read(root / "experiment.json")
        result = read(root / "predict_val" / "metrics.json")
        best = read(root / "train" / "best_checkpoint.json")
        if result["units"] != "fraction_0_to_1" or result["checkpoint_selection"] != "best_val_ap":
            raise ValueError("Need unified fraction metrics for best-val checkpoints")
        configs.append(cfg)
        metrics.append(result)
        budget = cfg["training_budget"]
        rows.append({
            "run_id": root.name, "seed": cfg["seed"],
            "AP50_95_percent": 100 * result["metrics"]["AP"],
            "AP50_percent": 100 * result["metrics"]["AP50"],
            "AP75_percent": 100 * result["metrics"]["AP75"],
            "best_iteration_zero_based": best["iteration"],
            "best_epoch": (best["iteration"] + 1) / budget["iterations_per_epoch"],
            "epochs": budget["epochs"], "iterations": cfg["overrides"]["SOLVER.MAX_ITER"],
            "supervision": cfg["dataset"]["supervision"],
            "auxiliary_mask_policy": cfg.get("scda", {}).get("mask_policy", "none"),
            "reg_selection": cfg.get("scda", {}).get("reg_selection", "none"),
            "annotations_sha256": result["annotations_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
    for key in ("seed", "training_budget", "dataset", "model", "overrides", "evaluation", "checkpoint_selection"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"Baseline/method protocol differs in {key}")
    for key in ("annotations_sha256", "protocol", "images"):
        if metrics[0][key] != metrics[1][key]:
            raise ValueError(f"Unified evaluation differs in {key}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# MaskDINO / SCDA F-v2：完整预算比较", "",
             "官方 train/val=880/120，报告 val bbox AP；同一 val 选择 best，无独立 test。数值单位：%。", "",
             "| Run | AP50:95 | AP50 | AP75 | Best epoch | Budget |",
             "|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row['run_id']} | {row['AP50_95_percent']:.4f} | {row['AP50_percent']:.4f} | {row['AP75_percent']:.4f} | {row['best_epoch']:.2f} | {row['epochs']} epoch / {row['iterations']} iter |")
    delta = rows[1]["AP50_95_percent"] - rows[0]["AP50_95_percent"]
    lines.extend(["", f"F-v2 − MaskDINO：AP50:95 {delta:+.4f} 个百分点。", "",
                  "两者原生训练均使用官方 bbox 和裁剪语义 mask。F-v2 额外 SCDA 只用原始单框图：分类核心至多2点，回归复用分类点加中覆盖率至多2点，各区间不互补。",
                  "同 ImageNet R50 初始化、batch4、AMP、512增强、优化器和完整100epoch；未从成熟基线或短跑续训。",
                  "这是单 seed42 的结果，不能据此声称统计显著性；多随机种子及机制消融仍未完成。", ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
