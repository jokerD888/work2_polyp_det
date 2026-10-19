"""Aggregate the fixed four-group experiment without inventing significance."""
import json
from pathlib import Path

import numpy as np

from polypbench.io import read_json, write_json


def run(output):
    out=Path(output); protocol=read_json(out/"protocol.json"); root=Path(protocol["workspace"])
    metrics={}; probes={}; replay={}
    lines=["# MSSL V1：四组机制归因报告","",
           "固定 Seed 42、50 epoch、相同源域划分与最佳源域验证 AP 权重规则。未搜索超参数，未按 PolypGen 调参。",
           "两组新增训练只改变额外定位损失；原生 L1/GIoU、Mask Loss、匹配和额外损失适用样本保持一致。","",
           "| 方案 | Kvasir AP | AP50 | AP75 | PolypGen AP | AP50 | AP75 |","|---|---:|---:|---:|---:|---:|---:|"]
    for group,run_id in protocol["runs"].items():
        train=root/"runs"/run_id
        source=read_json(train/"predict_val/metrics.json")["metrics"]
        external=(root/"runs/polypgen-zeroshot-20261018"/group if group in ("baseline","mssl") else out/"external"/group)
        target=read_json(external/"pooled/metrics.json")["metrics"]
        metrics[group]={"source":source,"polypgen":target,
            "checkpoint":read_json(train/"train/selected_checkpoint.json")}
        lines.append("| "+group+" | "+" | ".join(f"{100*m[k]:.3f}" for m in (source,target) for k in ("AP","AP50","AP75"))+" |")
        for label,path in (("training",train/"train/gradient_probes.jsonl"),("replay",out/f"{group}_replay/gradient_probes.jsonl")):
            if path.exists():
                rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                probes[f"{group}_{label}"]={"rows":len(rows),"eligible_rows":sum(r["eligible"]>0 for r in rows),"variants":{}}
                for variant in ("mask_cdf","box_cdf","mask_static"):
                    valid=[r["variants"][variant]["ratio_to_native_final_box"] for r in rows if r["eligible"]>0
                           and r["variants"][variant]["ratio_to_native_final_box"] is not None]
                    probes[f"{group}_{label}"]["variants"][variant]={"n":len(valid),
                        "ratio_median":float(np.median(valid)) if valid else None,
                        "ratio_min":float(min(valid)) if valid else None,"ratio_max":float(max(valid)) if valid else None}
        path=out/f"{group}_replay/replay.json"
        if path.exists():
            panel=read_json(path)["panel"]; sides=[s for image in panel for s in image["sides"]]
            distant=[s for s in sides if s["nontrivial_error_abs_d_gt_005"]]
            replay[group]={"sides":len(sides),"weak_slope_sides":sum(s["weak_slope_vs_local_1pct"] for s in sides),
                           "nontrivial_error_sides":len(distant),"weak_slope_nontrivial":sum(s["weak_slope_vs_local_1pct"] for s in distant)}
    comparisons={}
    for control in ("baseline","box_cdf","mask_static"):
        comparisons[control]={domain:{metric:100*(metrics["mssl"][domain][metric]-metrics[control][domain][metric])
            for metric in ("AP","AP50","AP75")} for domain in ("source","polypgen")}
    lines += ["","## 归因的读法","",
        "上表及 summary.json 保留精确差值；数值大小只代表本轮配对结果，不自动构成显著性或稳定性结论。",
        "MSSL 与 Box-CDF 的比较检验当前 CDF 形式下精细 Mask 支持的增量价值，不能单独证明更准确的形状理解或不可替代性。",
        "MSSL 与 Mask-static 的比较检验非线性变换相对自身局部二阶近似的经验价值；静态项无界、完整 CDF 有界，需结合大误差梯度和训练数值共同解释。",
        "即使 Mask-static 更好，也接受并围绕有效机制建立方法，不为保留 CDF 选择性报告。","",
        "## 预测与形态诊断","",
        "- [源域四组逐 IoU 阈值 AP](four_group_source_diagnostics/summary.md)",
        "- [源域覆盖、共同目标四边误差及形态关联](four_group_source_diagnostics/diagnostics.json)",
        "- [PolypGen 四组逐 IoU 阈值 AP](four_group_polypgen_diagnostics/summary.md)",
        "- [PolypGen 尺寸/中心分组及形态关联](four_group_polypgen_diagnostics/diagnostics.json)",
        "","候选覆盖采用最高总 IoU 一对一指派，所有固定 GT 保留，未覆盖记 0；同时保留 top-1/top-10/top-100 诊断。",
        "四边误差仅在双方共同覆盖 IoU≥0.5 的 GT 上比较，并同时报告覆盖数量；不能仅据此解释全部 AP 差值。",
        "几何指标仅用于原始单框图像；多框语义 Mask 不被假装为独立实例。相关系数、面积分层和控制面积/长宽比/基线 IoU 的回归均为探索性关联。",
        "AP 本身不能唯一分解为定位、排序和误检因果贡献；这些诊断用于检验解释的一致性。","",
        "## 梯度与曲线","",
        "[固定源域 GT 的损失与梯度曲线](loss_curves/curves.json)；同目录有 PNG/PDF。",
        "共享原生框头梯度为实际加权损失、AMP 缩放及裁剪前的 L2 范数，训练探针只取预先固定步骤的首个微批次，不等于全模型或累积梯度。",
        "探针反向固定乘 64 并在 FP32 中除回，降低未缩放 FP16 反向下溢；这不改变训练 GradScaler，仍有有限精度边界，零范数不能单独视为饱和。",
        "旧模型通过固定 16 张训练图像与相同 LSJ 随机种子重放；检查点局部行为不代表完整训练历史。",
        "饱和诊断使用 Gprime(d)<0.01*Gprime(0)，同时单列 |d|>0.05，避免把接近正确目标的低损失梯度称为饱和。","",
        "## 结论边界","",
        "源域 val 参与权重选择；PolypGen 已被研究者观察，其结果可作为无目标域训练的固定迁移复核，最终独立泛化验证应另留未参与开发的数据。",
        "当前按用户要求不做多 Seed；本报告不声称已验证跨训练随机性的稳定性，也不自动进入 V2 设计。"]
    write_json(out/"summary.json",{"metrics_fraction":metrics,"mssl_minus_control_pp":comparisons,
                                    "gradient_probe_summary":probes,"checkpoint_saturation_summary":replay})
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
