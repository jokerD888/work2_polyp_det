"""Fixed five-group COCO comparison; reuse the exact previous 2000 draws."""
import argparse
import json
from pathlib import Path
import numpy as np
from polypbench.io import read_json, write_json, sha256
from polypbench.analysis.coco_cache import CocoCache, metrics
from polypbench.analysis.uncertainty import run as bootstrap


def run(output):
    out=Path(output); protocol=read_json(out/"protocol.json"); workspace=Path(protocol["workspace"])
    old=workspace/"runs/mssl-polypgen-analysis-20261019"
    previous=read_json(old/"protocol.json")
    gt=read_json(previous["inputs"]["annotations"])
    predictions={k:read_json(v) for k,v in previous["inputs"]["predictions"].items()}
    predictions["ciou_replace"]=read_json(out/"external/pooled/predictions.bbox.json")
    destination=out/"five_group_comparison"; destination.mkdir(exist_ok=False)
    result=bootstrap(gt,predictions,destination/"uncertainty",draws_path=old/"uncertainty/image_multiplicities.npy")
    new_samples=np.load(destination/"uncertainty/bootstrap_ap_vectors.npz")
    old_samples=np.load(old/"uncertainty/bootstrap_ap_vectors.npz")
    maximum=0.
    for key in old_samples.files:
        np.testing.assert_allclose(old_samples[key],new_samples[key],rtol=0,atol=1e-12)
        maximum=max(maximum,float(np.max(abs(old_samples[key]-new_samples[key]))))
    assert sha256(destination/"uncertainty/image_multiplicities.npy")==sha256(old/"uncertainty/image_multiplicities.npy")
    write_json(destination/"old_four_group_parity.json",{"status":"passed","maximum_absolute_AP_vector_error":maximum,
        "draws_sha256":sha256(old/"uncertainty/image_multiplicities.npy"),"scope":"all original 4 models, 2000 draws, pooled and six centers"})
    source={}; target={}; vectors={}; training_rows=[]
    for label,run_id in protocol["runs"].items():
        trained=workspace/"runs"/run_id
        config=read_json(trained/"experiment.json")
        source_predictions=read_json(trained/"predict_val/predictions.bbox.json")
        cache=CocoCache(read_json(config["dataset"]["val"]["annotations"]),source_predictions)
        source[label]={**metrics(cache.reference),"AP50":float(cache.reference[0])}
        cache=CocoCache(gt,predictions[label]); vectors[label]=cache.reference.tolist()
        target[label]={**metrics(cache.reference),"AP50":float(cache.reference[0])}
        if label=="ciou_replace":
            training_rows=[json.loads(line) for line in (trained/"train/metrics.json").read_text().splitlines() if line.strip()]
    training_rows=[row for row in training_rows if "optimizer/successful_updates" in row]
    last=max(training_rows,key=lambda r:r["iteration"])
    summary={"source_fraction":source,"polypgen_fraction":target,"target_AP_by_IoU_fraction":vectors,
        "mssl_minus_ciou_paired_intervals":result["paired_intervals"]["ciou_replace"],
        "ciou_training":{"last_logged_iteration":last["iteration"],"successful_updates":last["optimizer/successful_updates"],
                         "last_AMP_scale":last["amp/loss_scale"]},
        "scope":"Practical competitor, not single-variable Mask/CDF ablation. One Seed42, same source selection; target already observed.",
        "uncertainty_limits":result["protocol"]["limitations"]}
    write_json(destination/"comparison.json",summary)
    lines=["# CIoU-Replace：五组竞争性比较", "", "固定Seed42及原训练协议。CIoU替换全部原生GIoU回归项；Matching保留GIoU。",
        "MSSL在原生损失上增加最终层Mask支持项，两者是实用竞争性比较，不是单变量机制消融。", "",
        "## 源域验证集", "", "该验证集参与Checkpoint选择，不能当作独立测试集。", ""]
    def table(values):
        rows=["| 方法 | AP | AP50 | AP75 | AP90 | AP95 |", "|---|---:|---:|---:|---:|---:|"]
        for label,v in values.items(): rows.append("| "+label+" | "+" | ".join(f"{v[k]*100:.3f}" for k in ("AP","AP50","AP75","AP90","AP95"))+" |")
        return rows
    lines+=table(source)+["","## PolypGen固定1473张队列",""]+table(target)
    lines +=["","## MSSL−CIoU配对差值（百分点）","","| 指标 | 点差 | 图像级95%区间 |","|---|---:|---|"]
    pair=result["paired_intervals"]["ciou_replace"]["pooled"]
    for key in ("AP","AP75","AP90","AP95"):
        value=pair[key]
        lines.append(f"| {key} | {value['point_delta_pp']:+.3f} | [{value['low_pp']:+.3f}, {value['high_pp']:+.3f}] |")
    ap=pair["AP"]
    conclusion=("区间包含零，不能根据单次总体AP点估计确认胜负。" if ap["contains_zero"] else
                "总体AP配对区间为正，支持本协议下MSSL相对该CIoU实现的正向差异。" if ap["low_pp"]>0 else
                "总体AP配对区间为负，本协议下CIoU具有竞争优势，应如实接受。")
    lines +=["",conclusion,"区间仅覆盖观测六中心内的图像重采样；患者相关未知、无多Seed，未作多指标校正，不作为普遍稳定性或Mask必然更优的证明。",
        "原始四组全部Bootstrap向量与历史结果核对；原2000次抽样计数逐字节复用，没有重新挑抽样种子。",
        "","## 训练与验收", "",f"最后日志迭代 {int(last['iteration'])}，实际成功更新 {int(last['optimizer/successful_updates'])}；预定2750迭代，AMP跳步单独记录。",
        "初始全分支及最终检查点的加权CIoU/GIoU值、预测坐标/共享框头梯度见 ../precheck/report.json 与 ../checkpoint_replay/report.json。诊断不用于调权重。",
        "","## 文件索引","","- comparison.json：五组数值及训练记录",
        "- uncertainty/uncertainty.json：全部五组区间、MSSL相对各对照的区间及留一中心",
        "- uncertainty/bootstrap_ap_vectors.npz：全部原始配对向量",
        "- old_four_group_parity.json：历史四组与原抽样一致性",
        "","PolypsSet、PolypDB未运行，V2未开发。后续方法取舍须结合全部指标和研究成本，不因某一指标较好自动冻结，也不因未胜出自动改方法。"]
    (destination/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",required=True); run(p.parse_args().output)
