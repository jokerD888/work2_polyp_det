"""Tables, scientific figures and claim boundaries for the locked analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polypbench.io import read_json,write_json

LABELS=("baseline","mssl","box_cdf","mask_static")


def run(output):
    out=Path(output); u=read_json(out/"uncertainty/uncertainty.json"); b=read_json(out/"behavior/behavior.json")
    s=read_json(out/"score_effects/score_effects.json"); group=read_json(out/"group_audit.json")
    figures=out/"figures"; figures.mkdir(exist_ok=False)
    raw=np.load(out/"uncertainty/bootstrap_ap_vectors.npz")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    x=np.arange(.5,.951,.05)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    delta_summary={}
    for ax,control in zip(axes,("baseline","box_cdf","mask_static")):
        differences=100*(raw["mssl__pooled"]-raw[f"{control}__pooled"])
        lo,hi=np.quantile(differences,[.025,.975],axis=0)
        # Point estimates are independent of bootstrap sample means.
        from .coco_cache import CocoCache
        point=read_json(out/"point_ap_vectors.json")
        d=100*(np.array(point["mssl"])-np.array(point[control]))
        delta_summary[control]={"by_iou_pp":d.tolist(),"high4_contribution_to_AP_pp":float(d[6:].sum()/10),
                                "high4_share_of_total_AP_delta":float(d[6:].sum()/d.sum()) if d.sum() else None}
        ax.plot(x,d,"o-",lw=1.5); ax.fill_between(x,lo,hi,alpha=.18)
        ax.axhline(0,color="grey",lw=.8); ax.set_title(f"MSSL minus {control}"); ax.set_xlabel("IoU threshold"); ax.grid(alpha=.2)
    axes[0].set_ylabel("AP difference (percentage points)"); fig.tight_layout()
    fig.savefig(figures/"paired_ap_by_iou.png",dpi=180); fig.savefig(figures/"paired_ap_by_iou.pdf"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5)); names=["pooled",*[f"C{i}" for i in range(1,7)]]
    for i,c in enumerate(names):
        r=u["paired_intervals"]["baseline"][c]["AP"]
        ax.hlines(i,r["low_pp"],r["high_pp"],color="C0"); ax.plot(r["point_delta_pp"],i,"o",color="C0")
    ax.set_yticks(range(len(names)),names); ax.invert_yaxis(); ax.axvline(0,color="grey",lw=.8)
    ax.set_xlabel("MSSL - baseline AP (pp); paired image-bootstrap 95% interval")
    fig.tight_layout(); fig.savefig(figures/"center_ap_intervals.png",dpi=180); fig.savefig(figures/"center_ap_intervals.pdf"); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label in LABELS:
        coverage=b["cohorts"]["pooled"]["models"][label]["maximum_cardinality_coverage"]
        axes[0].plot([1,5,10,100],[100*coverage[f"t0.90_k{k}"] for k in (1,5,10,100)],"o-",label=label)
    axes[0].set_xscale("log"); axes[0].set_xticks([1,5,10,100],["1","5","10","100"])
    axes[0].set_xlabel("Top-K by native fused score (selected100 pool)"); axes[0].set_ylabel("Strict coverage at IoU 0.90 (%)"); axes[0].legend(fontsize=8)
    axes[1].bar(LABELS,[100*s["models"][l]["all300_maximum_cardinality_coverage"]["0.90"] for l in LABELS])
    axes[1].set_ylabel("All300 strict coverage at IoU 0.90 (%)"); axes[1].tick_params(axis="x",rotation=20)
    for ax in axes: ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); fig.savefig(figures/"candidate_quality_and_ranking.png",dpi=180); fig.savefig(figures/"candidate_quality_and_ranking.pdf"); plt.close(fig)
    lines=["# MSSL V1：PolypGen 配对检测行为与不确定性报告","",
        "本次仅分析已训练的四组 Seed42 模型，不修改损失、不训练、不搜索参数。所有点估计、分组规则、配对重采样和反事实排序均可复核。","",
        "## 1. 数据和独立性","",
        "固定 1473 张可计分图像、1533 个原始框、126 张负样本；其余缺框图像不参与评分。原始重复冲突图像的排除敏感性另报。",
        group["conclusion"],
        "Bootstrap 在六中心内部按图像有放回抽样，同一多重集合用于四个模型；中心规模固定。区间条件于现有中心且未消除未知患者内相关，不能称为患者级区间，也不反映换训练 Seed 的不确定性。","",
        "## 2. 总体 AP 与配对差值区间","",
        "| 对照 | MSSL−对照 AP | 95%区间 | AP75差值 | 95%区间 | AP90差值 | 95%区间 |",
        "|---|---:|---|---:|---|---:|---|"]
    for control in ("baseline","box_cdf","mask_static"):
        r=u["paired_intervals"][control]["pooled"]
        cells=[]
        for k in ("AP","AP75","AP90"):
            z=r[k]; cells.extend([f"{z['point_delta_pp']:+.3f}",f"[{z['low_pp']:+.3f}, {z['high_pp']:+.3f}]"])
        lines.append("| "+control+" | "+" | ".join(cells)+" |")
    lines += ["","区间包含零时不排除无增益；不因点估计为正或图像数量较多宣称显著性。多指标区间为探索性、未作多重比较校正。",
        "每个重采样副本使用独立 image/annotation ID 的语义；缓存仅复用图像内匹配，每次仍重建全局排序与101点PR。合成重复/分数并列测试及真实全量前两次重采样都与官方 COCOeval 对齐，见 cache_verification.json。","",
        "## 3. 高 IoU 分量与严格候选覆盖","",
        f"MSSL 相对 baseline 的总 AP 差值中，AP80–95 四个分量贡献 {delta_summary['baseline']['high4_contribution_to_AP_pp']:.4f} 点，占 {100*delta_summary['baseline']['high4_share_of_total_AP_delta']:.2f}%。这是 AP 分量算术占比，不是定位因果贡献比例。","",
        "| 模型 | 严格Top10覆盖@.90 | 严格Top100覆盖@.90 | 全300 Query覆盖@.90 | 负样本误报图像数（score≥.5） |",
        "|---|---:|---:|---:|---:|"]
    for l in LABELS:
        a=b["cohorts"]["pooled"]["models"][l]; cov=a["maximum_cardinality_coverage"]
        lines.append(f"| {l} | {100*cov['t0.90_k10']:.2f}% | {100*cov['t0.90_k100']:.2f}% | {100*s['models'][l]['all300_maximum_cardinality_coverage']['0.90']:.2f}% | {a['negative_frames_with_detection_score05']}/126 |")
    lines += ["","覆盖使用 IoU 阈值约束的最大基数二分图匹配，单个预测不能覆盖两个 GT。Top100 已经经过原生分类筛选；全300 Query 是筛选前框几何的诊断，两者不能混称。",
        "第一兼容候选的 GT 内排名另存 paired_individual_ranks；它不做排他匹配，且双方均有候选的条件统计必须结合新增/丢失覆盖数量。所有框边误差以四组共同覆盖的同一 GT 集合计算，避免比较集合不同。","",
        "## 4. 固定框集合下的分数反事实","",
        "| 模型 | 原生融合分数 AP | 同框分类分数 AP |",
        "|---|---:|---:|"]
    for l in LABELS:
        a=s["models"][l]; lines.append(f"| {l} | {100*a['native_fraction']['AP']:.3f} | {100*a['class_only_fraction']['AP']:.3f} |")
    lines += ["","原生最终分数为分类分数乘预测 Mask 质量。分类分数反事实保留原来100个候选和原始保存的框，只改用于分析的排序分数；不是新的部署方案，也不能与框几何效应作唯一、可加的因果分解。",
        "四模型全部可计分图像的只读重放均与既有原生预测核对，见 score_replay/*/verification.json。","",
        "## 5. 留一中心与协议敏感性","",
        "| 排除中心 | MSSL−Baseline AP | MSSL−Box-CDF AP | MSSL−Mask-static AP |","|---|---:|---:|---:|"]
    for c,models in u["leave_one_center_out_fraction"].items():
        values=[100*(models["mssl"]["AP"]-models[l]["AP"]) for l in ("baseline","box_cdf","mask_static")]
        lines.append("| "+c+" | "+" | ".join(f"{v:+.3f}" for v in values)+" |")
    sensitivity=read_json(out/"duplicate_sensitivity.json")
    lines += ["","排除四张完全重复但标签冲突图像后的指标见 duplicate_sensitivity.json；分中心点估计、区间和所有 IoU 阈值见 uncertainty.json 与 bootstrap_ap_vectors.npz。留一中心是观测六中心的敏感性分析，不是新中心的独立验证。","",
        "## 6. 解释边界和后续决定","",
        "候选覆盖增加支持检测器产出了更多满足给定 IoU 的框；同一候选池内更靠前的排名支持检索/打分表现改善。两者可同时存在，AP 无法从这些观察唯一归因。",
        "不能仅因高阈值 AP 增加断言所有框边更准确；不能将最终分数变化全部称为分类改善；不能用图像 Bootstrap 证明跨训练 Seed 稳定。",
        "本分析不选择新参数、不引入 V2 或质量头。方法章节中的经验性解释须引用本报告的具体指标、范围和反例。成熟定位损失对照与独立外测另行决策。","",
        "## 文件索引","",
        "- protocol.json：事先冻结的输入/代码哈希、阈值、K和抽样规则",
        "- group_audit.json：原始可用元数据及分组审计",
        "- behavior/behavior.json、per_image.csv、per_gt.csv：严格覆盖、排名、误检、固定共同GT框边误差",
        "- uncertainty/uncertainty.json、image_multiplicities.npy：配对区间、分中心、留一中心和抽样计数",
        "- score_replay/：原生推理一致性、分数分解、全部300个Query框",
        "- score_effects/：固定原始框的分类分数反事实",
        "- figures/：逐IoU配对区间、中心区间、候选覆盖 PNG/PDF"]
    write_json(out/"analysis_summary.json",{"high_iou_components":delta_summary,"paired_intervals":u["paired_intervals"],
        "strict_coverage":b["cohorts"]["pooled"],"score_effects":s,"duplicate_sensitivity":sensitivity})
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
