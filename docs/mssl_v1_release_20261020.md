# MSSL V1：方法冻结与五组证据索引

冻结日期：2026-10-20。CIoU链路于北京时间2026-10-20 01:14完成，全部13阶段通过。本次将实际训练源码、配置、选定权重身份和既有结果统一归档；没有新增训练、推理、调参或统计分析。

**当前决定：保留MSSL V1，固定本轮方法与五组模型，进入后续独立外测准备。** 冻结是固定后续评价对象，不表示独立泛化、跨Seed稳定性或新颖性已全部验证。PolypsSet、PolypDB尚未评估，PICCOLO申请暂停，多Seed、V2及新增模块仍未安排。

## 1. 唯一当前入口

- [归档说明与校验方法](../repro/mssl_v1_20261020/README.md)
- [机器可读总清单：源码、模型身份、路径及SHA256](../repro/mssl_v1_20261020/manifest.json)
- [两份数据×五组完整指标CSV](../repro/mssl_v1_20261020/results.csv)
- [五组完整数值JSON](../repro/mssl_v1_20261020/evidence/comparison.json)、[全部配对区间与分中心结果](../repro/mssl_v1_20261020/evidence/uncertainty.json)
- [方法正文、数学性质与适用条件](mssl_v1_method_20261019.md)、[实际训练方法契约](../repro/mssl_v1_20261020/evidence/mssl_v1_contract.json)

本目录及 `repro/mssl_v1_20261020/` 保存当前结论与可核查证据。旧计划和启动记录保留其形成时间及历史决策，不能按其中的启动状态判断现状。

## 2. 冻结的方法与实际实现

GT Box定义定位目标；GT Mask轴向投影与靠近对应框边的权重提供方向性前景支持。额外项为四边CDF差的半平方，在有效匹配GT上平均。权重1.0，温度0.05，方向带宽0.15，轴向尺度下限4像素；这些是本次固定研究设置，不是已证最优参数。

仅最终Decoder的可靠匹配框接受额外监督；原始多框图、空Mask及退化框不接受该额外项。复用原生Matching，保留L1/GIoU、Mask、DN和其他层原生损失。没有新增可学习参数，推理保留原生MaskDINO及预测Mask参与的原生打分；推理不读取GT Mask。

| 模型 | 实际训练run-id | 原始训练源码包 | 选定迭代 |
|---|---|---|---:|
| Baseline | `maskdino-mssl-control-s42-20261018T025946` | [initial.zip](../repro/mssl_v1_20261020/sources/initial.zip) | 2750 |
| MSSL V1 | `mssl-official50-s42-20261018T025946` | [initial.zip](../repro/mssl_v1_20261020/sources/initial.zip) | 2750 |
| Box-CDF | `mssl-box-cdf-s42-20261018T114010Z` | [attribution.zip](../repro/mssl_v1_20261020/sources/attribution.zip) | 2750 |
| Mask-static | `mssl-mask-static-s42-20261018T114010Z` | [attribution.zip](../repro/mssl_v1_20261020/sources/attribution.zip) | 2750 |
| CIoU-Replace | `ciou-replace-s42-20261019T073936Z` | [ciou.zip](../repro/mssl_v1_20261020/sources/ciou.zip) | 2750 |

三个源码包分别来自当时执行协议的 `source_sha256` 清单。每个run自存的adapter源码也与对应包核对；当前工作区包含后续分析与对照功能，不应整体当作首轮MSSL实际训练版本。冻结ZIP保留历史README等原始字节，其中的状态描述属于当时记录。

共同协议为Seed42、Kvasir 880/120、50等效Epoch、2750预定迭代，microbatch1×累积16，按源域Val最大bbox AP50:95选权重、同分取最早。五组实际均2740次成功更新，AMP跳步保留原日志。验证集参与选权，不是独立测试集；该累积方案不宣称等价于物理batch16。

各组的 `experiment.json`、实际训练/推理 `config.yaml`、环境、初始化、选权记录与指标逐字节保存在 [models目录](../repro/mssl_v1_20261020/models/)。官方MaskDINO commit为 `3831d8514a3728535ace8d4ecc7d28044c42dd14`；Detectron2 commit为 `1e3e13bbf607b54f62205c4c33922521822fb298`，依赖详情见 [环境审计](../repro/mssl_v1_20261020/evidence/dependencies.json)。第三方源码需另按固定版本准备。

## 3. 五组统一结果

单位为0–100 AP点；JSON指标为0–1。Kvasir为参与选权的源域验证集。

| 方法 | Kvasir AP | AP50 | AP75 | AP90 | AP95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 66.072 | 86.129 | 73.184 | 34.579 | 12.898 |
| MSSL V1 | 66.273 | 86.531 | 75.860 | 37.121 | 8.091 |
| Box-CDF | 65.729 | 83.420 | 73.497 | 42.638 | 13.262 |
| Mask-static | 65.978 | 85.487 | 74.667 | 38.220 | 6.325 |
| CIoU-Replace | 65.582 | 82.145 | 72.751 | 44.852 | 11.607 |

PolypGen为已观察过的固定1473张外部队列，含1533个提供框与126张负样本，没有目标域训练；不能称为未参与研究决策的全新确认集。

| 方法 | PolypGen AP | AP50 | AP75 | AP90 | AP95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 48.818 | 67.281 | 53.006 | 26.994 | 7.057 |
| MSSL V1 | 49.751 | 67.525 | 53.566 | 29.805 | 9.712 |
| Box-CDF | 49.112 | 67.081 | 53.751 | 27.975 | 7.909 |
| Mask-static | 48.710 | 66.969 | 52.521 | 27.322 | 8.903 |
| CIoU-Replace | 48.615 | 66.133 | 53.044 | 27.449 | 8.421 |

| PolypGen配对比较 | AP差值 | 图像级95%区间 | AP90差值 | 图像级95%区间 |
|---|---:|---|---:|---|
| MSSL−Baseline | +0.933 | [+0.045,+1.777] | +2.811 | [+1.062,+4.542] |
| MSSL−Box-CDF | +0.639 | [-0.239,+1.512] | +1.831 | [+0.403,+3.460] |
| MSSL−Mask-static | +1.041 | [+0.130,+1.963] | +2.483 | [+0.808,+4.106] |
| MSSL−CIoU | +1.136 | [+0.209,+2.033] | +2.357 | [+0.815,+3.804] |

复用原2000次六中心内图像级配对抽样，原四组全部向量与历史最大误差为零，见 [一致性记录](../repro/mssl_v1_20261020/evidence/old_four_group_parity.json)。区间条件于观测中心、未处理未知患者内相关、未覆盖训练Seed且未作多指标校正。AP90差异作为探索性证据，不据此宣称患者级显著性或普遍稳定。

## 4. 可以与不能作出的结论

当前同协议实证支持保留MSSL V1：其PolypGen总体AP优于本次Baseline、Mask-static及CIoU-Replace实现；后者属于竞争性比较，不是单变量Mask/CDF消融。精细Mask相对Box-CDF的总体AP区间仍包含零，不能写成稳健优于矩形支持。

源域MSSL AP95低于Baseline，CIoU源域AP90高于MSSL；PolypGen MSSL−Baseline和MSSL−CIoU的AP75区间均包含零。没有全指标优势，也不把高IoU收益解释为所有框都更准确。

固定框集合使用分类分数时MSSL仍有AP增量，说明收益不能全部归于最终Mask重打分；候选供给、选择和排序的因果贡献尚不能唯一分解。已有行为证据见 [归档数据](../repro/mssl_v1_20261020/evidence/behavior.json) 与 [完整分析说明](mssl_polypgen_analysis_20261019.md)。近邻机制差异、新颖性、独立外测与跨Seed稳定性仍应分别审查。

## 5. 本地保存与验收

五组 `model_best.pth`、原始预测、训练日志、Bootstrap向量和数据仍在原位置，路径与SHA256见总清单。权重合计约3.13GB，不重复复制进Git。源码包、配置和小型证据随本次Git版本保存；本归档不代表已完成异机备份，不能删除原 `runs/` 与 `data/`。

```powershell
python scripts/archive_mssl_v1.py verify
python scripts/archive_mssl_v1.py verify --local
```

前者校验归档文件和ZIP内部源码；后者额外核对本地权重、预测与原证据。校验只读取文件，不加载模型、不启动实验。

历史详细报告：[四组归因](../runs/mssl-attribution-20261018-r2/summary.md)、[检测行为与统计](../runs/mssl-polypgen-analysis-20261019/summary.md)、[CIoU五组比较](../runs/ciou-competition-20261019/summary.md)。这些 `runs/` 链接仅在保留本地实验目录时可用；归档数值和配置在Git中另有副本。
