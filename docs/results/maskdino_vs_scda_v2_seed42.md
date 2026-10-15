# MaskDINO 与 SCDA F-v2：seed42 完整预算结果

> **2026-10-15 勘误：** 已确认这两组训练 mapper 的实例掩码补边存在 255 转前景和 padding 漏标问题。下表数字作为旧管线历史结果保留，不能视为修复版对照或将 F 的下降全部归因于 SCDA 设计。修复及实图 CPU 验证见 [说明](../instance_mask_padding_fix_20261015.md)。本轮未重训。

2026-10-15，Asia/Shanghai。两组均完成100 epoch / 22000 iteration，使用各自按 val AP50:95 选出的 best checkpoint 进行相同的统一 COCO bbox 评测。

| 方法 | AP50:95 (%) | AP50 (%) | AP75 (%) | Best iteration（零基） | Best epoch |
|---|---:|---:|---:|---:|---:|
| 原生 MaskDINO | 67.5271 | 82.8534 | 72.9928 | 18699 | 85 |
| MaskDINO + SCDA F-v2 | 65.7202 | 81.4498 | 71.4709 | 19799 | 90 |
| F-v2 − 原生（百分点） | -1.8070 | -1.4035 | -1.5219 | — | — |

**本轮 F-v2 没有提高检测 AP。** 这是固定数据和预算、seed42下的结果，不能据此声称统计显著性，也不能把负结果自动归因于代码故障。没有据此修改规则、调整超参数、延长预算或覆盖重训；DCNv4/Conv 实验不包含在本结果中。

## 数据、监督与协议

- Kvasir 官方 train/val=880/120，官方 bbox=941/130；类别为 polyp。bbox 保留官方 xmin/ymin/xmax-xmin/ymax-ymin，评测 area 为 bbox 面积。
- 两者原生训练均使用官方 bbox 与按 bbox 裁剪的语义前景 mask。该 mask 不是可靠的多实例标注。F-v2 额外辅助监督只接受原始单框图，多框图仍计算原生损失。
- F-v2 分类取 coverage>=.75 的至多2个核心点；回归复用实际分类点，加 .25<=coverage<.75 的至多2点；不足不跨区间补位。沿用统一 owner、原生 Top-Q 排除、精确像素积分、整单元 padding 排除、归一化中心距离及各 lambda=.05。
- 相同 ImageNet R50 初始化、batch4、AMP、512 LSJ（scale .5–1.5）、seed42、AdamW与学习率日程，100epoch/22000iter；第80/90epoch按原协议降学习率，每5epoch验证和保存。
- 使用同一 val 选择 best 并报告 val AP；没有独立 test。maxDets=[1,10,100]，pycocotools=2.0.11。未用成熟基线或短跑权重初始化 F-v2。

## 完成与核验

F-v2 训练开始于11:39:00，结束于18:40:12；最佳权重预测结束于18:41:16，完整链路18:41:17以退出码0结束。训练/推理进程均退出，完成检查时GPU计算进程为空。

两组训练日志最后一个训练iteration均为21999，记录的原生/辅助损失均有限；正式配置中数据、模型、训练预算、增强、优化器、评测与checkpoint选择一致。预测使用的checkpoint SHA与各自所选best相同；F-v2纳入快照的源码没有变化。验收记录在本地 `runs/_launcher/scda-proposal-v2-kvasir-s42/completion_verification.json`。

- F-v2冻结实现commit：`5188ba8`。
- 原生MaskDINO commit：`3831d8514a3728535ace8d4ecc7d28044c42dd14`。
- 原生best SHA256：`5ce92e2e9cbe7e1a8725f22ab40532aff821d66d237eca22969b190048f74362`。
- F-v2 best SHA256：`7fdb9e06e00f0877df6064d4e1c5b333c3f91954d6c5e5f8ebb3732668c9bba4`。
- 相同val标注SHA256：`db96be0dca4e0a4cebb308f096a9f67eaf5a6349a0fabfe01f5107dc43aed929`。

本地统一结果：`runs/maskdino_vs_scda_v2.csv`、`runs/maskdino_vs_scda_v2.md`及`runs/results.csv`。详细权重、预测和源码快照分别保留在 `runs/maskdino-official-s42/` 和 `runs/scda-proposal-v2-kvasir-s42/`；数据、权重、runs和环境不上传Git。

工程验收过程见 [F_PRECHECK.md](../../methods/scda_proposal/F_PRECHECK.md)，规则见 [F_V2_RULES.md](../../methods/scda_proposal/docs/F_V2_RULES.md)。本轮比较完成后停止原定自动跟进；后续实验按用户新指令安排。
