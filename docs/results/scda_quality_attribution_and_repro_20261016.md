# Q-BoxOnly 归因消融与 F-Reg-Quality seed 复现报告（A6000，2026-10-16）

第三轮两组实验：Q-BoxOnly（机制归因，seed42 配对）与 F-Reg-Quality seed43 复现。协议与前六组完全一致（B16、100ep=5500 iter、LR 4e-4、官方 880/120、ImageNet R50、每 275 iter 评估按 val AP50:95 选 best、统一 COCO bbox 评测）。全部为单 seed 点估计。

## 1. 结果总表（新增两行加粗）

| 组 | seed | reg 选点 | mask 门 | AP50:95 (%) | Δvs A(s42) | AP50 | AP75 | Best iter | 坍塌 | 衰减前均值 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| Clean A | 42 | — | — | 65.069 | — | 80.453 | 71.681 | 5500 | 无 | 55.55 |
| F-Reg | 42 | 覆盖率排序 | 有(≥.25) | 63.089 | −1.980 | 82.416 | 68.952 | 4674 | 4124→35.71% | 53.09 |
| F-Reg-Quality | 42 | IoU 排序 | 有(≥.25) | 66.143 | +1.073 | 84.583 | 72.779 | 5224 | 无 | 56.36 |
| **Q-BoxOnly** | 42 | **IoU 排序** | **无** | **64.751** | **−0.318** | 83.813 | 71.332 | 4949 | 无 | 54.21 |
| **F-Reg-Quality** | **43** | IoU 排序 | 有(≥.25) | **66.010** | +0.940* | 81.767 | 71.064 | 5500 | 无 | 54.11 |

*s43 无同 seed 原生基线，与 seed42 A 的比较为跨 seed 参考值（预先声明）。其余背景组（F-v2 62.524 / F-Cls 64.436 / F-Cls-Safe 61.624）见 `runs/_launcher/scda-safe-quality-a6000-s42/results/`。

## 2. 三个问题的回答

**（1）去掉 GT Mask 筛选后，性能提升、下降还是持平？——下降（实测，同 seed42 配对）。**

- F-Reg-Quality（IoU 排序 + mask 门）66.143 → Q-BoxOnly（同排序、去掉 mask 门）64.751，**−1.39**，且从高于 Clean A（+1.07）跌到低于 Clean A（−0.32）。
- 归因分解（同 seed 链条）：F-Reg 63.089 →（改为 IoU 排序）Q-BoxOnly 64.751，+1.66 →（加回 mask 门）66.143，+1.39。**IoU 质量排序解释了约 55% 的收益，GT Mask 引导贡献其余约 45%，两者互补、不可互相替代**；仅靠 IoU 排序不足以超过原生基线。
- 佐证：Q-BoxOnly 的 AP50（83.81）仍高于 A，但 AP75（71.33）回落到 A 的水平；mask 门的价值主要体现在高 IoU 阈值的定位质量上。
- 稳定性：Q-BoxOnly 无 ≥20 点坍塌，但 loss_ce 尖峰比 F-Reg-Quality 多且强（[2699,2719] 2.919、[3379] 2.657 vs 后者 [1679,1699] 2.136、[2699] 0.688），衰减前均值 54.21 低于 F-Reg-Quality 的 56.36——去掉 mask 门后稳定性优势也部分消失。

**（2）F-Reg-Quality 在新 seed 下是否仍然稳定？——是（实测）。**

- s43 best 66.010% @ iter 5500，与 s42 的 66.143 相差仅 **0.13**；两 seed 都稳定高于 seed42 原生基线约 +0.94 ~ +1.07（跨 seed 参考）。
- 训练健康度一致：无坍塌、无 loss_ce 失稳窗口（尖峰最弱 [2119,2139] 0.796）、衰减后持续上升、best 落在终点附近。AP 轨迹形状两 seed 有正常的随机差异（s43 中段波动更大），但终值与量级复现。
- 边界：只有 2 个 seed、无 seed43 原生配对基线，"≈+1 AP"的量级仍是参考性结论；统计显著性需要更多 seed。

**（3）对 SCDA 下一步设计意味着什么？**

- **保留完整定义**：SCDA 的当前有效形态 = "辅助回归 + 预测框-IoU 质量排序 + GT mask coverage 资格门"——两个组件各自贡献、叠加后才超过原生（实测归因如上）。任何后续消融都应把 F-Reg-Quality（quality_iou）作为新基线，而不是 F-v2 或 F-Reg。
- **GT mask 并非可去项**：mask 门提供了 IoU 排序之外的独立 +1.39 与稳定性收益；它的工作方式是"资格约束"（coverage ≥ 0.25 的池）而非排序，这一分工（mask 定资格、IoU 定次序）值得保留并作为后续设计的基本骨架。
- **辅助分类暂缓**：F-Cls / F-Cls-Safe 的证据显示分类辅助在当前形态下无正收益，优先把资源投在回归选点管线的细化（多 seed 显著性、τ_reg 敏感性、与 DCNv4/Conv 适配器的组合——均属后续授权范围）。
- 所有结论为单/双 seed 点估计，不构成统计结论；本轮未修改 SCDA 主线、未调参、未引入新算子。

## 3. 产物

- 两个 run：`runs/scda-qboxonly-pad0-a6000-b16-s42`、`runs/scda-freg-quality-pad0-a6000-b16-s43`（train/ + predict_val/ 完整）。
- 八组汇总 CSV：`runs/_launcher/qboxonly-repro-a6000/results/eight_group_results.csv`（含指纹、数据/checkpoint/预测哈希、稳定性字段）。
- 稳定性 JSON 与 λ 不变性/梯度审计：`runs/_launcher/qboxonly-repro-a6000/results/`；启动记录：同目录 `LAUNCH_RECORD.md`。
- 数据哈希不变：train `775b6574…`、val `264a0532…`；初始化 `16fd03f0…`；MaskDINO `3831d851`；代码 HEAD `8a3462c` + 门控方法改动（快照随 run 保存）。probe 目录已清理。
