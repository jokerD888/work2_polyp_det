# SCDA 单任务消融与第二轮改进实验报告（A6000，seed42，2026-10-16）

覆盖六个 run：Clean A / Clean F-v2 / F-Reg / F-Cls（第一轮消融）与 F-Cls-Safe / F-Reg-Quality（第二轮单变量改进）。协议全部一致：batch16 单卡、LR 4e-4（backbone×0.1）、100ep=5500 iter、warmup 125、steps [4400,4950]、AMP、clip 0.01、官方 880/120 与官方框、补边=0、ImageNet R50 初始化、每 275 iter 评估并按 val AP50:95 选 best、统一 COCO bbox 评测（pycocotools 2.0.11，maxDets=[1,10,100]）。全部为单 seed 点估计，不构成统计结论。

## 1. 统一对照表

| 组 | λ_cls | λ_box | reg_selection | cls 形式 | **AP50:95 (%)** | ΔvsA | AP50 (%) | AP75 (%) | Best iter/ep | 训练 s | s/iter | 峰值显存 MB | ≥20 点坍塌 | 衰减前均值 AP |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Clean A | — | — | — | — | **65.069** | — | 80.453 | 71.681 | 5500/100 | 10159 | 1.698 | 28258 | 无 | 55.55 |
| F-v2 | .05 | .05 | core_plus_transition | BCE | 62.524 | −2.545 | 81.665 | 68.976 | 4674/85 | 11093 | 1.830 | 28305 | 2199→33.13% | 52.38 |
| F-Reg | 0 | .05 | core_plus_transition | BCE | 63.089 | −1.980 | 82.416 | 68.952 | 4674/85 | 11021 | 1.835 | 28302 | 4124→35.71% | 53.09 |
| F-Cls | .05 | 0 | core_plus_transition | BCE | 64.436 | −0.633 | 80.959 | 69.651 | 5500/100 | 11057 | 1.844 | 28302 | 无 | 52.53 |
| F-Cls-Safe | .05 | 0 | core_plus_transition | **focal 正项** | 61.624 | −3.445 | 77.765 | 69.747 | 4674/85 | 11033 | 1.840 | 28304 | 无 | 52.88 |
| **F-Reg-Quality** | 0 | .05 | **quality_iou** | BCE | **66.143** | **+1.073** | **84.583** | **72.779** | 5224/95 | 11193 | 1.848 | 28301 | 无 | **56.36** |

run-id：`maskdino-official` / `scda-v2` / `scda-freg` / `scda-fcls` / `scda-fcls-safe` / `scda-freg-quality`，均带 `-pad0-a6000-b16-s42` 后缀。模型参数六组相同（52,017,477）。

## 2. 主要结论（区分实测与推断）

1. **辅助回归的选点质量是决定性变量（实测）**。F-Reg-Quality 对 F-Reg 只改一处——reg 候选排序从覆盖率改为预测框-GT IoU——AP50:95 从 63.089 升至 66.143（+3.05），AP50/AP75 双双超过原生（84.58/72.78 vs 80.45/71.68），且成为唯一在衰减前均值上超过 A（56.36 vs 55.55）的 SCDA 变体。此前"SCDA 必然损害基线"的模式被打破。
2. **辅助分类的强 BCE 信号本身有价值，"focal 对齐"是错误的修复（实测）**。F-Cls-Safe 相对 F-Cls 唯一改动是把辅助正项 BCE 换成原生一致的 focal 正项，AP 反而下降 2.81（64.436→61.624），AP50 跌到全组最低（77.77）。focal 把辅助值压到约 1/40（探针实测 0.00044 vs 0.0178），等效于大幅削弱辅助信号。结合第一阶段"失稳并非 cls 辅助独有"的证据，推断：辅助分类的主要作用是强排序信号而非冲突源；其与原生监督的摩擦不是当前主要矛盾。
3. **稳定性（实测）**：F-Reg-Quality 无坍塌、loss_ce 尖峰最少最弱（[1679,1699] 峰值 2.136 与 [2699] 0.688）；loss_ce_interm 后三段均值（0.436/0.333/0.159）低于 A（0.554/0.372/0.234）——辅助回归改用质量选点后，分类路径的扰动实测消失甚至反向受益。机制推断：按 IoU 选点使辅助梯度落在 box 头已对齐的 token 上（辅助残差 0.134 vs F-Reg 0.274），避免了对特征的强矫正对抗（与第一阶段测到的 box 辅助梯度在 encoder/backbone 反向 cos −0.24/−0.39 相对照）。
4. **两个坍塌事件的原始问题在质量选点下未再出现**（F-v2 @2199、F-Reg @4124；F-Reg-Quality 同窗口 AP 为 56.61 / 59.08，无异常）。
5. **边界**：+1.07 优势来自单 seed 单划分；AP 的 val 即选集，无独立 test；F-Reg-Quality 的训练时长与显存与其它 SCDA 变体持平（辅助开销 ~8% 相对原生）。

## 3. 对既定问题的最终回答

1. **F-v2 负收益归因**：联合监督中辅助回归的 coverage 选点是主要损害来源（实测：同 λ 下换成质量选点即反超 A）；辅助分类单独代价小（−0.63）且有正贡献。
2. **移除辅助分类能否缓解失稳**：不能（F-Reg 仍坍塌）；失稳由辅助监督对共享路径的强对抗引起，与哪个分支关系不大，而与选点质量强相关。
3. **哪种单任务辅助有独立正收益**：辅助回归 + 质量选点（F-Reg-Quality +1.07 vs A）；辅助分类单独为负（−0.63）。
4. **后续保留什么**：保留并围绕 **quality_iou 回归选点**推进（多 seed 验证、与 DCNv4/Conv 适配器的组合属后续授权范围）；辅助分类维持 BCE 形式，是否保留/改造留待以 F-Reg-Quality 为新基线的下一轮消融。本轮按约定不修改方法主线之外的东西，不引入 DCNv4/Conv。

## 4. 产物与可复现性

- 六组 run 目录（train/ + predict_val/ 完整：metrics.json 逐 iter 损失与逐次评估、console.log、experiment.json、source 快照、checkpoint、selected_checkpoint.json、predictions、metrics）。
- 汇总 CSV：`runs/_launcher/scda-safe-quality-a6000-s42/results/scda_round2_six_group_results.csv`（含 config_fingerprint、数据/checkpoint/预测哈希、稳定性字段）。
- 第一轮诊断：`runs/_launcher/scda-single-task-ablation-a6000-s42/results/diagnosis_phase1.md`；两轮启动记录：各自 launcher 的 `LAUNCH_RECORD.md`。
- 数据哈希（六组一致）：train `775b6574…`，val `264a0532…`；初始化 R-50.pkl `16fd03f0…`；MaskDINO commit `3831d851`；项目 HEAD `8a3462c` + 第二轮未提交方法改动（config.py / scda_assigner.py / scda_loss.py / test_cpu.py，均门控默认关，快照随 run 保存）。
- 冒烟证据：SCDA CPU 27/27、仓库 22/22、补边审计、两轮 λ 不变性+梯度审计 JSON（零 λ 分支梯度严格 0；原生路径偏差 ≤8.5e-8，仅 DN dice 的 ulp 抖动）。
- probe run 目录（6 个）已按惯例在验收记录落盘后清理。
