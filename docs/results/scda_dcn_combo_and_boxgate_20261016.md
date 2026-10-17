# F-Reg-Quality + DCNv4 与 Box-Gated IoU 结果报告（A6000 ×2，2026-10-16）

第四轮两组实验：**F-Reg-Quality + DCNv4 组合**（GPU0）与 **Box-Gated IoU 校准消融**（GPU1）。
协议与前三轮完全一致（B16、100ep=5500 iter、LR 4e-4、官方 880/120、ImageNet R50、每 275 iter 评估按
val AP50:95 选 best、统一 COCO bbox 评测）。全部为 seed42 单 seed 点估计。
执行入口、验收证据与启动记录见 `runs/_launcher/quality-boxgate-a6000-s42/`（`acceptance/ACCEPTANCE_RECORD.md`）。

## 1. 结果总表（新增两行加粗）

| 组 | seed | reg 选点 | mask 门 | AP50:95 (%) | Δvs A(s42) | AP50 | AP75 | Best iter | 坍塌 | 衰减前均值 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| Clean A | 42 | — | — | 65.069 | — | 80.453 | 71.681 | 5500 | 无 | 55.55 |
| F-Reg-Quality | 42 | IoU 排序 | 有(≥.25) | 66.143 | +1.074 | 84.583 | 72.779 | 5224 | 无 | 56.36 |
| Q-BoxOnly | 42 | IoU 排序 | 无 | 64.751 | −0.318 | 83.813 | 71.332 | 4949 | 无 | 54.21 |
| **F-Reg-Quality + DCNv4** | 42 | IoU 排序 | 有(≥.25) | **64.738** | **−0.331** | 80.047 | 71.890 | 5500 | 1 次 | 51.76 |
| **Box-Gated IoU (ρ=0.84)** | 42 | IoU 排序 | **全局 box 门** | **63.231** | **−1.838** | 82.079 | 70.358 | 5500 | 2 次 | 45.70 |

四组对比均为 seed42 同协议同初始化配对；Clean A / F-Reg-Quality / Q-BoxOnly 数字取自各自
`predict_val/metrics.json`（统一 COCO bbox 评测，协议字段逐项一致）。

## 2. 两个问题的回答

**（1）在 F-Reg-Quality（66.143）基础上加入 DCNv4 能否获得额外检测收益？——不能（实测，同 seed42 配对）。**

- 组合 64.738，比 F-Reg-Quality **−1.40**，跌到低于 Clean A（−0.33）；AP50 从 84.583 大幅回落到 80.047（−4.54），
  甚至低于原生基线的 80.453；AP75 基本持平（71.890 vs 72.779）。
- 稳定性同步变差：出现一次 ≥20 点瞬态坍塌（1924→31.70，历史 F-Reg-Quality 无坍塌）、loss_ce 尖峰
  [2019,2399] 2.747、衰减前均值 51.76（vs 56.36）。
- 机制解读（保守）：`S8 + 0.1·DCNv4(S8)` 适配器与辅助回归选点管线叠加未产生正交收益，反而损害了
  高 IoU 阈值之外的宽召回（AP50 口径）。这是**当前协议、单 seed** 下的否定证据；不外推为"DCNv4 本身有害"，
  但"即插即用额外收益"的假设在本管线上被否定。未引入 FCOS 辅助头，SCDA 仅训练时生效，与设计一致。

**（2）GT Mask 的真实形状信息是否比普通矩形空间门控更有价值？——在 seed42 下，mask 门优于该全局
数量校准的 box 门（实测）；"形状先验本身更有价值"尚未被分离证明。**

> 修订说明（同日坍塌归因后）：初版此处的"确实更有价值"表述过强，已按下列限定改写；归因详见
> `collapse_attribution_20261016.md`。

- Box-Gated 用 ρ=0.84 的全局矩形门把候选池**总量**匹配到 mask 池（N=129,080 → box 93,002 vs
  mask 93,567，相对误差 −0.60%，128 张 train 图校准，schema v2 全链证据绑定）。
- 结果 63.231：比 F-Reg-Quality（真 mask 门）−2.91，比完全无门的 Q-BoxOnly（64.751）还低 −1.52，
  比 Clean A 低 −1.84。
- 该比较的三个已知混淆（同日已部分解决，见下）：
  1. **总量匹配 ≠ 组成匹配**：逐层候选池相对差达 [−13.1%, −5.1%, −4.8%, +0.9%]；
  2. **Box-Gated 训练自身遭遇 2 次坍塌**（含一次近零），坍塌压低了训练轨迹与 best——
     63.231 不是"box 门无干扰的上限"（坍塌归因见 `collapse_attribution_20261016.md`：均为真实瞬态
     失稳，评估链经 6 次逐位复评排除）；
  3. 单 seed 点估计。
- **后续反事实诊断（同日，`../../../runs/_launcher/quality-boxgate-a6000-s42/diagnostics/proposal_quality/`）
  进一步收窄了解释空间**：同一权重、同一批校准图、同一 forward 的 payload 上，mask 门与 ρ=0.84 box 门
  经 `assign_scda` 的选点平均重合 3.94–3.97/4，box 门无饥饿 GT、可证明被其排除的高 IoU 选中候选仅
  0–4/484，top-1 IoU 差 ≤0.0014。即：**两轮训练的回归监督目标几乎相同**，66.143 vs 63.231 的差距
  不能归因于门控几何或被选 proposal 质量，更可能由训练动力学（同 seed 数值路径差异的混沌放大 +
  Box-Gated 的 2 次坍塌）主导。
- 仍可安全陈述的：**(a)** seed42 下，两次训练的最终结果即 66.143 > 63.231，但两者监督几乎相同，
  差距主体是动力学噪声而非"mask 形状信息优于 box 门控"——后者作为机制命题**未被证明也部分被否定**；
  **(b)** mask 门与无门（Q-BoxOnly，池大 38%）的选点差异是实质性的，第三轮 +1.39 归因仍然可能是真实的，
  但单 seed 比较的有效噪声可达 1–3 AP 量级（本轮即为实例），多 seed 是下一步的必要条件；
  **(c)** Box-Gated 的正确归档身份是"同监督、不同数值路径的扰动对照"（它测出了管线的动力学敏感度），
  继续调 ρ 没有意义（选点几乎不会变）。
- 预先声明口径：该消融使用全局训练 mask 统计校准 ρ，**不是**"从未利用任何 mask 信息"的方案；
  运行时空间门控不读取逐 GT 形状。ρ 只匹配校准样本总体候选池数量，不声称逐图/逐层/选点数/梯度一致。

## 3. 稳定性对比

| 组 | ≥20 点坍塌 | loss_ce 尖峰窗（>2.0，去 warmup） | 衰减前均值 |
|---|---|---|---:|
| F-Reg-Quality s42 | 无 | [1679,1699] 2.136、[2699] 0.688 | 56.36 |
| Q-BoxOnly s42 | 无 | [2699,2719] 2.919、[3379] 2.657 | 54.21 |
| +DCNv4 | 1924→31.70 | [2019,2399] 2.747 | 51.76 |
| Box-Gated | 2199→0.57、2749→22.69 | [2419,2799] 2.608 | 45.70 |

两个新变体的训练稳定性都差于 F-Reg-Quality；Box-Gated 最差（含一次近零坍塌，均为瞬态、已恢复，
best 选择未受影响）。瞬态坍塌在该管线上有历史先例（F-Reg 4124→35.71%），best_val_ap 选择机制有效。

## 4. 对 SCDA 下一步设计的含义

- **F-Reg-Quality 仍是唯一超过原生基线的变体**（+1.07，另有 s43 复现 66.010）。本轮两条"改进路线"
  （算子增强、门控近似化）在同协议下均未超越，进一步支持第三轮结论：保留完整定义
  （辅助回归 + IoU 质量排序 + GT mask coverage 资格门），任何后续消融以它为基线。
- 对 DCNv4 的否定是"当前协议、组合形态、单 seed"下的：无法区分 DCNv4 自身与它和 SCDA 的交互；
  当前协议的 DCNv4-only 是正确的下一个实验（待授权）。"DCNv4 在旧框架下的收益"受旧补边管线与
  协议差异限制，不能外推到当前协议（见设计文档）。
- 资源不应再投向 DCNv4 组合或矩形门控近似；更值得做的是 mask 门机制本身的细化
  （τ_reg 敏感性、被选 proposal 质量/形状的测量）与多 seed 显著性验证——均属后续授权范围。
- 所有结论为单 seed 点估计，不构成统计结论；本轮未修改 SCDA 主线、未调参。
  三个坍塌点的归因结论见 `collapse_attribution_20261016.md`（真实瞬态失稳，评估链已排除）。

## 5. 执行与验收记录

- 启动前验收全部通过后才启动正式训练：DCNv4 P0 CUDA 算子门（官方后端 `4b848f7d`、参数量 160,368、
  初始恒等、有限差分通过）；双卡 190/280 iter 短跑探针（Box-Gated 初始 loss 6299 与历史 seed42 完全一致）；
  Box-Gated 梯度审计（cls 分支严格 0、box 分支全模块活跃、4GT×4 选点无缺额）。
- **本轮修复一个校准脚本真实 bug**（GPU 首跑暴露，CPU 替身测试覆盖不到）：
  `scripts/calibrate_scda_box_gate.py` 调 `load_coco_json` 未传 `dataset_name`，GT 标签保持 COCO 1 基，
  在 `num_classes=1` 的 DN `label_enc`（Embedding(1)）处触发 CUDA device-side assert。修复为与正式
  dataloader 一致的连续 id 重映射；不改变门控公式、ρ 网格、拟合目标、配额、损失或训练日程。
  失败/成功日志与 box-gate CPU 测试（26/26）记录在 `acceptance/`。
- 校准证据（schema_version=2）：ρ=0.84；N=129,080、M=93,567、box@0.84=93,002（相对误差 −0.60%）；
  eligible_gt=121，capped 选点 484/484、缺额 0；分层池 mask [1314,4941,15885,71427] vs
  box [1142,4690,15123,72047]；绑定 train 标注 sha `775b6574…`、128 张抽样训练图原始字节 SHA256、
  入口+`_BASE_` 模型配置 SHA256、15 个源文件、ImageNet R50 `16fd03f0…`、seed42。
  校准 JSON 嵌入 experiment.json 并另存 `runs/scda-boxgate-iou-pad0-a6000-b16-s42/box_gate_calibration.json`。

## 6. 产物

- 两个 run：`runs/proposal-dcn-freg-quality-pad0-a6000-b16-s42/`、`runs/scda-boxgate-iou-pad0-a6000-b16-s42/`
  （train/ + predict_val/ 完整，含原始 checkpoint、训练日志、完整 AP 曲线 train/metrics.json、统一 COCO bbox 评测）。
- 校准与配置：`runs/_launcher/quality-boxgate-a6000-s42/`（5 份冻结配置 + preparation.json + calibration.json）。
- 稳定性 JSON 与汇总：`runs/_launcher/quality-boxgate-a6000-s42/results/`
  （stability_*.json、round_summary.json）；启动/验收记录：同目录 `acceptance/`、`seq_gpu*.log`、`formal_*.log`。
- 数据哈希不变：train `775b6574…`、val `264a0532…`；初始化 `16fd03f0…`；MaskDINO `3831d851`；
  代码 HEAD `e134d03` + 校准脚本修复（快照随 run 保存）。探针目录已清理。
