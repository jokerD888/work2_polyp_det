# MSSL V1 实现与启动记录

**当前状态（2026-10-20）：** 本文所述MSSL与新基线已完成训练及评估；机制对照、PolypGen分析和CIoU比较也已完成。当前以[方法冻结与五组证据索引](mssl_v1_release_20261020.md)为准。下文验收数字和调度细节保留为2026-10-18实现记录。

日期：2026-10-18。方法代码：[methods/mssl](../methods/mssl/README.md)。

## 实现范围

独立模块／runner，不修改官方 MaskDINO、旧 SCDA、项目默认配置或共享 mapper。
只有最终 Decoder 的匹配框增加 GT Mask 支持 CDF 损失，复用该次原生匹配。
方向支持为轴向前景质量乘以靠近对应 GT 边的指数权重；平方 rho，温度 0.05、
方向带宽 0.15、最小尺度 4 像素、辅助权重 1.0。全部为待验证初值。
GT Box 不由 Mask 重算，没有预测 Mask 教师，没有新增学习参数／推理头。

单框训练图使用原始语义 Mask，且新基线的原生 Mask Loss 也采用同一规则；
原始多框图维持已有原生标签，跳过新增监督。保留零补边修复和官方检测框。
因此不能直接与旧分辨率／裁剪 Mask 协议的历史结果作方法归因比较。

## 官方设置与资源适配

沿用固定官方 R50 结构、ImageNet 初始化、原生损失和匹配、AdamW 1e-4、
backbone 0.1、WD 0.05、梯度裁剪 0.01、AMP、1024 输入、LSJ 0.1–2.0。
官方约 50 epoch 的曝光预算映射为 Kvasir 2750 个更新迭代，衰减点 2444/2648。
验证／保存每 275 步，最高 val bbox AP50:95 选权重；seed 42。
测试保留解析出的原生 resize 800/1333，关闭 TTA。

本机 RTX 5070 12GB：microbatch 1 × 梯度累积 16，有效 batch 16。
原生每微批次归一化和 DN 构造意味着不与物理 batch 16 完全数学等价。
新基线同样采用这一适配；不把本配置描述为完全复现原论文 COCO 训练。

## 已完成验收

- MSSL 9 项测试全部通过，包括 GPU 累积一次更新、数值梯度、Mask 额外信息、
  空目标、多框排除、与原生 aux／encoder 损失及 RNG 精确一致，以及末尾仅评估记录的正确读取。
- 现有项目测试在 WSL 模型环境 **22 项全部通过，无跳过**。
- 真实图像 GPU 预检：15 条增强样本含原始多框图；原生 105 项损失与 RNG
  精确一致；MSSL 对最终框及原生框头非零梯度；完整 AMP 反向和一次有效更新；
  移除 criterion 后仅图像推理结果精确一致。
- 该样本 1024 训练峰值分配显存约 6144 MiB。

证据：[GPU report](../runs/mssl-implementation-20261018/precheck-3/report.json)、
[MSSL tests](../runs/mssl-implementation-20261018/tests-final.log)、
[现有测试](../runs/mssl-implementation-20261018/existing-wsl-tests.log)。

初始 AMP loss scale 出现溢出，遵循原生 GradScaler 自动回退，真实预检在 scale=64
获得有限梯度与有效更新。前两份预检保留了初始检查假设过严／回退预算不足的失败记录，
没有掩盖或将跳过更新计为成功。连续 smoke 另验证完整累积训练，不以单图预检代替。

## 后台执行与定位

[campaign](../runs/mssl-implementation-20261018/campaign-r2/campaign.json) 固定了源码副本、
配置与哈希。后台顺序为：16 步 smoke train → 加载 best 的完整 val 预测 → 健康门槛 →
2750 步 MSSL 正式训练／预测 → 相同数据与预算的新基线训练／预测。
任何阶段报错或冻结源码被改动，停止后续阶段，不继续占用资源盲跑。

最新状态看 [state.json](../runs/mssl-implementation-20261018/campaign-r2/state.json)，
smoke 验收看 [smoke_gate.json](../runs/mssl-implementation-20261018/campaign-r2/smoke_gate.json)，
各阶段日志与 Linux PID 记录在同一目录。阶段完成前可能尚不存在 gate 文件。
训练使用冻结副本，当前工作区后续编辑不会悄悄改变已排定的对照。

首个 campaign 的 smoke 训练与重载评估均完成；调度器误读最后的仅评估行而停止。
已修正并补充回归测试，旧失败记录保留。r2 复用该已完成 smoke 前，严格核对七个
训练关键模块哈希、训练设置、完整数据与原始 Mask 哈希，仅免除重复的同构试跑。
实际 smoke 为 16 步／6 次有效 AMP 更新，最后 scale=64、无跳过、MSSL loss 有限。

正式组 `mssl-official50-s42-20261018T025946` 与新基线
`maskdino-mssl-control-s42-20261018T025946` 均已完成。两组各2750预定迭代、
2740次成功更新，均按源域Val最大AP选中2750迭代权重。历史PID不是当前运行状态。

本节验收本身只证明工程健康；性能证据与限制另见当前发布索引，不由验收通过自动推出。
