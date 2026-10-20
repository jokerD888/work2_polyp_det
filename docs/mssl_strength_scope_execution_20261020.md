# MSSL E/F执行记录（2026-10-20）

用户已授权实现和启动固定有限队列。本轮只新增E/F两次完整训练，既有V1与Baseline复用。

| 模型 | 额外损失 | 作用位置 |
|---|---|---|
| B：已完成MSSL V1 | Lf | 最终普通Decoder |
| E | 2Lf | 最终普通Decoder |
| F | Lf+La | 最终与倒数第二个普通Decoder |

`La=outputs.aux_outputs[-1]`，复用该层原生Hungarian匹配。两个系数显式设置，外部MSSL权重固定1，每层保留四边平均和可靠GT归一化。E亦测量La，但其目标系数为0。原生分类、Mask、L1/GIoU、DN、Encoder和其他辅助层损失保持原样。

两组均从共同ImageNet R50初始化，Seed42，2750预定迭代、50等效Epoch，微批量1、累积16，原增强/LR/275周期源域Val最高bbox AP选权规则不变。不会从V1、Q或Smoke权重接续。

## 验收与队列

开发检查：原15项MSSL回归全通过；新增3项回归通过（原生匹配/损失/RNG与坐标梯度代数、探针不改变RNG/.grad/反传、拒绝歧义配置）。E/F真实单目标与多目标GPU验收均通过，无参数更新；初始化全张量一致、匹配与Torch RNG一致、推理逐张量一致。原生GPU损失允许既有`rtol=2e-6, atol=1e-6`舍入容差，额外坐标损失与梯度代数精确一致。

开发验收文件：`runs/mssl-scope-engineering-20261020/{E,F}-precheck/report.json`。

后台队列：`runs/mssl-strength-scope-20261020-r1/`。

- `protocol.json`：同时冻结E/F配置、源码SHA、既有对照和固定评估输入SHA。
- `queue.json`：26项任务的开始、结束和状态。
- `state.json`：当前阶段、控制器PID及子进程PID。
- `pipeline.log`与各阶段日志：故障定位；任何阶段失败即停止后续任务。
- `source/`：执行源码与正式/短跑配置快照；不会因工作区后续修改漂移。
- `windows_power_guard.json`：运行期间阻止自动系统睡眠，允许屏幕熄灭；完成/失败后解除。

顺序：新/旧/Bootstrap回归 → 两组GPU验收 → E与F各50迭代Smoke（训练、完整Val推理、数值门槛） → E完整训练/选权推理/最佳权重验收 → F同样流程 → 固定目标图像SHA验证 → 两组各一次PolypGen开发复核 → 全300 Query源域重放、同GT错排和Top100筛选/排序诊断 → A/B/E/F比较。

两组Smoke门槛均通过后才开始任一正式训练，不用Smoke AP作放行条件。检查成功优化更新、有限数值、正确损失路由和可加载选定权重。正式训练和尾部评估均已列入队列，正常运行无须逐项手动启动。

## 梯度与结论边界

首微批次迭代0、10、每275迭代记录共享bbox头`gf/ga/total`与原生最终/全部回归梯度的范数、比值和余弦；记录名义系数、实际分层损失及FP16线性组合残差。测量使用统一缩放、预定溢出回退，不修改训练GradScaler、RNG、参数或`.grad`。零范数余弦为null，有限精度边界写入JSON。增大梯度不是成功标准。

主指标AP50:95，同时报告AP50/AP75/AP90/AP95、候选覆盖与重复FP。报告E−B、F−B、F−E；PolypGen按原2000次中心分层图像重采样复用抽样与A/B向量，区间不代表多Seed或患者级不确定性。源域Val120张参与开发，PolypGen属于已观察的外部开发复核。

若E/F均无有说服力的总体收益，保持V1，停止扩大权重和监督层级。即使提升，也只是既有MSSL机制优化，不自动构成第二创新点。不追加Quality Head、DCNv4、局部对比、权重网格或使用PolypsSet/PolypDB。

预计总时长需按实际Smoke吞吐更新。优先连续运行；若中断，单独记录恢复边界，不能承诺旧检查点未保存RNG时逐位复现连续轨迹。
