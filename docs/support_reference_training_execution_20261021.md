# 初始化第二机制：冷启动优化与三条件完整实验

用户已授权执行：正式训练接入 → 固定20次优化Smoke → 同预算独立对照。MSSL组合不是本队列任务，待独立收益审查后决定。本文记录实际执行，不将工程通过或单seed增益称为第二创新。

## 固定实验范围

| 条件 | 推理初始化 | 新参数 | 执行 |
|---|---|---:|---|
| native | 原生Mask-to-Box | 0 | 冷启动Smoke参照；完整Baseline已有 |
| blend05 | 归一化cxcywh两框固定0.5融合 | 0 | Smoke后独立完整训练 |
| geometry | 原生Mask参考logits＋校准残差，只用16维全局标量 | 5444 | Smoke后独立完整训练 |
| support | 同校准器，额外使用64维空间profile | 5444 | Smoke后独立完整训练 |

geometry保留相同80→64→4参数结构，profile输入置0；相同参数存储不等于相同有效容量。两组均使用原生aux0框损失提供新增参数梯度，不增加Quality Head或GT质量标签。固定融合不使用新增回归头。Matcher公式不变，学习后匹配结果可以变化。

所有新增完整组：原始ImageNet R50权重资产，Seed42、2750个计划优化迭代、microbatch1×累积16、50等效epoch、1024 LSJ(0.1–2.0)、原学习率/裁剪/Mask及分类/框监督。每275迭代在Kvasir Val选最高bbox AP50:95、相同则取最早。复用原生Baseline完整参照。Smoke权重不续训，不能把其20次成功更新计入或混入完整预算。

## 源码与入口

新增`methods/support_reference/model.py`在正式模型构造时注册adapter，因此早于原生optimizer创建；`methods.mssl.config`仅在显式support_reference字段下切换meta-architecture，既有V1配置行为不变。训练/推理入口为`python -m methods.support_reference`。预测严格加载包括新增adapter的所有键。

正式worker新增原生初始化SHA门禁和optimizer参数发现记录。三条件真正的DefaultTrainer启动预检必须全部通过，才开始任何长训练。新的核心模型/数据/累积实现与通过Smoke的源码逐项比对SHA。队列源码快照独立冻结，工作区后续修改不影响后台运行。

## 冷启动Smoke

目录：`runs/support-reference-cold-smoke-20261021/`。四条件顺序运行，分别从原始资产初始化。每组要求20次**成功**AMP更新，最多40次尝试；溢出跳过不计更新。原effective batch16及学习率调度保留，不计算AP，不按早期loss幅度选方法。

检查内容：全部可训练参数恰好进入optimizer一次；新参数数目0/5444符合条件；零输出末层更新后隐藏层取得有限非零梯度，参数实际更新；全部权重有限；模型/optimizer/scheduler/GradScaler状态可保存恢复；严格reload后图像预测相同。全部条件的原生起点参数SHA与消费到的图像/GT Box增强序列前缀必须相同。

当前回归：19项CPU检查、15项既有MSSL检查通过。第19项检查验证：打乱历史annotation顺序时仍按GT ID正确统计恢复/丢失，并保留负图误报。

四组均已达到20成功更新，各有前10次AMP跳过，均为30次尝试；最终loss和参数有限，checkpoint及状态恢复和预测一致。四组原生起点SHA同为`4f4017a861f05148cb91c38eecf02f032c0030e0ea7c0f15b559ce04d0605571`，消费到的480个microbatch图像/Box增强序列前缀逐项相同。geometry/support隐藏层均取得非零有限梯度，4个新增参数张量均实际更新。数据序列一致是四组Smoke之间的检查，不声称手工Smoke循环与DefaultTrainer启动的数据迭代时机完全相同。正式入口另做共同原生参数起点门禁。

四组耗时约4.2分钟/组，峰值约6.1GiB（含optimizer与AMP真实反传）。该结果仅确认冷启动的短程数值可行性，不评价最终效果。梯度监测中的after-clip范数只在成功更新行按此解释，AMP跳过行没有执行optimizer裁剪。

## 评估与终止范围

额外collector先在每个Smoke模型的两张Kvasir Train图片验证观察与普通推理完全一致、四阶段框导出和Top100映射正确，不评价AP。之后才允许正式长任务队列启动。

三组完整训练全部结束后，对各自Kvasir Val选定checkpoint做一次固定1473张PolypGen开发集评估。推理不带GT；同时小体积导出Encoder框、原生Mask框、实际初始化框、最终300框及Top100身份/分数。GT只进入后续CPU评价。

输出总体AP50:95、AP50/75/90/95，阶段一对一覆盖及逐GT最大IoU覆盖，覆盖恢复/新增丢失/净变，中心/相对面积四分位/单多目标分组，重复FP、定位FP、背景FP和负图误报。分别训练的Query没有共享身份，不能将GT池的恢复冒充同Query因果恢复。

PolypGen是主要外部开发基准；Kvasir Val负责选checkpoint和辅助检查。PolypsSet、PolypDB保持未使用。所有组跑完前不按中途AP改配置；工程失败停队列，保留记录后修复验收，不自动换阈值、增加训练或组合MSSL。

## 当前执行状态

四组冷启动Smoke及共同起点门禁已完成；collector预检也已通过：每个新条件使用Kvasir Train图像1与4，观察前后预测逐位相同，四阶段300框与Top100映射验收通过，无AP评价。预检目录为`runs/support-reference-external-preflight-20261021/`。

完整实验代码和固定协议已准备，下一步由冻结控制器完成19项CPU、15项V1回归及三个正式DefaultTrainer启动门禁，然后顺序运行三组训练。最终队列位置和启动回执将在任务执行后补充。
