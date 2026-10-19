# CIoU-Replace实现、验收与后台执行记录

**全部完成：北京时间2026-10-20 01:14。** 全部13阶段通过，[五组最终报告](../runs/ciou-competition-20261019/summary.md)已生成。CIoU按源域规则选定2750迭代权重，实际2740次成功更新。PolypGen上MSSL−CIoU AP为+1.136点，图像级95%区间[+0.209,+2.033]；AP90为+2.357点，区间[+0.815,+3.804]。AP75/AP95区间含零，源域AP90则为CIoU较高，不声称全指标优势。最终4张样本×21分支梯度验收通过，1473张外测及2000次Bootstrap完成，旧四组抽样向量与历史最大误差为零。下方运行中表述保留为历史记录。

2026-10-19：用户已审查通过并授权执行全部链路。只新增一组Seed42/50等效Epoch成熟定位对照；PolypsSet、PolypDB、PICCOLO、多Seed及V2均未安排。

## 完成状态与入口

**2026-10-20归档：** 三类测试、冻结GPU验收、smoke、正式训练、源域与PolypGen评估、检查点重放及五组汇总全部完成。统一读取[方法冻结与五组证据索引](mssl_v1_release_20261020.md)。smoke实际40次成功更新、最初10次AMP回退、最终scale64；正式训练从共同ImageNet初始化重新开始，未用smoke AP调设置。

冻结链路的[最终状态文件](../runs/ciou-competition-20261019/state.json)为 `completed`，无需继续等待原进程；其中PID仅供历史追溯。

- [冻结协议与输入哈希](../runs/ciou-competition-20261019/protocol.json)
- [状态、当前阶段和PID](../runs/ciou-competition-20261019/state.json)
- [后台日志](../runs/ciou-competition-20261019/pipeline.log)
- [依赖版本、Detectron2 commit及CIoU源码哈希](../runs/ciou-competition-20261019/dependencies.json)
- [冻结GPU全分支验收](../runs/ciou-competition-20261019/precheck/report.json)
- [各分支加权损失和梯度CSV](../runs/ciou-competition-20261019/precheck/branch_values_and_gradients.csv)
- [正式解析配置与基线的差异核对](../runs/ciou-competition-20261019/resolved_config_parity.json)
- [原四组与CIoU smoke初始化文件一致性](../runs/ciou-competition-20261019/initialization_parity.json)
- [smoke门控](../runs/ciou-competition-20261019/smoke_gate.json)
- [正式训练日志](../runs/ciou-competition-20261019/formal_train.log)
- [最终五组报告](../runs/ciou-competition-20261019/summary.md)

正式训练目录为`runs/ciou-replace-s42-20261019T073936Z/`；独立smoke为`runs/ciou-smoke-20261019T073936Z/`，不复用其权重开始正式训练。

Windows主目录为`D:\code\work2_polypy_det\runs\ciou-competition-20261019`。链路已经完成，无需为该任务继续保持后台会话。原执行规则为异常写入failed并停止后续阶段，失败记录保留。

## 固定实现

[独立代码入口](../methods/ciou_replace/README.md)通过可选`box_regression`配置选择CIoU模型/Criterion，复用原训练器、共同原始Mask管线及原生推理。此前四组冻结源码、权重、预测保持原样。

使用Detectron2 0.6、commit `1e3e13bbf607b54f62205c4c33922521822fb298`的`ciou_loss`；`losses.py` SHA256 `6426d06f915f12f0eb5bf4f31f94092b23422d663337a22fc27a8b4ea8630ccb`。PyTorch2.7.1+cu128、Torchvision0.22.1+cu128。eps1e-7、alpha停止梯度、无IoU门控。

CIoU用FP32计算，不重复解码、不裁剪或删除无效框。原`loss_giou*`键为原生层级/权重兼容而保留，槽位实际为CIoU，元数据明确标记。Matcher的GIoU成本、L1权重5、替换回归权重2及原生归一化保持。

## 验收证据及边界

- 新增7项组合测试通过，覆盖参考实现数值/梯度、固定alpha有限差分、非法尺寸及空匹配、原生其他损失与匹配/RNG、多框、结构性零梯度及旧抽样精确复用。
- [开发GPU验收r4](../runs/ciou-implementation-20261019/precheck-r4/report.json)通过。一个单框、一个增强后三框图像，各21个回归分支，保存加权CIoU/GIoU值、坐标及原生共享框头梯度、比值与实际系数。
- 匹配和Torch随机状态完全一致；其他损失按FP32容差检查，首图`loss_mask_dn`最大绝对差9.5367e-7，其余为零。报告保留差值及原生重复重放值，不宣称所有GPU标量逐位相同。
- 移除Criterion前后，原生推理的框、分数、类别、Mask完全一致；Criterion没有新增训练参数，保留原生buffer。
- 冻结后重新运行GPU验收及测试：CIoU7项、MSSL回归15项、COCO分析4项。冻结目录下MSSL历史路径依赖项跳过1项；另在原工作区运行完整15项，见[工作区回归日志](../runs/ciou-implementation-20261019/mssl-workspace-regression.log)。

开发失败记录r1/r2/r3均保留：依次是把原生buffer误当新参数的断言、对初始化分支要求非零梯度的过强断言、DN Mask单ULP差异触发逐位断言。修正的是验收定义，未为通过验收更改框坐标、Mask计算或原生detach。

当前INITIAL_PRED与Mask增强初始化下，`loss_giou_0`、`loss_giou_dn_0`对应停止梯度的初始坐标，原生GIoU和CIoU对框头梯度均为零；其余19个可学习分支在固定样本上有有限实际梯度。两项初始化损失仍按原生规则替换、计算并列出，不能为制造非零梯度而取消detach。

## 自动顺序

1. CIoU、既有MSSL及COCO/Bootstrap回归检查。
2. 冻结真实GPU全分支验收。
3. 50迭代smoke训练、权重重载及完整源域验证、数值门控。
4. 唯一新增正式训练：2750预定迭代，源域最大bbox AP选权重。
5. 正式选定权重的源域预测与COCO评估。
6. 最终检查点4张固定训练图像的梯度重放，无更新、无调权重。
7. 固定1473张PolypGen推理，先核对源域单图预测一致性。
8. 原定pooled/分中心/重复敏感性评估。
9. 五组统一结果、原2000次配对Bootstrap及最终报告。

抽样计数逐字节复用，旧四组全部抽样向量与归档结果核对。区间包含零不判胜负；图像级Bootstrap不替代患者或训练Seed不确定性。该比较不扩大为Mask或CDF必然优于Box-only/CIoU。
