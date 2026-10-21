# 空间支持参考框校准：真实Detector工程验收

本轮已完成[规格v0](support_reference_mechanism_spec_20261021.md)的工程接入和无optimizer反传。**验收通过，可以进入受控训练准备；不表示涨点或第二创新已经成立。** 没有完整训练、阈值调整或PolypGen输入。结果见[汇总数据](results/support_reference_engineering_summary_20261021.json)及本机`runs/support-reference-acceptance-20261021-r5/`。

## 固定范围

- 本机WSL Ubuntu-22.04，原生固定Mask DINO版本；Baseline/MSSL V1既有checkpoint。
- 隔离接入代码为`methods/support_reference/integration.py`，不改third_party源文件。禁用调用原始bound forward；启用在Mask-to-Box后、DN拼接前运行校准器。
- 新参数正常注册到Decoder的state_dict；checkpoint仅允许缺少新增adapter键，其他漏载/多余键报错。关闭上下文时移除模块、hook及临时forward。
- 有效画布由输入image尺寸、LSJ padding_mask和原生批次补齐尺寸组成，不读取targets。Mask格点采用nearest-exact中心代表采样；不将原生Decoder里被清零的attention mask冒充图像有效区域。
- 按Kvasir Train最小GT相对面积四分位×单/多目标分层，每个非空组最多两张，组内按固定hash选样，不使用预测成败。实际为13张、两个固定LSJ种子42/31415：Baseline26例、MSSL8例复核；同图增强视图不当独立统计样本。
- 两模型各4张源域FP32图像检查原生/禁用/零输出推理一致性；训练模式使用原分辨率1024、AMP，另检查固定非零残差和真实batch2 DN拼接。没有optimizer、没有参数更新。

## 选样限制与试运行记录

训练图按最小GT相对面积分层：多目标q1/q2/q3/q4为40/4/1/0张，单目标为180/216/219/220张。q3多目标只有一张，q4没有多目标；不复制或伪造样本填满16格。这是按图最小目标面积分布，不是逐GT总体面积分布。

1. `runs/support-reference-acceptance-20261021/`：原先错误假定每格有两张，标注预检时停止，没有模型前向。记录保留。
2. `runs/support-reference-acceptance-20261021-r2/`：观察器没有处理原生eval阶段aux0 Mask为None，在首张原生推理的记录阶段失败。修复为保留None；不改模型行为，记录保留。
3. `runs/support-reference-acceptance-20261021-r3/`：FP32推理一致性通过，首个训练模式对比失败。验收脚本原先将no_grad关闭分支与有梯度启用分支比较，修正为相同梯度模式，并保存后续任何差异的逐项记录；没有放宽容差。
4. `runs/support-reference-acceptance-20261021-r4/`：同梯度模式的推理及已执行训练视图一致性通过。发现观察器的bound callback循环引用延后释放图，峰值12.59 GiB，显著拖慢执行，主动中止保留日志。另发现“adapter末层范数”的切片误含原生参数；主adapter范数及判定不受影响。修复释放逻辑与统计切片，增加独立CPU回归检查，未改算法或数值容差。
5. `runs/support-reference-acceptance-20261021-r5/`：正式从头验收全部通过，源码快照、固定panel和35个源文件/输入SHA复核通过。独立16项CPU检查通过；真实GPU验收耗时89.53秒。以前失败/中止结果不混入本轮汇总。

## 最终结果

| 项目 | Baseline | MSSL复核 |
|---|---:|---:|
| 固定图像数/LSJ视图数 | 13/26 | 4/8 |
| 有效匹配/全部GT匹配 | 35/36 | 15/15 |
| 有效Query观测/全部 | 7709/7800 | 2394/2400 |
| 无有效匹配的视图 | 0 | 0 |
| 工程峰值已分配显存GiB | 6.576 | 6.770 |
| aux0框损失对adapter梯度范数中位数 | 3.7573 | 2.9786 |

有效匹配仅指原生aux0匹配Query具有非空、非单行/单列的预测支持，能够接受校准框监督，不是恢复病灶数、检测召回或可靠实例Mask数量。重复增强及两模型同图不是独立GT证据。按分层仍只有一张q3多目标、没有q4多目标，不外推总体。

八张FP32源域推理的原生/关闭/零输出结果逐项一致：Top300身份、content、参考框、早期/最终Mask、分类、框、Top100 query及分类/Mask质量/最终分数。34个训练视图在同一有梯度AMP模式下严格一致，包含原生DN输入及初始/最终框、全部损失数值；两模型还各通过一组真实batch2（保留GT数4与1）及固定非零残差的直接DN框隔离。

所有分解梯度有限。aux0仅作用于校准器新增路径，未接入原生被测Box/Mask末层bias；Encoder interm和DN初始项对adapter均为0。首个真实Decoder框损失能回传，最终框损失对adapter梯度通常更小，部分值为0，不能将AMP下的0都解释为完全断路。后续DN经注意力的间接梯度非零，符合原生实现。MSSL组4个多目标视图没有原有MSSL资格，另4个单目标视图保留最终层MSSL；不能把包含这些资格差异的中位数解释为协同证据。

同一裁剪病例（image431、seed42）在两个模型中各有1个GT Mask被裁空但provided Box保留，预测支持仍可校准。保留原生框监督并记录缺额，不从GT空状态生成推理开关，也不伪造实例Mask。规格已澄清GT空Mask与预测空支持的区别。

关闭观察器、每模型单张固定图预热后5次FP32测量：Baseline中位数0.1069→0.1113秒，MSSL0.1058→0.1098秒，约增加4.0–4.4毫秒。仅为工程样本，不是全数据集吞吐基准。显存也是无optimizer、选择参数autograd.grad验收的峰值，不包含完整训练的全部梯度与optimizer状态。

## DN验收定义

“隔离”指校准器不改DN输入框或GT加噪过程，不给DN复制校准损失；不要求所有DN损失对校准器的梯度为0。原生attention mask允许DN读取普通Query，后续DN损失可以经普通Query隐藏状态间接回传。

因此分别检查：DN原始reference及aux0框数值一致、DN初始损失对adapter无梯度、后续DN的间接梯度实测记录。Encoder interm损失对adapter也应为0；普通aux0框损失应能更新有效匹配的adapter，却不能通过detach描述子更新原生Encoder/Mask预测头。

## 结论边界

真实工程面板使用已经训练好的模型权重。通过只说明接入、当前源域训练视图的监督可用性和梯度路径，不证明从原始预训练开始能收敛，不证明空间profile优于普通几何头，也不证明bbox AP或创新性。最终性能仍以完整受控实验、PolypGen主开发评价为依据。

下一步可以准备同起点/同预算的Baseline独立对照：固定融合重训、普通几何校准、空间支持校准；训练接入时先做原始预训练起点的有限优化smoke，检查optimizer参数发现及收敛数值，再放完整预算。只有空间支持相对普通几何提供总体增量，才进入MSSL组合。不再追加本轮PolypGen诊断，不自动排入50epoch队列。

## 五维复核

贡献：仍可能是普通小头，待独立信息对照。清晰度：挂载、坐标、DN直接与间接路径均已明确。实验强度：有真实模型/AMP/batch2/反传证据，尚无优化和性能证据。评价完整性：保留旁路、裁空和空分层；PolypGen主评估角色不变。方法合理性：有直接框监督路径，未证明训练后的好框保护或跨域收益。
