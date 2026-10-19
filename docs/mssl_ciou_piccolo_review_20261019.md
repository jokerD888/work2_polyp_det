# MSSL V1：CIoU-Replace与PICCOLO执行前核对

日期：2026-10-19。依据是用户粘贴的网页端执行规格摘要、公开原始来源及本地代码。未收到摘要提及的三份文件全文，不将本核对写成对三份全文的逐项验收。

当前状态（2026-10-20）：CIoU-Replace已完成数值/梯度/GPU验收、正式训练与五组评估，见[方法冻结与五组证据索引](mssl_v1_release_20261020.md)。PICCOLO申请已按后续用户决定暂停；PolypsSet/PolypDB独立外测尚未开始。下文是2026-10-19执行前的核对记录，PICCOLO并行申请建议已被后续决定取代。

## 1. 总体决定

原安排可推进。冻结V1，采用一个成熟Box-only定位替代对照；PICCOLO申请与对照实现并行，不等待外部审批才开始本机工作。近邻讨论与训练基线选型分开，不增加第二种定位训练对照。

## 2. CFIoU术语修正

[CFIoU原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10223589/)：[出版社及DOI](https://doi.org/10.3390/s23104961)，Sensors 2023, 23(10), 4961。

原文§3的公式使用预测矩形框B、目标矩形框Bgt、最小外包矩形C、角点距离和框面积关系。“foreground area”由目标框几何表示，不是逐像素GT Mask或Mask投影。论文有真实检测器训练，可列为Box-only几何回归优化近邻；不能说它已经证明精细前景Mask参与框回归的收益，也不能把它与MSSL的监督来源写成相同。

Related Work分别讨论CFIoU/CIoU等Box-only几何优化、MbIoU的Mask覆盖度量、DSC等预测Mask反馈，以及MSSL的GT方向性前景支持。机制差异不是首次或穷尽查新的证明。

## 3. 本地回归路径静态审计

沿用此前四组的Mask DINO版本`3831d8514a3728535ace8d4ecc7d28044c42dd14`。依据：

- `third_party/MaskDINO/maskdino/modeling/criterion.py`：`loss_boxes()`计算L1及匹配框对的`1-diag(GIoU)`，除以原生`num_boxes`；`get_loss('boxes',...)`按配置路由到该函数或panoptic函数。
- `criterion.py::forward()`：最终层、`aux_outputs`、已知DN输出及各层DN输出、`interm_outputs`均经`get_loss`计算回归项；DN采用原生`num_masks*scalar`归一化。
- `third_party/MaskDINO/maskdino/modeling/matcher.py`：独立计算`cost_giou=-GIoU`，应保留。
- 已训练基线`runs/maskdino-mssl-control-s42-20261018T025946/train/config.yaml`：`DN=seg`、`DN_NUM=100`、深监督和两阶段开启、panoptic关闭；L1/GIoU回归权重5/2，Matching的Box/GIoU成本5/2。

CIoU-Replace应替换原生最终/辅助/Encoder/DN回归路径的GIoU，不只在MSSL最终层挂载点添加CIoU；关闭MSSL额外项，不继承其可靠单框筛选，所有原生有效目标继续参与回归。保留L1、Mask监督、匹配、各分支归一化、原系数与原生推理；不引入Cluster-NMS。Matching函数不变，不意味着训练中的实际匹配结果也必须相同。

以上为静态审计，实际损失键、分支覆盖及模型前向仍须集成验收。

## 4. CIoU实现约定和验收

[作者实现](https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py)示例含输入Sigmoid、宽高exp、IoU门控和alpha停止梯度等约定。Mask DINO回归损失接收已解码的归一化cxcywh框，不能重复Sigmoid/exp。

训练前固定唯一版本：公式、坐标转换、alpha停止梯度、IoU门控、eps、dtype和reduction。公式与具体实现有差异时写清依据，不混用多个版本或训练后更换。只计算原生对应匹配框对，不将全部N×N框对求和。保留分支归一化及权重2；日志如保留`loss_giou`兼容名称，须说明其槽位实际已替换为CIoU。

验收覆盖相同框、非重叠框、同中心不同尺度、不同长宽比、极小正尺寸、空匹配；FP32/float64数值对照、既定alpha语义的梯度、各原生分支键和权重、Matching保持原样、真实GPU/AMP短试跑及检查点重载。停止梯度的alpha不能用每次重新计算alpha的完整函数有限差分作错误的梯度标准；空匹配需给可反传零值。

通过后开启唯一新增Seed42、50等效Epoch正式对照；复用数据、初始化、优化、增强、累积和源域选Checkpoint规则。不按PolypGen调参，不新增MSSL+CIoU组合。

## 5. PICCOLO申请与评价准备

[Basque Biobank官方PICCOLO页面](https://www.biobancovasco.bioef.eus/en/Sample-and-data-catalog/Databases/PD178-PICCOLO-EN.html)提供[申请表](https://labur.eus/EzJUN)和联系渠道。Zenodo不是完整图像Mask包。官网一般说明写76病灶/48患者，技术说明写71病灶/39视频，且与Zenodo40患者口径不同；最终数量、分组及划分以收到的数据和元数据审计为准。

官方列出RGB图像、病灶Mask及void区域Mask。除五组模型、WLI/NBI、Mask-derived Box及重叠检查外，协议须提前固定void的评价作用、有效图像区域、空Mask/void重叠/多病灶处理及是否改变输入。默认保留原生RGB推理，不为某模型单独裁剪或屏蔽图像。

不看预测再决定如何忽略void内误报。如需忽略，明确共同框级规则及评估器并提前验证。不能把每个连通分量自动当作独立病灶，实例拆分需有标注依据。

申请可能需要申请人、机构和用途等真实信息；本轮仅核对渠道，未代填、提交或发送消息。审批不阻塞本机CIoU准备。数据到位前，外测是协议草案，不是已有完整文件列表和哈希的冻结队列。

## 6. 下一项实际工作

1. 实现隔离的CIoU-Replace，固定实现约定和完整训练配置。
2. 完成公式、分支覆盖及真实GPU短试跑验收，通过后运行一组正式对照。
3. 同步准备PICCOLO申请材料与转换规格，数据到位后冻结实际清单。
4. 固定五组检查点后统一外测，更新论文竞争比较和独立确认结论。

不再讨论研究方向，不安排V2、新模块或第二个成熟对照。跨Seed稳定性仍是尚未验证的主张。
