# 第一层双参照特征读取：H1/H2 最小规格

本轮是候选机制验证，不是已经成立的第二创新。MSSL V1继续保留；不运行MSSL+blend05，也不继续扩大坐标校准头。

## 问题和边界

同预算blend05在PolypGen上AP49.436（Baseline48.818），AP50提高3.396个百分点、AP90下降3.930个百分点；普通geometry和support分别48.398/48.332。已有证据说明改变初始化存在覆盖与精定位取舍，但不能证明补充空间采样有效。

固定版本MaskDINO的同一个reference_points影响位置编码、cross-attention采样和迭代回归。新实验只改变**额外**分支的读取参照，不改原生Mask-to-Box、Top300索引、内容Query、位置编码生成、主采样、回归参考或原生损失定义。

## 确切结构

仅在第0个Decoder层的cross-attention输出、原生dropout/norm之前加入额外残差：

`CA_out = CA_native(q_mask, r_mask, F) + W · CA_extra(q_mask, r_extra, F)`

两组额外CA均为第一层原生MSDeformAttn的**独立可训练副本**，起始权重逐张量相同；W是256×256、无bias的线性投影，初始化为零。整个新增分支构造保留CPU/CUDA随机序列。未使用双零门控、阈值、可靠性头或额外GT目标。

| 条件 | r_extra | 其他差异 |
|---|---|---|
| 已有Baseline | 无额外分支 | 原生参照 |
| H1 mask_extra | 原生第一层Mask参考、已经乘native valid_ratios | 无 |
| H2 encoder_extra | 同一Top300 Encoder box的sigmoid，按原生公式乘各层valid_ratios | 无 |

H1/H2使用同一Mask位置编码和同一query内容；区别仅在额外采样参考。额外模块、参数数目、采样点/尺度/Query数和融合形式一致。不是共享同一个可训练CA模块执行两次，也不是先平均两个框。

额外CA只处理末尾300个普通Query；DN前缀保持原生cross-attention结果，不添加残差、不更改DN参考。后续Decoder层仍用原生迭代更新公式；因为普通Query特征发生变化，后续框及DN通过原有交互产生间接变化是允许的，不能宣称整条定位路径训练后不变。

INITIAL_PRED仍开启。辅助输出0是Decoder前的预测，与新增分支无直接梯度；辅助输出1才是第一层Decoder输出。验收分别检查initial/encoder/first-decoder/DN-first-decoder梯度路由。零W时额外采样器第一步梯度为零是预期，W更新后必须观察到额外采样器的非零有限梯度。

Encoder坐标仅作额外采样、保持detach；共享memory和query可接收原生检测/分割损失的梯度。推理不使用GT。不能补回Encoder Top300之前已经丢失且没有对应Query的目标，也不保证域外精定位得到保护。

## 查新约束

- [Mask DINO，CVPR2023](https://arxiv.org/abs/2206.02777)：已有Mask增强初始化，不能把Mask初始化称为新机制。
- [DAB-DETR，ICLR2022](https://arxiv.org/abs/2201.12329)：参考框提供空间查询与注意力先验已有基础。
- [DI-MaskDINO，NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)：研究早期Decoder检测与分割不平衡，提出DI和BATO。不能只因聚焦早期层就主张创新；本轮不移植这两个模块，研究的是固定原生几何下第二采样参照的独立价值。
- [MSTR，CVPR2022补充材料](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Kim_MSTR_Multi-Scale_Transformer_CVPR_2022_supplemental.pdf)及[HOI视觉上下文，ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Exploring_Predicate_Visual_Context_in_Detecting_of_Human-Object_Interactions_ICCV_2023_paper.pdf)：其他任务已有多实体/双参照空间读取，因此“使用两个参照点/框”本身不足以宣称首次。

当前定向查新没有完成对全部相关机制的穷举，也未证明方法新颖。H2即使胜出，也须进一步厘清与既有双参照机制的差异；先以有限受控实验检验科学问题，不提前命名为论文贡献。

## 执行与停止点

1. 源域真实Baseline验收：禁用/零残差的FP32预测、AMP训练损失与中间量一致；双图DN批次；非零探针；坐标/valid_ratios；实际GPU时延和内存。
2. 三条件native/H1/H2从原始ImageNet R50起点各做20次成功优化Smoke（最多40次尝试），不计算AP；检查共同native参数SHA、图像/框增强序列、参数发现、两阶段梯度和严格检查点重载。Smoke权重不进入正式训练。
3. Collector源域预检及两组正式DefaultTrainer起点门禁通过后，H1/H2各Seed42、2750计划迭代、micro1×累积16、50等效Epoch，原生优化器/增强/损失/学习率不变。只用Kvasir Val按既有规则选点。
4. 两组全部结束后，固定1473张PolypGen主要外部开发评估，报告AP50:95/50/75/90/95、候选覆盖净得失、负图误报/重复/定位FP及中心/尺度/单多目标。PolypsSet/PolypDB保留。
5. H2须同时优于原生和H1、并审查精定位及误报副作用，才考虑MSSL组合。相近或均无实用收益时停止该支线，不自动扩大层数/注意力头或追加门控。单seed小差值不能当作稳定收益或独立创新证据。

所有任务进入有限冻结队列，任何工程门禁失败立即停队列并留日志。没有AP驱动的中途改配置；没有自动MSSL组合或自动调参。
