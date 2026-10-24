# H1/H2 第一层额外特征读取执行记录

**完成更新：2026-10-24 04:32（北京时间），r2全部21项任务完成。H1/H2各1473张PolypGen推理验收通过，防自动睡眠已释放。H1：Kvasir Val AP65.203、PolypGen AP48.849（相对Baseline +0.031）；H2：Val64.444、PolypGen48.236（相对Baseline -0.582、相对H1 -0.613）。H2没有显示第二空间参照的独立收益，按预定停止规则，不启动MSSL组合、不扩大层数或注意力头。该结论针对当前同预算单Seed实现，不证明所有Encoder补充特征方案无效。保留MSSL V1（PolypGen49.751）。完整结果见`runs/reference-features-full-20261023-r2/summary.md`和`comparison.json`。以下保留验收与启动时记录。**

用户授权：先规格/查新和工程验收，通过后只做H1/H2两组独立完整训练；不直接组合MSSL。规格见[reference_features_spec_20261023.md](reference_features_spec_20261023.md)。

## 工程验收已通过

目录：`runs/reference-features-acceptance-20261023-r5/`。只使用已有Baseline权重及Kvasir Train固定分层图像，0次optimizer更新，无AP评价，无PolypGen图像/GT输入。

- H1/H2合计8组FP32推理：未挂载/禁用/零残差的捕获预测、各阶段框、分类分数、原生Mask质量融合分数逐位相同。
- 16个单图AMP训练视图：捕获输出、原生损失一致。2组双图DN批次：预测与参考逐位一致，Mask总Loss归约允许`rtol=2×float32 eps, atol=0`，另记录原生自重复差异；没有放宽预测一致性。
- 初始预测aux0、Encoder框损失对额外分支无直接梯度；第一Decoder输出aux1对W有非零有限梯度；第一层DN输出没有新增分支直接梯度。固定非零W探针后额外采样器获得非零有限梯度。
- H1使用原生第一层Mask参考；H2只把同一Encoder参考按native valid_ratios送入额外采样。DN前缀与初始参考不改。原生权重/缓冲未变，构造不改变CPU/CUDA随机状态。
- 固定源域图像、FP32、5次预热后测量：H1禁用/启用均值107.16/108.20ms，H2为104.01/108.51ms；峰值分配显存约2024MiB。这个小样本工程测量不能代替训练后正式速度基准。

首次验收暴露的工具问题均保留：r1读取历史配置缺少后加的LOSS_FORM默认项；r2的独立加载器未指定dataset_name，COCO类别1未映射到连续类别0；r3/r4因双图总Mask Loss的一次FP32舍入差异（2.384e-7）触发过严字典比较。r5恢复完整默认配置和原生类别映射，原始预测保持逐位检查，对标量CUDA归约单独记录机器精度差异。未因此修改模型或训练超参数，未使用AP调参。

## 有限冻结队列

当前目录：`runs/reference-features-full-20261023-r2/`；控制器PID2149，Windows隐藏防自动睡眠PID25152。冻结源码基于commit`00fa14d`，协议另记录每个源码文件SHA。

首轮`runs/reference-features-full-20261023/`在原生回归测试因PYTHONPATH缺少原生MaskDINO路径而停止，未开始GPU优化；修复启动环境并新建r2，原始失败日志保留。r2中27项新机制/既有支持模块检查全部通过，MSSL原生回归15项中14项通过、1项因历史相对路径资产不在冻结目录而按原测试规则跳过。已开始native冷启动Smoke。

有限任务顺序共21项：

1. 27项单元检查及MSSL回归。
2. native/H1/H2分别最多40次尝试，要求20次成功更新；无AP评价、Smoke权重禁止用于完整训练。
3. 核对三组共同native参数SHA、图像及Box增强序列前缀、H1/H2相同参数数目与真实梯度；严格model/optimizer/scheduler/scaler重载。
4. 两张源域图像的collector预测一致性、各阶段300框/Top100导出预检；两组正式DefaultTrainer起点和optimizer发现门禁。
5. H1/H2各50等效Epoch完整训练、Kvasir Val预测及正式验收。
6. 两组全部结束后，对固定1473张PolypGen依次推理/评分，再输出完整AP/覆盖净得失/FP比较。

| 条件 | 完整训练run_id |
|---|---|
| H1 | features-mask_extra-s42-20261023T040838Z |
| H2 | features-encoder_extra-s42-20261023T040838Z |

两组均Seed42、2750计划迭代、microbatch1×累积16、原始ImageNet R50起点、原生1024 LSJ、学习率/分类/Mask/框损失与匹配均不改。原Kvasir Val选点规则保留，PolypGen仍为主要外部开发基准。PolypsSet/PolypDB保留。

任务通过门禁后自动接续，失败即停止；不根据中途AP改配置，不自动追加MSSL组合、门控、层数/注意力头搜索。**当前队列启动不表示冷启动Smoke、长训练或AP已通过。** 实时状态以该目录state.json/queue.json为准。
