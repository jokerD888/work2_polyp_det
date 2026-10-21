# 初始化可靠性的最后一次缓存CPU审计

**已完成：四组955800个Query主审计538秒，15707个输入哈希复核通过；报告、GT/Query见证对应、初始化前有效性控制和图形检查均完成。8项验收通过，未加载模型/调用GPU。决策为Go，仅进入机制设计；完整训练不放行。详见`initialization_identifiability_decision_20261021.md`与运行目录`decision.md`。**

用户批准执行。MSSL V1保持冻结，E/F/Q及固定融合搜索结束。本轮不加载模型、不调用GPU、不拟合门控、不扫描融合阈值、不增加推理干预；不访问PolypsSet/PolypDB。

入口`polypbench.analysis.initialization_identifiability`，报告入口`polypbench.analysis.initialization_identifiability_report`。输出`runs/initialization-identifiability-20261021/`。原始审计源码和协议在读取缓存前保存，输入JSON/NPZ先记录SHA、完成后重核。所有已有推理输出保持不变。

## 样本和标签

Baseline/MSSL × PolypGen1473张1533 GT、Kvasir Val120张130 GT；各图全部300 Query。模型之间按GT比较，不对齐相同数字Query ID。

每种干预(Encoder-only、blend05)在IoU .5/.75/.9分别保留四类：两者未覆盖、原生未覆盖而干预覆盖、原生覆盖而干预未覆盖、两者覆盖。分别报告GT池及同Query事件，不把Query改善等价为新增GT检出。GT边际覆盖与已完成干预的一对一覆盖分别保留，本轮重核GT恢复/新增丢失与既有报告一致。

GT辅助归属固定使用Encoder初框、Mask初框、三条件最终框对GT的IoU>=.5兼容并集。只兼容一个GT为唯一归属，兼容多个GT为歧义，无兼容为未归属。初框所指GT不一致和最终条件间所指GT变化单独计数；变化计数与歧义重叠，不能加作独立实例。全部Query保留，不能只留下唯一归属的容易候选。

主子集为all300、原生Encoder排序前10/前1，不按GT或最终分数挑选。额外提供Encoder参考框非边界子集、PolypGen原生合法Encoder子集。GT唯一归属/尺寸子集明确标为事后诊断，不当推理选择规则。

GT恢复与原Encoder最佳Query恢复分别报告；后者使用GT挑Query，只解释轨迹。补充统计真正见证GT池新增覆盖的唯一归属Query，以及重复改善原本已被其他Query覆盖GT的情况；此标签也只用于事后评价。

## 初始化前特征与缺额

固定14项，不挑最好的一项：Box–Mask不重叠程度、log(Mask/Encoder面积)、按Encoder宽高归一化中心偏移、Mask外接框未填充比例、前景占图比例、前景落在padding的比例、空Mask、单行/单列退化、Encoder边界坐标、Encoder面积、Mask面积、原生Encoder选择排名、1减Encoder密集选择分数、Encoder非原生合法标志。

Mask占用分母使用二值Mask整数极值的包含端点面积；空Mask占用留缺失，不填造正常值。全部几何基于实际归一化cxcywh参考框。原生proposal logits可为无穷，但sigmoid参考框有限，数值边界作为特征/分层而非自动回退依据。

PolypGen的密集选择分数/原生合法性从上一轮历史NPZ恢复，并核对历史SHA和Top300 token身份；这是Encoder选择分数，不是最终Decoder或融合分数。Kvasir Val缺这两字段，保留NaN。未缓存完整Mask及空间矩，不计算连通性、碎片或空间协方差，不为补字段调用GPU。低占用只能作为代理，不声称是碎片证据。

## 统计与保护对照

每个统计子集内每张图总权重相等；同时报告Query数、涉及图像数、类别基数、缺失数量。图内Query相关性不能被当成独立样本量。

固定方向高值预测恢复，AUC低于.5也照实报告，不翻分挑最大AUC。分别报告恢复vs损坏、恢复vs原生覆盖、恢复vs全部其他Query的AUC/排序AP/同一有效样本基数。排序AP不是bbox AP，也不是已训练门控AP。

正常保护包括原生已覆盖(损坏+两者覆盖)的特征分布，以及原生>=.9的后续退化计数。各类分位数、正常候选数量、合并GT盲十分位分箱的分布重叠均保留。缺失值只在对应特征统计中排除，基数随之报告，不伪造可用值。

报告另外把.5恢复/实际GT恢复见证与**原生>=.9保护对照**直接比较，明确不同于“.9恢复vs.9损坏”；这两个问题不能混用。仍使用同一组固定特征和已有.5/.9定义，不选操作阈值。

固定首三项几何特征，在前10 Query、IoU .5上做100次seed42按中心分层的图像块重采样，给排序AP区间；不对每个Query做独立bootstrap。按中心、单/多/负图、唯一归属GT面积四分位及其单多交叉报告。四分位沿用各自cohort冻结定义，跨cohort的q1不是同一像素尺度。

## 决策边界

Go仅指是否值得进入机制设计及源域监督核查，不等于门控可部署或允许直接开完整训练。现有Train提供880图941 Box，可在合理GT关联后比较框质量，但缺训练Query特征/分支结果缓存，尚不能验证实际可学习性；835个可靠单实例语义Mask图与多框实例身份缺额仍须区分。

证据必须在不按GT挑Query的子集中成立，并兼顾正常候选基数、真实GT恢复见证、跨中心/模型与源域的方向/重叠。仅在GT唯一归属或少数出错图里出现高AUC，不能通过。若不具备一致证据，给当前No-Go并暂缓路线，不追加系数、阈值或统计搜索。

三条件更换所有Query，会共同影响self-attention；不能离线拼接各分支输出当作真实自适应初始化的结果，也不报告GT Oracle AP上界。PolypGen已参与开发；患者、视频、训练seed稳定性未验证。
