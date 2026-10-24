# 单帧息肉检测对比方法调研与建议（2026-10-24）

本次为论文、官方文档和作者源码的静态调研，没有安装、修改或训练外部模型。
当前L1/L2队列继续使用冻结源码；本文件不改变正在运行的实验。

## 建议清单

建议新增6种外部方法；加上已有Mask DINO、MSSL及通过验证后的最终组合，
主结果表约8–9行。Box-CDF、Mask-static、CIoU、L1/L2另放机制实验表。
优先级表示对论文问题的价值，不承诺某方法在当前数据上更强。

| 优先级 | 方法/版本 | 监督 | 要回答的问题 | 复现条件 |
|---|---|---|---|---|
| P0 | DI-MaskDINO R50，官方4-scale版本 | Box+适配Mask | 同一Mask DINO家族的已有联合任务改进是否更好 | 验证DI/BATO完整性、环境兼容和共同数据协议 |
| P0 | DINO R50，官方4-scale版本 | Box | 成熟纯检测DETR在该任务上的表现 | 记录官方900 queries等结构差异，统一最终评分 |
| P0 | Mask R-CNN R50-FPN | Box+适配Mask | 另一种成熟联合检测/分割架构的表现 | 与Mask DINO共享原始Box及辅助Mask适配规则 |
| P0-领域 | ECC-PolypDet，优先论文已有R50版本 | Box | 领域前景/背景对比与困难样本机制是否有竞争力 | 默认公开代码有忠实性疑点，验收完整机制后才列主表 |
| P1 | D-FINE-M，官方HGNetv2-B2版本 | Box | 较新分布框回归检测器的表现 | 记录真实骨干与预训练来源，不能标为R50同骨干 |
| P1 | YOLO11m | Box | 成熟高效单阶段检测器的实际竞争力 | COCO检测预训练与ImageNet骨干预训练分组报告 |

工程接入顺序建议：先DINO/DI-MaskDINO/Mask R-CNN的配置与数据审计，同时审查
ECC忠实性，再接D-FINE/YOLO。这个顺序不依赖目标域AP；尚未启动这些训练。

## 相对另一份AI清单的调整及核查依据

### 1. DI-MaskDINO应进入核心对照

作者模型针对检测/分割失衡设计DI和BATO，与现有基础模型的关系比另一个通用
实时检测器更直接。官方50ep配置已提供R50、相同torchvision ImageNet初始化入口、
batch16、AdamW 1e-4、1024 LSJ 0.1–2.0等设置，具备建立共同协议的条件。
这不代表已在当前Torch/CUDA上运行成功，也不代表所有模型细节与Mask DINO相同。
[论文](https://proceedings.nips.cc/paper_files/paper/2024/hash/6f1346bac8b02f76a631400e2799b24b-Abstract-Conference.html)、
[官方50ep配置](https://github.com/CQU-ADHRI-Lab/DI-MaskDINO/blob/main/configs/dimaskdino_r50_4scale_bs16_50ep.yaml)。

### 2. 经典模型优先Mask R-CNN，Faster R-CNN作为扩展

选择依据是监督条件：我们训练期本来已有辅助Mask，因此另一种Box+Mask框架
比再增加一个Box-only经典模型更有针对性。两种R-CNN均有成熟官方FPN实现。
若篇幅/算力足够，可再跑Faster R-CNN，形成同一家族的Box-only与Box+Mask参照；
两者仍须分别训练，不能把关闭推理Mask头当作Faster R-CNN训练结果。
[Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md)。

共享的是现有数据适配，不把835张原始单框语义Mask、45张多框图的框内辅助Mask
重新命名为精确实例标注。保留所有提供框，公布Mask监督的可靠性及适配方式。
如果另设“多框实例不计算Mask loss”版本，必须在对比的Mask模型之间执行共同
政策，并另列结果，不能只替某一个模型改变监督。

### 3. ECC重要，但不能默认公开脚本等于完整论文

论文包含BCL、SFFPN、HP及ISR第二阶段；作者同时报告过R50和PVTv2骨干。
论文PolypGen实验是C1–C5训练、C6测试，不能引用其AP与我们的Kvasir训练结果
直接排名。[论文正文](https://arxiv.org/html/2401.04961v1)。

静态源码发现三个具体审计点：

- `model.py`的ECCPolypDet默认`self.fuse='simple'`，实际使用Fusion；FAFPN构造被
  注释。需要核对论文SFFPN实际对应的发布配置/代码，不能只运行默认脚本就
  宣称复现全部模块。[模型实现](https://github.com/yuncheng97/ECC-PolypDet/blob/main/model.py)。
- `data.py`将bbox坐标除4后写入经Resize得到的Mask；模型对比特征又上采样到输入
  分辨率。该坐标链存在需实测的尺度一致性疑点。此处是静态审查发现，尚未完成
  作者代码的实际数据前向验收。[数据实现](https://github.com/yuncheng97/ECC-PolypDet/blob/main/data.py)。
- `train.py`中的BCL基于batch内前景/背景汇聚特征，默认以阈值化F1选权。
  完整比较需要保留跨图关系及第二阶段ISR，并把验证选择和最终评分接入共同
  COCO协议。梯度累积microbatch1不等价于原生batch内BCL；当前同图点对比L1
  不能称作ECC-BCL复现。[训练代码](https://github.com/yuncheng97/ECC-PolypDet/blob/main/train.py)、
  [困难样本阶段](https://github.com/yuncheng97/ECC-PolypDet/blob/main/adaptive_hard_mining.py)。

建议先核对发布实现和论文对应关系。无法消除差异时，只能标为公开实现或明确
说明重实现版本，不以缺模块结果代表论文完整方法，也不为了凑领域方法强行列入。

### 4. D-FINE与DINO都有不能忽略的版本差异

D-FINE-M官方配置采用HGNetv2-B2，并非ResNet50。它作为现代高质量定位竞争方法
有价值，但不能把最终差值解释为只更换一个回归损失。仓库同时支持普通训练和
加载检测预训练微调，不是只能加载COCO/Objects365完整检测器。
[官方配置](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/dfine_hgnetv2_m_coco.yml)、
[骨干预训练设置](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/include/dfine_hgnetv2.yml)、
[训练与微调入口](https://github.com/Peterande/D-FINE)。

DINO官方4-scale配置是900个query；当前Mask DINO是300。保留官方结构并报告
差异，再将最终评价maxDets统一为100，比悄悄改成300更忠实。若另做DINO-300，
必须标明适配，不能把它当成唯一官方版本。
[DINO配置](https://github.com/IDEA-Research/DINO/blob/main/config/DINO/DINO_4scale.py)。

### 5. 补查候选与暂不优先项目

| 候选 | 建议 | 原因/证据 |
|---|---|---|
| Faster R-CNN R50-FPN | 预算允许时补充 | 补齐经典Box-only参照；与Mask R-CNN共享工程框架 |
| RT-DETRv2 R50-vd | 备选扩展 | 有官方实现；DINO和D-FINE已覆盖Transformer，优先级可后置。R50-vd也不应写成与torchvision R50完全相同。[官方仓库](https://github.com/lyuwenyu/RT-DETR) |
| YOLO-OB | 领域备选，可在ECC无法忠实复现时优先审计 | 作者公开BiSPFPN/ObjectBox实现。训练脚本验证调用使用IoU阈值0.3，不能直接拿默认mAP做本项目AP50:95。发布的SUN权重也不能混入Kvasir-only训练组。[代码](https://github.com/seanyan62/YOLO-OB)、[训练入口](https://github.com/seanyan62/YOLO-OB/blob/master/train.py) |
| YOLO26m | 较新版本扩展或预先替换YOLO11m | 官方确认2026年发布；YOLO11不能称为当前最新。没有必要因版本堆数量；若论文强调最新实时竞争力，应预先选择加入一个。[官方文档](https://docs.ultralytics.com/models/yolo26) |
| DEIM / DEIMv2 | 追求较新强基准时扩展 | DEIM为CVPR2025；DEIMv2的S/M/L/X使用DINOv3相关特征，额外预训练信息需单列。[DEIM](https://github.com/Intellindust-AI-Lab/DEIM)、[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) |
| DADA | 相关工作或数据增强对照 | 作者MICCAI2025工作通过扩散模型合成困难负样本，是训练数据生成路线，新增生成模型先验和流程，不是本轮优先的完整检测架构基线。[作者代码](https://github.com/Huster-Hq/DADA) |
| PolypSegTrack | 不列当前核心表 | 论文任务含视频分析及多任务基础模型，需另外确认单帧检测设置与预训练边界。[论文](https://arxiv.org/abs/2503.24108) |
| PraNet/U-Net等语义分割模型 | 不列核心bbox检测表 | 转框还需实例分离和框分数规则，不能把Dice或mask转框成绩直接当作标准实例检测对照 |

这份清单是代表性方法比较，不能据此宣称覆盖截至2026年的全部SOTA。

## 方法机制的证据另组织

外部模型排名回答任务竞争力，以下实验回答新增机制的必要性：

- 已有：Mask DINO、MSSL、Box-CDF、Mask-static、CIoU-Replace。
- 正在执行：MSSL+Box局部对比（L1）、MSSL+Mask优先局部对比（L2）。
- 若局部机制有效：补完整2×2，即无MSSL/无局部机制、仅MSSL、仅局部机制、
  二者组合；再按明确主张补全图/局部背景、边界处理等必要消融。
- 如需声称优于ECC的BCL机制，再定义忠实组件移植对照；当前L1只能称普通Box
  局部对比。若不能保留跨图正负关系，不沿用作者模块名称。

MbIoU确为2024年已发表的Mask相关框回归工作，应纳入近邻方法查新。
本次核实了期刊页面和摘要；全文获取失败，尚未完成公式、可微性及实际训练
实现核查，因此不列为已可复现/必跑对照。[期刊页面](https://dergipark.org.tr/en/pub/tjmcs/article/1423292)。
CFIoU的foreground-area表述也不能仅凭名称理解成使用GT像素Mask。
[原论文](https://www.mdpi.com/1424-8220/23/10/4961)。

本项目已有MSSL对Box-CDF总体AP差值的图像重采样区间跨零，需继续保留这一
限制。赢过多个外部检测器也不能替代精细Mask支持的独立证据。

## 比较协议建议

1. 使用固定880/120划分和941/130个提供框；不从Mask重算Box。实际MSSL基线为
   1024 LSJ、50等效epoch，不能套用历史`experiment_protocol.md`的512/100epoch。
2. 同框架机制对照固定初始化、日程、增强、匹配、数据、有效batch与microbatch。
   跨架构比较允许合理官方配方，但须预先登记尺寸、训练曝光、更新次数、验证
   候选数量、调参预算和算力；不足收敛的同预算结果要明确其范围。
3. 表中明确Box/适配Mask监督，以及ImageNet骨干、COCO检测、Objects365、
   DINOv3等预训练来源。YOLO从随机初始化训练不等于ImageNet预训练条件对齐。
   如用COCO权重，属于额外预训练的实用基线，不能写成严格同训练信息。
4. 所有模型按Kvasir Val bbox AP50:95选权。验证频率/候选次数尽量一致；使用
   EMA时预先指定raw/EMA选择政策，不能只给某方法增加挑权机会。
5. 统一原图坐标、类别、COCO bbox evaluator、maxDets100和完整图像队列。
   评分前不要沿用ECC的0.3高置信度截断或YOLO-OB的IoU0.3自带指标。
   保留各模型必要的原生后处理，NMS与低分导出阈值在源域固定；公平不要求
   给Mask DINO加NMS或把YOLO/Faster R-CNN的NMS全关掉。
6. PolypGen固定1473图、1533 GT及126张负图，保留已观察外部开发集身份。
   报AP50:95/AP50/75/90/95、分中心/尺寸/单多目标、漏检误报及模型开销。
   不使用PolypGen选择模型尺寸、学习率、后处理或训练期限。
7. 至少对原生Mask DINO、MSSL、最终组合及关键强竞争方法安排预先固定的多Seed。
   图像Bootstrap补充测试样本不确定性，不能替代训练Seed或患者级独立性。
8. 方法最终冻结后再使用保留的PolypsSet/PolypDB作外部确认。此处尚未启动外部
   模型训练，也未因本次调研改变正在运行的L1/L2。

## 自审与待办边界

| 问题 | 本次结论 |
|---|---|
| 是否覆盖同架构、联合监督、领域方法和较新强检测器？ | 是；ECC仍需忠实性门禁 |
| 是否把公开代码误当成已运行成功？ | 否；本次仅静态审查，尚无外部模型复现成绩 |
| 是否能够将外部排名归因到MSSL？ | 不能；因果归因依赖共同框架对照 |
| 是否已有多Seed和未观察外部集证据？ | 尚无；须后续补充 |
| 是否需要当前就追加全部实验？ | 不需要；先锁版本、初始化、数据和评价适配，再冻结有限训练队列 |
