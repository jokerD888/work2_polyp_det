# 交给训练机 AI 的执行 Prompt：Kvasir 单帧息肉检测对比实验

请直接执行下面任务，完成真实代码、配置和工程验收，不要只给建议或计划。

## 0. 本次任务与执行开关

我们正在研究单帧息肉框检测：Mask DINO 为基础，MSSL 是训练期利用 GT Mask 辅助框定位的方法。第二机制正在别处实验，本任务负责外部比较方法的工程实现和首轮统一预算训练准备。

```yaml
execution:
  implement_adapters: true
  run_engineering_acceptance: true
  run_cold_smoke: true
  run_full_training_after_gates: false
  run_polypgen_evaluation: false
  run_extra_seeds: false
  run_hyperparameter_search: false
```

`run_full_training_after_gates=false` 时，完成代码、验收、Smoke、冻结正式配置及可一条命令启动的有限队列；不要启动六组长训。如果我交给你的版本将该开关改成 `true`，则验收通过后自动执行已冻结的首轮长训，不必再次确认。

无论开关如何，都不自动追加模型版本、备用方法、超参搜索、更多种子或外部数据评估。发现局部方法被阻断时记录证据并继续其他独立任务，不要用简化模型冒充原方法。

先检查当前仓库、AGENTS.md、数据、CUDA环境、GPU数量/显存和已有运行任务。不要假设机器一定有两张A6000，不要停止现有进程或覆盖已有产物。各方法使用隔离环境；不要为兼容旧仓库而降级正在运行实验的公共环境。

代码同步补充：本任务所需研究代码在 `codex/candidate-stage-diagnostics` 分支，不能只拉默认 `main`。仓库为 `https://gitee.com/JokerMao/work2_polyp_det.git`，GitHub镜像为 `https://github.com/jokerD888/work2_polyp_det.git`。取得该分支后应能看到 `methods/mssl/`、`repro/mssl_v1_20261020/` 及第2节所列冻结文档。已经确认默认main缺少这些目录，不能据此判断研究源码未提供。若训练机当前工作区有修改，使用新checkout/worktree隔离接入，不覆盖已有工程。

## 1. 待实现方法

| ID | 方法 | 版本要求 | 允许的训练监督 |
|---|---|---|---|
| C1 | DI-MaskDINO | 官方R50、4-scale，完整DI+BATO | GT Box＋共同适配的辅助Mask |
| C2 | DINO | 官方R50、4-scale | GT Box |
| C3 | Mask R-CNN | Detectron2 R50-FPN | GT Box＋共同适配的辅助Mask |
| C4 | ECC-PolypDet | 优先论文已有的ResNet50版本；完整机制须先核查 | GT Box；BCL区域由Box生成 |
| C5 | D-FINE-M | 官方HGNetv2-B2版本 | GT Box |
| C6 | YOLO11m | 官方检测版，不是seg版 | GT Box |

作者/官方仓库：

- DI-MaskDINO：https://github.com/CQU-ADHRI-Lab/DI-MaskDINO
- DINO：https://github.com/IDEA-Research/DINO
- Mask R-CNN：https://github.com/facebookresearch/detectron2
- ECC-PolypDet：https://github.com/yuncheng97/ECC-PolypDet
- D-FINE：https://github.com/Peterande/D-FINE
- YOLO11：https://github.com/ultralytics/ultralytics

获取代码后记录真实commit、dirty diff、依赖版本和全部本地兼容补丁。不能仅写“latest”。优先作者代码；不得替换成名称类似的第三方重写而不说明。

本次不训练Faster R-CNN、RT-DETRv2、YOLO-OB、YOLO26、DEIM、分割转框模型，也不把MSSL或正在实验的局部对比加到这些外部模型上。

## 2. 当前研究基线：必须按这个版本理解

现有基线是官方Mask DINO R50：

```yaml
reference_maskdino:
  upstream_commit: 3831d8514a3728535ace8d4ecc7d28044c42dd14
  upstream_config: configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml
  num_classes: 1
  num_queries: 300
  decoder_layers: 9
  initialization: detectron2://ImageNetPretrained/torchvision/R-50.pkl
  initialization_file_sha256: 16fd03f0b0123e85b959cb9b64aaf9eeb4a7dc68bd9b0b134a9220d4b24c15c6
  seed: 42
  train_images: 880
  val_images: 120
  train_gt_boxes: 941
  val_gt_boxes: 130
  nominal_epochs: 50
  effective_batch: 16
  micro_batch_per_gpu: 1
  num_gpus: 1
  gradient_accumulation: 16
  planned_optimizer_iterations: 2750
  optimizer: AdamW
  base_lr: 0.0001
  backbone_lr_multiplier: 0.1
  weight_decay: 0.05
  lr_scheduler: WarmupMultiStepLR
  lr_steps: [2444, 2648]
  lr_gamma: 0.1
  warmup_factor: 1.0
  warmup_iterations: 10
  full_model_gradient_clip_l2: 0.01
  amp: true
  train_input_format: RGB
  train_image_size: 1024
  lsj_scale: [0.1, 2.0]
  horizontal_flip_probability: 0.5
  eval_short_edge: 800
  eval_max_long_edge: 1333
  eval_period_iterations: 275
  checkpoint_period_iterations: 275
  checkpoint_keep_regular: 2
  checkpoint_selection: maximum_Kvasir_Val_bbox_AP50_95
  checkpoint_tie_break: earliest
  detections_per_image: 100
  tta: false
  dataloader_workers: 2
```

训练几何为原生水平翻转、ResizeScale、1024 FixedSizeCrop/pad；原生图像padding值保留，二值Mask的padding必须为0。原生Mask DINO的分类、Mask、L1/GIoU、匹配、DN、Mask2Box初始化和Mask参与的原生推理打分均保留。

上面是**现有Mask DINO基线的完整参照条件**，不是要求所有架构使用相同优化器、梯度裁剪和损失系数。

现有MSSL V1只额外监督最终Decoder匹配框，权重1.0；它不是这六个外部方法的组成部分。不要因复用 `methods.mssl` 的数据/运行工具而误启用MSSL、local_contrast、quality_alignment或历史改进模块。

如仓库有以下文件，优先读取实际冻结证据：

- `docs/mssl_v1_release_20261020.md`
- `repro/mssl_v1_20261020/`
- `methods/mssl/configs/kvasir_official50.yaml`
- `methods/mssl/data.py`
- `polypbench/adapters/bitmask_mapper.py`
- `polypbench/adapters/instance_lsj.py`
- `polypbench/evaluation.py`
- `docs/comparison_methods_survey_20261024.md`

历史 `docs/experiment_protocol.md` 中512输入、batch4、100epoch设置不是当前MSSL基线，不能误用。

代码取得后的复用要求：C1优先直接复用 `methods/mssl/accumulation.py` 的 `AccumulatingAMPTrainer`，以及现有共享数据适配，不重新凭文字猜一套累积实现。导入这个训练工具不等于启用MSSL；C1仍使用作者DI-MaskDINO模型/criterion，不经MSSL模型配置入口，不附加 `loss_mssl` 或局部对比Loss。当前训练器相对原始归档增加了一些诊断记录，历史精确版本位于 `repro/mssl_v1_20261020/sources/initial.zip`；如需提取比对，放入独立目录，不能覆盖工作树。先执行 `python scripts/archive_mssl_v1.py verify` 验证归档，再核对数值语义。硬件/框架兼容适配可以做，但必须保存diff并跑累积回归；仅在取得研究分支后仍存在实际接口障碍时，才说明原因并实现等价适配，不能把独立重写称为逐字复用。

## 3. 数据协议：所有方法共享，禁止悄悄改标签

### 3.1 数据定位与校验

先在训练机定位项目根、Kvasir原始数据和转换数据，写入独立路径配置，不照搬本机Windows盘符或WSL路径。

现有转换目录结构：

```text
data/kvasir_official_bbox_coco/
  train.json
  val.json
  train/                 # 880张图
  val/                   # 120张图
  conversion.json
  <转换时归档的原始bbox JSON副本>
```

另外需要Kvasir-SEG原始语义Mask目录。没有原始Mask时，Box-only方法可以继续准备，C1/C3需要补齐数据；不得用裁剪Mask冒充原始单框语义Mask。

冻结的COCO标注文件SHA256：

```yaml
train_json_sha256: 9b5d73d9b3206daa78ce57d15dbe9fe2dd75b432176419762475a4318ec727b6
val_json_sha256: db96be0dca4e0a4cebb308f096a9f67eaf5a6349a0fabfe01f5107dc43aed929
```

若迁移文件字节不同，不能直接跳过校验：先比较全部图像ID/文件名、官方划分顺序、类别、图像尺寸、逐实例Box坐标、数量和RLE语义，说明差异并保存新的本地文件SHA及语义等价证据。若划分、Box或Mask语义改变，暂停依赖该输入的任务。

只使用官方880/120划分：train941框、val130框，总计1071框。类别只有polyp。训练内部可以用连续类别0，COCO导出必须还原JSON中的类别ID1。不得重新随机划分或合并Val训练。

为全部图像、原始Mask、split文件、原始bbox JSON及转换标注记录SHA和manifest；训练完成后复核。没有提供的文件如无法从已有证据恢复，明确记录缺失，禁止伪造身份校验。

### 3.2 Box协议

Box来自原始 `kavsir_bboxes.json`，只将xyxy转换为xywh：

```text
[xmin, ymin, xmax-xmin, ymax-ymin]
```

不增加VOC的±1偏移，不从Mask或连通域重新生成Box，不按目标大小删除原始GT，不把多框图改成单框图。原图COCO area按Box宽×高。

增强后保留各方法合理的原生退化框过滤，但记录过滤规则和数量；Mask DINO同家族须复用现有规则。不能按“Mask是否非空”擅自删掉仍有效的提供框。

YOLO格式转换必须通过浮点坐标往返测试，验证归一化xywh回到原图后与COCO Box一致，不能经整数舍入丢失小框。

### 3.3 使用辅助Mask的模型

训练集835张原始单框图、45张原始多框图，后者共有106个GT框。

- 原始单框图：使用原始语义Mask，灰度`>127`，不裁到GT Box内；这是现有Mask DINO控制组和MSSL共同的数据规则。
- 原始多框图：保留现有转换JSON中的框对应辅助Mask，不强拆语义Mask生成伪造的精确实例真值。
- “原始单/多框”身份必须在增强前确定；裁剪后只剩一个框，不会改变原始多框身份。
- 原生图像、Box、Mask必须使用同一个几何变换；二值Mask padding=0，正确维护padding/valid区域。
- Box不能用增强后的Mask重新计算。不要直接用原生mapper默认行为覆盖这项约定。
- 不把适配辅助Mask描述成独立人工实例标注。

C1/C3需要使用同一标签适配政策。若某方法对现有辅助Mask政策无法正常训练，先报告实际问题，不能只对它删除45张图、关闭多框Mask损失或改变Box；替代政策须另列配置并说明与现有基线的差异。

C2/C4/C5/C6只允许Box监督。不能因原始数据带Mask就额外给它们加入精细GT Mask监督。ECC由矩形框构造BCL区域属于Box监督，不属于精细Mask监督。

## 4. 首轮预算与公平性：明确哪些相同、哪些允许不同

本次准备的是**Kvasir相同划分、首轮50等效epoch的预算对照**。它不保证每个架构均达到最佳收敛，不得把这种有限预算结果宣传成各方法全部潜力或普遍优劣。

### 4.1 同家族严格参照

C1 DI-MaskDINO采用官方R50/4-scale/50ep架构配置，完整保留DI/BATO，Kvasir适配严格沿用第2节：1024 LSJ、有效batch16、2750计划迭代、2444/2648衰减、microbatch1×累积16，以及相同原始ImageNet R50资产。

不要因为训练机显存充足，就将这个严格对照改成物理batch16后直接与历史microbatch1结果混用。梯度累积不等于物理batch，尤其涉及DN、GT归一化和batch内关系时。

具体累积语义：每个计划optimizer迭代只zero_grad一次，连续处理16个microbatch；每个microbatch保留原生GT归一化和DN构造，weighted总Loss先除16再AMP backward；累积结束后统一unscale/全模型裁剪及一次optimizer/scaler处理。scheduler按现有实现的计划外层迭代推进，AMP跳步单独记录，不把2750计划迭代改成2750次成功更新。历史Seed42模型为2750计划迭代、2740次成功更新；其他硬件不要求跳步次数相同，也不能为凑同样跳步人为调整数值。

如必须改变这项训练语义，应另列配置，并在相同新政策下安排原生Mask DINO和MSSL桥接对照；当前默认队列不得静默加入这些额外完整训练。

### 4.2 跨架构预算适配

C2/C3/C5/C6首轮以50个数据集等效epoch为预算参照。有效batch16是默认目标；若总batch16且每epoch880张，则每epoch55次更新，共2750次计划更新、44,000次基础图像曝光。

- 各架构保留官方优化器类型、损失、归一化、匹配和核心结构，不把Mask DINO的AdamW/clip0.01硬套给Mask R-CNN或YOLO。
- 输入尺寸和增强按合理的作者配方适配：优先保持官方核心机制，正式前在协议中列明与基线1024 LSJ/800–1333推理的差异。不能称为相同分辨率、相同增强或严格单变量对照。
- 基础图像曝光与Mosaic/MixUp等带来的额外源图像使用分别记录。不要只凭“都50epochs”宣称训练信息/计算完全相同。
- 尽量使用有效batch16；若物理batch、累积或全局batch不同，必须重新计算更新预算与LR阶段，单列实际值。不要让框架自动按GPU数缩放batch/LR而未记录。
- 作者不同训练日程映射到50epoch时，保持主要阶段的相对位置。写明原预算、目标预算、映射公式及最终整数milestone；特别检查warmup、关闭强增强阶段、EMA warmup/restart等，不能只改max_epoch导致核心阶段从不触发。
- 物理batch对BN、ECC-BCL等有影响；不能用累积声称复现了跨图统计/关系。
- 每个模型只固定1个首轮正式配置、Seed42；不按PolypGen或中途AP追加搜索。
- 不同GPU/框架上的增强随机序列无需逐位相同；必须分别可复现。只有同路径无操作回归才要求严格输出/随机状态一致。

C4 ECC的两个训练阶段合计计入50epoch预算；不能第一阶段50epoch后再悄悄追加第二个50epoch。先从论文/官方配置确定原有阶段比例并映射，保留ISR机制，记录额外挖掘推理成本。若预算映射无法保留方法核心，标为协议待解决，不以删掉第二阶段的模型代表完整ECC。

完成后报告收敛曲线及预算限制。充分收敛的官方配方扩展、多Seed和调参留待后续；当前不自动启动第二套更长预算。

## 5. 预训练协议：先审计，再固定唯一入口

主任务只用Kvasir Train进行任务训练，不使用目标外测数据训练。

- C1/C2/C3/C4：优先ImageNet分类预训练骨干，检测头从头初始化；核对具体资产、数据来源和兼容的R50变体。
- C1优先复用第2节同一torchvision R50资产。不要混用MSRA/Caffe2、ResNet-vd或其他变体后称为完全相同初始化。
- C5：使用官方HGNetv2-B2分类预训练骨干入口，核实实际预训练来源。它不是R50。不能未经登记就用Objects365/COCO完整检测器权重。
- C6：默认选择官方YOLO11m COCO检测预训练作为实用基线，并明确单列“额外COCO检测预训练”组。不要强行把随机初始化YOLO称为ImageNet预训练公平组；也不要未经验证用分类权重部分加载冒充等价初始化。

如果某个指定权重入口不存在、来源无法确认或加载覆盖率异常，先报告该方法的实际阻断证据。不得静默切换初始化条件。严禁使用已在Kvasir/PolypGen等相关息肉任务上训练的未知来源权重。

每个权重必须保存URL、本地文件SHA256、预训练数据来源、加载/缺失/意外键清单及覆盖情况。不能只根据文件名或README猜来源。

### 5.1 训练机无法直连Hugging Face/Google Drive时的处理

先使用以下已经从官方源码核实的资产入口，不要先假定权重只在Hugging Face或Google Drive：

- C5 HGNetv2-B2：`https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth`。D-FINE官方 `src/nn/backbone/hgnetv2.py` 的B2配置直接引用此GitHub Release资产；Release元数据确认文件名和大小24,362,501字节。本地路径按实际冻结版本的加载器要求配置。该大小不能替代SHA/完整性验证。
- C4 ECC-R50：作者 `resnet.py` 加载 `resnet50-11ad3fa6.pth`，对应TorchVision `ResNet50_Weights.IMAGENET1K_V2`，官方入口为 `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`。作者硬编码的 `./snapshot/` 与README目录有差异，应改成显式本地路径配置，不改权重数值。注意该资产不等于第2节Mask DINO的转换pkl；两者不能未经验证就声称相同初始化。

训练机已确认github.com、PyPI、dl.fbaipublicfiles.com正常，但GitHub Release可能重定向至其他域名，download.pytorch.org也尚需实测；必须确认真实文件下载成功，不能用网页访问成功代替。下载失败保留错误，不关闭TLS验证，不把HTML错误页当checkpoint。

允许的后备顺序：

1. 作者/上游维护的另一个官方链接，明确为同一资产；记录原始地址、实际地址、版本和SHA。
2. 在可联网机器下载官方原文件，经用户已有的文件传输渠道离线复制到训练机；发送端和接收端SHA256一致，再做架构/加载检查。不要在Git仓库中提交大权重或凭据。
3. 如仅文件封装/键名不兼容，可做有依据的格式转换，保存脚本、源/目标SHA，逐张量验证数值未变，并验证模型结构、BN、stride及预处理兼容；不能把另一种骨干或另一套训练权重称为格式转换。

不允许静默更换为随机初始化、COCO/Objects365检测权重、息肉任务权重或未经验证的第三方镜像，也不把Detectron2 R50权重直接塞给HGNetv2。来源/等价性无法确认时，只将该方法标为等待资产，继续其他独立方法。以上下载和离线复制政策已经是本任务的处理约定，不需就每个可验证的官方替代下载地址反复询问。

## 6. 六个方法的专门验收

### C1：DI-MaskDINO

官方参考：`configs/dimaskdino_r50_4scale_bs16_50ep.yaml`。

核对DI、BATO真实参与训练与推理；新增参数进入optimizer，损失能回传。兼容Torch/CUDA的补丁不能改变其数学定义。保持共同Box/Mask适配、bbox-only评分和作者原生推理路径。

### C2：DINO

官方参考：`config/DINO/DINO_4scale.py`，原生900 queries。保留900，不擅自改成Mask DINO的300；最终评测统一maxDets100即可。保留原生DN、匹配和框回归。将900/300和Decoder等差异写入表格。

### C3：Mask R-CNN

使用真实Mask R-CNN R50-FPN，验证ROI采样、Mask target裁切和共同辅助Mask标签兼容，Box仍来自提供框。推理保留原生Box分支与NMS，不从预测Mask取外接框替代Box head。

### C4：ECC-PolypDet——先忠实性审计，再决定是否进入正式队列

论文：https://arxiv.org/html/2401.04961v1

已有静态审查发现，需要你实际核对：

1. `model.py`默认 `self.fuse='simple'`，FAFPN构造被注释，默认并不直接启用该分支。确认论文SFFPN对应的实际发布实现和配置。
2. `data.py`将bbox除4后写入Mask，而模型将对比特征上采样到输入分辨率。用真实样本可视化和合成已知框检查坐标一致性，不能只看loss有限就通过。
3. BCL是batch内前景/背景汇聚关系。记录真实参与对比的物理batch及DDP跨卡关系；microbatch1×累积16不等价。不得替换成当前L1那种同图点对比。
4. 保留HP与ISR困难样本挖掘、第二阶段训练。ISR只能使用Kvasir Train及其GT，不读Val/PolypGen GT生成训练权重。
5. 默认验证F1和较高置信度截断不能直接沿用为共同选权/评分方式；接入第7节的Val bbox AP选权与低阈值预测导出。

先查作者发布配置/历史commit/论文说明，再写清必要补丁的依据。证据不足的结构重建不要伪装成作者完整实现，也不要在本轮进行自创机制补全。C4未过忠实性门禁时，将它标为待解决并继续其他5种方法，不自动改跑备选模型。

### C5：D-FINE-M

官方参考：`configs/dfine/dfine_hgnetv2_m_coco.yml`。

确认实际骨干为HGNetv2-B2，FDR及GO-LSD等作者定义保留。首轮50epoch适配需特别核对原配置中132epoch/后期阶段与EMA政策的映射；不要因阶段索引错误而关闭核心训练机制。

### C6：YOLO11m

使用官方检测架构和1类输出。检查COCO预训练头替换、坐标/类别转换、letterbox逆变换、batch/累积、EMA和原生NMS。允许其声明的原生输入/增强配方，但正式前必须冻结并记录差异。

选择Val权重必须使用bbox AP50:95，不使用默认综合fitness替代。原生框架可额外记录其他指标，但不能直接将其默认Val mAP数值当作共同评测器结果。

## 7. 统一验证、选权与推理导出

统一使用项目的 `polypbench/evaluation.py` 或经严格回归验证等价的pycocotools COCO bbox evaluator：

```yaml
evaluation:
  task: bbox
  iou_thresholds: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
  max_dets: [1, 10, 100]
  include_all_gt_images: true
  validation: fixed_Kvasir_Val_120_images
  primary_metric: AP50_95
  report: [AP50_95, AP50, AP75, AP90, AP95, AR100]
  tta: false
```

- 50epoch单阶段模型在等效epoch5/10/15/20/25/30/35/40/45/50进行10次候选验证；batch16时对应275/550/.../2750次更新。每次完整评估120张图；正式之前不要用AP调配置。
- ECC验证也按整个两阶段累计预算的相同节点安排；最终模型从第二阶段可选节点中选取，第一阶段checkpoint只用于既定阶段衔接，不靠增加额外验证机会选最终模型。将其候选范围差异明确报告。
- 按最大Val bbox AP50:95选权，同分取最早。保留best、final和最近2个普通checkpoint。
- 原始权重/EMA二选一：保留方法声明的最终模型政策，正式前固定；不允许同一方法把raw/EMA都作为额外候选择优。两阶段衔接/最终候选范围也必须事先写明。
- 保存COCO预测：`image_id/category_id/bbox/score`；bbox为原图浮点xywh，类别ID恢复为1。
- 无预测图像仍参与评分，不能只评估有预测的图。不得读取GT决定预测数、坐标、置信度、NMS或过滤条件。
- 保留每个方法必要的原生后处理，NMS阈值与低分导出阈值在正式前从作者设置/源域工程依据固定。不要求所有模型禁用NMS，也不额外给Mask DINO加NMS。
- 不沿用ECC的0.3高分截断做AP导出。无低分过滤的模型不新增过滤；如框架必须设置阈值，使用预先声明的低阈值并记录截断影响，不能按目标域调它。
- 不能从预测Mask提框替代原生框预测，也不能在推理时提供GT Box/Mask。
- AP90/AP95从COCO precision张量正确提取；JSON使用0–1，汇总表使用0–100，明确单位。

历史源域Val AP约为原生Mask DINO66.072、MSSL66.273。这些只作为已有记录，不是外部方法必须达到的门槛，也不要求不同硬件重训逐位复现AP。

## 8. 工程验收与冷启动Smoke

每个方法通过以下门禁后，才允许进入正式队列。

### A. 数据与评价CPU验收

- 全量验证880/120图、941/130框、split互斥、类别、原图尺寸、SHA和逐框坐标。
- 检查原始多框、较小目标、增强后空Mask/被裁切目标、padding，以及Box不等于Mask外接框的样本。
- Box格式/YOLO归一化坐标往返；resize/letterbox/crop逆坐标检查；RLE形状、孔洞和零padding检查。
- 使用完美预测、重复预测、空预测及含负图的合成数据验证统一evaluator和AP单位。完美预测应接近1，空预测应正确处理，不得漏图或崩溃。

### B. 真实GPU训练路径验收

- 使用Kvasir Train固定的小型面板，不用Val/PolypGen挑样调机制。
- forward/backward有限；关键作者模块确实启用并获得合理梯度；所有可训练参数被optimizer发现且仅一次。
- 检查AMP、loss归一化、累积除数、clip和scheduler实际触发频率。
- 无操作数据/评价适配应尽可能保持原模型行为；严格张量一致性用于真正相同执行路径，不对不同模型或不同物理batch强求一致。
- 图像独立推理不接收训练GT或辅助Mask字段；checkpoint严格加载。

### C. 冷启动Smoke

- 从该方法正式指定的初始权重冷启动，使用实际正式训练入口、数据loader、优化器和累积政策。
- 每个方法达到20次成功optimizer更新；AMP方法最多40次尝试。记录跳步、loss scale、真实成功更新、loss各分项、梯度和峰值显存。
- 若非AMP方法同样执行20次有效更新；不能为了统一形式强制不支持的AMP路径。
- 严格检查model、optimizer、scheduler、scaler以及适用时EMA的保存/重载；重载后固定图像预测在声明精度内一致。
- Smoke产物标为测试专用，不允许作为正式长训热启动。
- ECC还须验证ISR权重生成和第二阶段实际入口，以及必要状态衔接；不能只验收第一阶段。

不根据Smoke的AP挑方法/超参。数值失败先定位实现、数据或兼容问题，保留失败证据；修复后用新验收ID。不能通过放宽阈值、删掉难样本或跳过核心模块来“通过”。

## 9. 冻结正式队列与故障处理

验收通过后，逐方法冻结源码副本、上游身份、配置、补丁、环境、初始资产、数据manifest和评价脚本SHA。运行不依赖持续修改的工作区源码。禁止覆盖既有run-id。

正式队列顺序默认C1→C2→C3→C4→C5→C6；C4未过门禁时跳过并显式记录待解决，不以其他模型顶替。若硬件允许，可并行独立进程，但同一GPU同时负载会影响时延测量，正式测速须独占且记录环境。

不要依据训练中途AP调整分辨率、损失权重、日程、增强或预训练。工程错误影响某方法时停止该方法，保留日志，其他独立方法可以继续；共享数据/evaluator错误则暂停全部依赖任务。

支持安全中断恢复：恢复model/optimizer/scheduler/scaler/EMA，以及可恢复的RNG、sampler/累积状态；不重复计算已完成更新，不重置学习率。若无法精确恢复，明确标记为restart的新run，不能把不精确恢复混成连续训练。

## 10. 交付物

请交付实际可执行文件，而不是只有实验建议：

1. 六种方法的adapter、正式配置、数据转换/校验工具和统一预测/评分入口；受阻方法明确列出已完成部分及证据。
2. `protocol.md` / `protocol.json`：逐方法骨干、监督、预训练数据、输入尺寸、增强、epochs、总batch、物理batch、累积、计划更新、优化器/LR/调度、选权/EMA政策、推理设置与基线差异。
3. 数据与初始化manifest、上游commit/patch、依赖锁定、验收报告和Smoke报告。
4. 可用的一条命令启动/查询有限队列的脚本，含`state.json`、`queue.json`、分阶段日志和失败原因；停止/恢复命令要能定位到该队列，不能误杀其他任务。
5. 若长训开关为true：每组best/final及选择记录、全120图预测、统一Val指标、loss/LR/AMP/更新曲线、训练耗时和峰值显存。
6. `results.csv` / `summary.md`：区分未运行、受阻、失败、已完成；不为未运行方法填入论文成绩，不因成绩不佳隐藏已完成结果。比较表中显式列监督、预训练和配方差异。
7. 为之后统一PolypGen评估准备图像独立预测接口和选定权重SHA；本轮不读取PolypGen图像/GT作开发，也不因训练机缺少PolypGen而阻塞Kvasir任务。PolypsSet/PolypDB保持保留。

若你只有本Prompt而没有完整项目，先检查能否找到指定数据和现有实现；仅在确实缺少且无法自行定位的输入上提出具体问题，同时继续其他不依赖该输入的工作。不得自建随机划分或从Mask重算Box来替代缺失的协议。

最后汇报：已实现什么、哪些门禁真正通过、哪些方法受阻、正式训练是否启动、当前阶段及产物路径，并给出完整启动/状态/停止/恢复命令。不要把“配置已生成”写成“训练已完成”。
