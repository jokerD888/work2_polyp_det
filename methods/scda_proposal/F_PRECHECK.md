# F_PRECHECK — 代码阶段记录

记录日期：2026-10-15。**状态：CODE_ONLY / UNVERIFIED；不是工程 GO。**

用户明确要求当前仅编写代码，MaskDINO 基线正在训练，没有额外算力；本次没有运行测试、模型导入、前向/反向、推理、smoke/pilot 或训练。设计文档的运行验收要求留待后续，不作为本次额外占用算力的授权。

## 已静态读取的源码与配置

- 当前原生仓库：`third_party/MaskDINO`，HEAD 指向 `3831d8514a3728535ace8d4ecc7d28044c42dd14`。
- 基线实验：`configs/maskdino_kvasir_official.yaml`。
- 原生模型配置：`configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml`（相对原生仓库）。
- F 配置：`methods/scda_proposal/configs/kvasir.yaml`，继承基线，不替换数据/权重/预算。
- 从 YAML 与原生 config 定义静态可见：R50、hidden dim 256、4 个 feature levels、300 queries、`TWO_STAGE=True`、`INITIAL_PRED=True`、`INITIALIZE_BOX_TYPE=mask2box`、`DN=seg`、`SEMANTIC_CE_LOSS=False`、bbox coefficients 5/2。
- 当前基线配置：1 类，batch 4、AMP、512 LSJ、LR 1e-4、100 epochs/22000 iters，官方 train/val 880/120；仍按 val AP50:95 选 best 并报告 val AP。以上是配置文本信息，**不是本次运行得到的 resolved cfg**。

原生 CUDA 兼容补丁及已有环境未在本次改动；完整环境、运行时 resolved cfg 和权重 SHA 只能在后续授权执行时由入口保存。

## 已确认的接入关系

1. 原生 pre-TopQ 张量为 `enc_outputs_class_unselected: [B,N,C]`、`sigmoid(enc_outputs_coord_unselected): [B,N,4]`；`N=sum(H_l*W_l)`。实际张量尺寸未测量。
2. 原生 Top-Q 原句保留：`torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]`。未重排、替换或引入 GT query。
3. `interm_outputs` 类别另经 `decoder_norm`，不能代替上述 pre-TopQ logits。本实现直接使用原生 raw tensor，不额外调用分类/框头。
4. 原生 flatten 对 `x` 逆序取层，层内为行优先；本方法使用返回的 `spatial_shapes/level_start_index` 建网格，不猜层级或 stride。
5. 原生 Proposal mask 在当前 forward 中可能被禁用，且模型训练调用没有传递 LSJ padding。F 使用独立 mapper 元数据排除 padding，但**不修改**原生 proposal mask、Top-Q 或 attention。
6. 原生 GT bbox 按增强图 `W/H` 归一化；mask pad 到 ImageList。SCDA 按原生各层 `valid_W/valid_H` 和实际 padded 网格推导坐标域，再转到相同 GT bbox 单位。该转换仍需真实像素案例验证。
7. `mask2box` 下真实 Decoder 初始框来源于中间 mask，与 raw Proposal 框不同；分别记录覆盖率，并区分二者的坐标域。
8. 原生 criterion 和 weight_dict 保留；新增 losses 在原生乘权重/移除未知 loss 的循环之后加入，避免二次加权或被删除。
9. 分类/回归分别按 GT 内均值，再对各自有效 GT 归一化；DDP 使用一次全局计数 all-reduce 和 `global_count/world_size` 分母。空候选通过 attached tensor 的空切片求和返回零，避免 `0*inf`。
10. `ENABLED=False` 留原生类名和 mapper；`eval()` 使用原生 forward 且不构建 payload。这里是代码路径描述，**不是一致性测试结果**。

## 用户确认的设计细节

- 沿用 Kvasir；本实现不接入文档中的 PolypDB/PolypGen，也不引用 A/B/C 数值作为当前工程结果。
- 只允许原始单框图片参与 SCDA；全部图片仍参与原生训练。可靠性判定在数据增强之前完成。
- token 单元为实际 feature grid 对应 padded 图像的矩形；mask coverage 为精确像素面积积分。
- 中心距离为按 GT 宽高归一化的平方距离；固定 `(level,u,v,GT id)` 平局规则。
- 任何含 padding 的单元全部丢弃，包括部分 padding 单元。
- Top-Q 匹配占比和重复超额数/Q；多 GT 重复计数仅为诊断按最大 IoU 归属、平局固定 GT 顺序。

现有 mask 仍是 bbox 裁剪的语义前景，不因本次选择而变成真实多实例分割标注。没有改写 COCO JSON、bbox、划分或 mask；没有自动 box-only 回退。

## 已编写内容

独立 Decoder/Meta-architecture、训练期 payload、LSJ 元数据、无参数 assigner、positive-only cls 和 L1/GIoU、全局有效 GT 归一化、关闭开关、源码版本契约、诊断接口、独立训练/预测进程、继承基线的 F 配置和独立结果快照。

实现仅支持 F-v1，没有实现 E/D、FCOS、额外 Query、排名 Loss、动态软标签、centerness、负例采样或权重退火。检测到不兼容 two-stage/softmax/源码版本时明确报错。

## 待后续授权验证

| 验收项 | 当前状态 |
|---|---|
| 实际构建与模型环境导入 | 未执行 |
| A/F unique 参数数、冻结状态、optimizer 对齐 | 未执行；代码未增加预测层/Parameter |
| 关闭开关的 train loss/key/梯度一致性 | 未执行 |
| 同权重 eval boxes/masks/scores/Top-Q 一致性 | 未执行 |
| 单图/不同尺寸 batch、LSJ/batch padding、像素积分案例 | 未执行 |
| 0/1/多 GT、空 mask、无效框、小/细长/贴边目标 | 未执行 |
| 候选配额填满率、空候选率、面积/fill ratio/source/level 分布 | 未采集 |
| 原始与 mask-derived Top-Q coverage、冗余 | 未采集 |
| 共享头/Encoder/Backbone/特征梯度范数、比例、cosine | 未采集 |
| AMP/FP32 数值与 DDP 各 rank 空候选 | 未执行 |
| 显存、步时、部署 FLOPs/Latency | 未测量 |
| Smoke/pilot 与同阶段 A 比较 | 未执行 |
| GO/NO-GO 和正式训练 | 未裁决、未启动 |

文档没有量化“显著多 GT”“大幅干扰”等门限，本次不擅自填入阈值。后续完成上述验收和门限确认后再决定是否长训。没有编造参数量、覆盖率、梯度或性能数值，也没有生成 `F_RESULT_AND_ABLATION_DECISION.md`。
