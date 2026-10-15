# F_PRECHECK — 实际验收与历史记录

记录日期：2026-10-15。**最新状态（18:42）：F-v2 正式100epoch训练、最佳权重预测、统一评测及比较全部完成；本轮val AP50:95=65.7202%，相对原生67.5271%低1.8070个百分点。** 以下先记录最新结果，再保留 F-v1 历史过程；历史中的“待授权/待决策/待正式结果”不再代表当前状态。

## 完整训练与最终统一结果

| 方法 | 预算 | Best iteration（零基）/epoch | AP50:95 (%) | AP50 (%) | AP75 (%) |
|---|---|---|---:|---:|---:|
| MaskDINO | 100ep / 22000iter | 18699 / 85 | 67.5271 | 82.8534 | 72.9928 |
| SCDA F-v2 | 100ep / 22000iter | 19799 / 90 | 65.7202 | 81.4498 | 71.4709 |

F-v2训练18:40:12结束，best-val预测18:41:16结束，后台链路18:41:17完成且退出码0，训练/推理进程退出，GPU计算进程为空。`completion_verification.json`核验两组状态completed、训练末步21999、记录的损失有限、完整训练协议一致、相同评测协议/val标注、预测checkpoint确为best及F-v2冻结源码未变化。统一结果位于 `runs/maskdino_vs_scda_v2.csv` / `.md` 和 `runs/results.csv`。

F-v2 best SHA256为 `7fdb9e06e00f0877df6064d4e1c5b333c3f91954d6c5e5f8ebb3732668c9bba4`。完整协议与结论见 [seed42结果记录](../../docs/results/maskdino_vs_scda_v2_seed42.md)。本轮没有观测到AP收益；只有单seed、同一val选best并报告val，不能声称统计显著性或泛化结论。没有为改善结果改规则、延长预算或重训。原15分钟自动跟进在最终核验后停止。

## F-v2 最新验收（用户明确确认核心2＋中覆盖率2，不跨区间补位）

新定义见 `docs/F_V2_RULES.md`，显式配置 `configs/kvasir_v2.yaml`。F-v1 默认配置与旧产物保留。用户已经授权验收通过后完整训练，不需要再次请求相同批准。

- `runs/_launcher/scda-proposal-v2-kvasir-s42/cpu_tests.log`：20项CPU/Gloo测试通过（3.318秒）。新增覆盖区间边界 .25/.75、核心/中覆盖率缺额不互补、平局排序、Top-Q排除、中覆盖率仅回归梯度、实际3点均值、无候选/无效/可靠性及策略配置拒绝。
- `runs/scda-v2-precheck-20261015/report.json`：真实模型验收通过。unique/trainable参数52,017,477、397个optimizer组，A/F一致；128图mapper精确一致；SCDA-off FP32/AMP loss及梯度检查通过；eval逐元素完全相同；共享分类/框头辅助梯度非零、总梯度有限；峰值7.72GiB。沿用前次固定数值容差与原生自身重复检查，没有放宽预算。
- 初始和成熟原生权重各128张train图：121个可靠有效GT，242个cls点、484个reg点，其中242个为中覆盖率，比例50%；核心和中覆盖率都无缺额。逐GT验证实际cls复用、各区间配额、无重复、不跨区间补位。证据 `SCDA_DECOUPLING_EFFECTIVENESS.json` 和 `mature_baseline_diagnostic.json`（均在上述v2预检目录）。成熟权重只用于诊断，无更新、无val调参、不作为F训练初始化。
- `real_geometry.json`：真实混合尺寸319×447/383×511、pad384×512、合法anchor/token/GT坐标、整单元padding排除、AMP反向通过。
- `scda-proposal-v2-kvasir-smoke50` 于11:36:45完成50iter（保持原始warmup与LR）。188次可靠GT观察均满核心2＋中覆盖率2，376/752 reg点中覆盖率；最终训练记录iter49：total_loss424.8065、weighted cls0.152839、weighted box0.176847，全部有限。末次val bbox AP约0.01445%，仍在warmup，只用于健康检查，不作为与成熟A的性能结论。
- `post_smoke_gradients.json`：加载独立smoke50 final，仅在train batch诊断，不做optimizer更新。AMP损失/总梯度有限，共享分类/框头有辅助梯度；box辅助到encoder/backbone的梯度非零，分别约0.001086/0.000333，排除“初始化零末层导致上游永远无梯度”的误解。bbox头weighted aux/native范数比约0.00163，cosine约0.593；没有据此调整lambda或研究规则。
- `scripts/verify_scda_v2_readiness.py` 已生成 `runs/_launcher/scda-proposal-v2-kvasir-s42/acceptance.json`：检查实际方法源文件SHA与预检/短跑一致、50步完整审计、各项通过、正式配置和基线协议相同。正式run为 `scda-proposal-v2-kvasir-s42`，从同一ImageNet R50重新初始化，100epoch/22000iter，batch4、512增强、AMP、官方bbox/划分、相同LR与val选best。

正式链路保存源码/配置/commit与SHA，依次train、best-val预测、统一评测、CSV及Markdown比较。正式训练后不修改纳入快照的.py/.yaml；不覆盖基线/F-v1/短跑，不把50步AP当作完整性能结果。`scripts/compare_detection_runs.py` 只接受两者均完整结束并且协议一致的结果。

11:38提交正式后台链路，11:39训练worker实际启动。run-id `scda-proposal-v2-kvasir-s42`，实现commit `5188ba8` 已推送私人origin/main；Windows wsl PID14484，Linux supervisor PGID1753，实际训练worker PID3146（后续状态以真实进程为准）。冻结配置SHA `c15d89aa87dc5e1c2e8c03db66bbd4d83e8744e02da7d81a44083d433aba0fd9`，继承基线配置SHA `a0cb684f079354fd92b7152a66c9e7af1a8a33faaf2fc39e5d8c0ecd8f7cb758`。`freeze.json` 和run内source保存代码/配置；正式预测自动检查一致性。每15分钟跟进的原自动任务已恢复，仅实质变化通知；最终输出 `runs/maskdino_vs_scda_v2.csv` / `.md`。

## F-v1 历史过程（保留原始失败与复验）

初次提交只编写代码。随后用户明确授权：基线训练和预测全部完成后执行诊断、修复实现问题，验收通过后启动完整 F。2026-10-15 06:45 开始接续，GPU 仍被基线使用时先以 `CUDA_VISIBLE_DEVICES=""` 执行 CPU 验收，12 项测试通过（日志 `runs/scda-cpu-tests-20261015.log`，3.164 秒）。未中断基线、未改其 adapter/native 源码，真实模型/GPU/正式 F 验收仍待完成。

已通过：手工像素相交计算与非整数矩形积分对齐；任何部分 padding 的排除；不同 canvas/image 单位转换；2/4 配额和 Top-Q 排除、确定性、集合包含；中覆盖率候选不足时阈值确实生效；0/多 GT、空/不可靠 mask、小/细长/边缘及无效框；统一 owner；正类-only 梯度和框梯度；空集 attached zero；逐 GT 均值；两个 CPU Gloo rank 中一 rank 无 GT 及全部无 GT，与串行梯度对齐。

06:56 又在现有模型环境、CUDA 不可见的条件下实际构建原生 A 与开启 SCDA 的 F：两者 unique 参数均 **52,017,477**、trainable 参数均 **52,017,477**；冻结标志、state_dict shapes、全部 **397** 个 optimizer 参数组完全一致。证据 `runs/scda-cpu-model-check.json` 和 `.log`。GPU forward/eval 仍未以此替代验证。

## 后续真实 GPU 验收与机制结果

基线于 07:05 完成训练、最佳权重预测和统一评测：val AP50:95 **67.5271%**、AP50 **82.8534%**、AP75 **72.9928%**，best iteration 18699（约第85 epoch）。不是独立 test AP。

真实模型验收记录为 `runs/scda-precheck-20261015-retry3/report.json`，已通过：参数/optimizer、128 张真实增强图的 image/padding/bbox/mask/class 精确对齐；关闭辅助路径的 FP32 batch1 和 AMP batch4 loss/梯度对齐；开启 SCDA 的 FP32/AMP eval masks/boxes/scores/classes **逐元素完全相同**；辅助 cls/box 对共享头有有限非零梯度；完整反向有限。峰值已分配 CUDA 内存 **8,289,397,248 bytes**（约7.72 GiB）。

最初逐元素梯度检查失败，原生 A 在相同 RNG 下重复反向也失败。所有初次失败和 retry1/retry2 的数值证据保留。验证阶段关闭 TF32、使用确定性 CuDNN，原生 FP32 stem 梯度重复的相对 L2 波动约 `6.8e-7`，A/F 约 `8.9e-7`；AMP 原生重复约 `7e-4`。因此梯度采用**逐参数张量 L2 误差**检查，仍使用原先固定 FP32 `rtol=1e-4, atol=1e-5`、AMP `rtol=5e-3, atol=5e-4` 的误差预算，并要求 A/A 自身重复和 A/F 都通过；元素误差仍完整记录，没有声称 bitwise backward 一致。此数值控制仅用于验收，未改正式训练 worker 的精度/协议。

另以真实图像做受控像素裁剪的混合尺寸 batch：319×447、383×511，ImageList pad 到384×512；合法 anchor 的像素中心、GT bbox 反归一化、整单元 padding 排除和 AMP backward 通过。原生已经置 inf 的无效 anchor 不参与坐标等价检查，也从未进入 SCDA 候选。证据 `runs/scda-real-geometry.json`；可重现入口 `scripts/check_scda_geometry.py`。

**机制发现：阈值放宽未在当前样本中生效。** 同一固定 seed 的128张 train 样本（136个原始GT、134个增强后GT）中，121个可靠GT均取得2个cls/4个reg点，可靠GT空候选率0；全增强GT有13个无辅助候选（9.70%，均来自已确认跳过的多框图）。回归独有位置比例50%，分类集合包含于回归集合；但回归中 `.25<=coverage<.75` 的比例为0，固定k_reg=4将阈值提高到.75时 **0/121** 个GT改变选点。

分别用 ImageNet 初始化和成熟原生 best checkpoint 进行了同样的离线 train 诊断，两阶段上述统计一致；成熟权重仅用于诊断，**没有**作为 F 初始化或用于参数选择。证据 `runs/scda-precheck-20261015-retry3/SCDA_DECOUPLING_EFFECTIVENESS.json`、`runs/SCDA_DECOUPLING_MATURE_BASELINE.json`；可重现成熟诊断入口 `scripts/profile_scda_checkpoint.py`。

初始 AMP 分类头辅助/native 梯度范数比约 `3.70e-6`，cosine约 -0.541；bbox头对应比约0.00252，cosine约0.226。初始 bbox最后层为零初始化，故回归辅助到上游特征的梯度初始为零，而共享bbox头末层梯度非零。不能把初始阶段的比例外推为全程效果；短训练还需要确认优化器更新后行为。没有据此擅自调整lambda。

已询问用户：保留当前“纯前景核心cls2/reg4”版本做完整受控实验，或先确认新的回归候选规则。目前只执行已授权的50iter健康检查；未擅自加入中覆盖率配额、改变排序、阈值、可靠性或预算，正式100epoch等待这个方法定义决策。

07:31，`scda-proposal-kvasir-smoke50` 已完成50iter健康检查和末次val评估，保留best/final，最后记录 total_loss=424.882、weighted scda cls=0.152973、box=0.142312，全部有限。50步的188次可靠GT观察都填满2/4配额，752个reg点的transition计数为0、阈值变更影响计数为0，进一步确认机制现象。其val AP50:95仅0.01457%（只训练50步、处于原始warmup中），不能与原生100epoch best AP比较。证据 `runs/scda-proposal-kvasir-smoke50/train/state.json`、`metrics.json`、`scda_audit.rank0.jsonl`。

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
| 实际构建与模型环境导入 | CPU/GPU 实际构建通过 |
| A/F unique 参数数、冻结状态、optimizer 对齐 | CPU 已通过：52,017,477；397 optimizer 组完全一致 |
| 关闭开关的 train loss/key/梯度一致性 | FP32 batch1 / AMP batch4 通过；数值误差记录见正文 |
| 同权重 eval boxes/masks/scores/Top-Q 一致性 | FP32/AMP 输出精确相同；原生Top-Q调用保留 |
| 单图/不同尺寸 batch、LSJ/batch padding、像素积分案例 | CPU手工案例、128图mapper对齐、真实混合尺寸GPU验证通过 |
| 0/1/多 GT、空 mask、无效框、小/细长/贴边目标 | 纯函数 CPU 用例通过；真实模型待验证 |
| 候选配额填满率、空候选率、面积/fill ratio/source/level 分布 | 128图、初始化/成熟A均已采集；可靠GT均满2/4，transition=0 |
| 原始与 mask-derived Top-Q coverage、冗余 | 两阶段已采集，随候选JSON保存 |
| 共享头/Encoder/Backbone/特征梯度范数、比例、cosine | FP32/AMP已采集；参数/特征空间分开 |
| AMP/FP32 数值与 DDP 各 rank 空候选 | GPU有限性、CPU Gloo两rank均通过；未假装双GPU测试 |
| 显存、步时、部署 FLOPs/Latency | 预检峰值7.72 GiB；正式步时/部署性能尚未测量 |
| Smoke/pilot 与同阶段 A 比较 | 50iter健康检查已完成；没有同阶段A的50iter val点，不能作AP胜负比较 |
| GO/NO-GO 和正式训练 | 工程验收通过；方法定义待用户决策，100epoch未启动 |

文档没有量化“显著多 GT”“大幅干扰”等门限，本次不擅自填入阈值。后续完成上述验收和门限确认后再决定是否长训。没有编造参数量、覆盖率、梯度或性能数值，也没有生成 `F_RESULT_AND_ABLATION_DECISION.md`。
