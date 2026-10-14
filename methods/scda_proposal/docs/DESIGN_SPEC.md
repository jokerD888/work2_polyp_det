# SCDA-Proposal (F)：零新增参数的形状条件化双任务 Proposal 监督

> **状态**：可交付训练机 AI 的研究设计与工程验收规范，**不是已验证的有效方法**。  
> **版本**：F-v1.0，2026-10-15  
> **主任务**：单帧结直肠息肉 Bounding Box Detection；GT instance mask **仅在训练中**可用。  
> **硬约束**：训练/推理的**唯一可学习参数量均 ≤ 同配置原生 Mask DINO**；不增加任何可学习层、检测头、Query 或 Decoder；初版不改原生 Top-Q、Hungarian、DN、Mask Loss、推理后处理。  
> **证据等级**：`[SRC]` 官方公开源码/论文；`[OBS]` 用户汇报的现有实验；`[HYP]` 未验证的机制假设；`[DEC]` 当前提出的具体设计决定；`[GATE]` 训练前必须通过的验收项。

---

## 0. 先给结论：为何选择这一版 F

**推荐 F-v1：Match-isolated, Positives-only, Shape-conditioned Dual-task Proposal Supervision（MIPS-SCDA）**。

- **Match-isolated**：保留原生选中 Top-Q 的 Intermediate Hungarian 监督；SCDA-v1 额外损失只作用于**Top-Q 之外**、且与可靠 GT mask 关联的合法 Encoder Proposal，避免同一选中候选在原生一对一监督与新增一对多监督之间直接获得相反的分类标签。**参数共享造成的梯度竞争依然可能存在，需测量**。
- **Positives-only**：首版**不重新定义额外的全图负样本分类目标**。对每个 GT 只挑选少量高可靠前景位置作为正分类监督；全图背景压制继续交给原生 Mask DINO 目标。避免历史 C/B 中“正样本分母变化放大背景梯度”的风险，但新增正样本可能造成分数膨胀/重复查询，必须监控。
- **Dual-task**：分类正位置集合较严格，回归正位置集合可适当放宽，**均从原生 Proposal 的预测得到损失**；不引入 FCOS LTRB、Centerness 或独立辅助头。
- **Minimal change**：保留原生模型及输出，训练时仅附加无参数的 GT 分配与两项辅助 Loss；推理关闭并导出原生权重结构。

**为什么不直接照搬 Co-DETR / Group DETR / Rank-DETR**：这些研究分别提供“稀疏监督不足”“一对多/一对一协同”“分类分数与定位质量错位”的合理动机，但其额外 Query、辅助头、重排损失或解码器改造会引入独立的复杂因素；F-v1 只吸收动机，不照搬模块。见 §12。

**重要限制**：隔离 Top-Q 是降低训练冲突的保守启发式，**不是文献证明的最优解**。如果因此几乎没有剩余高质量候选，F-v1 不应直接长训，应先报告覆盖不足。

---

## 1. 任务背景与现有实验事实

### 1.1 研究问题

原生 Mask DINO 已有 Mask 监督、Hungarian 以及 Encoder Proposal。我们不声称其“缺乏形状信息”，而是研究：**原生仅选中候选的一对一监督之外，能否利用训练期 GT Mask，为其他 Encoder 空间候选配置更可信的分类和框回归训练责任，从而改善候选表示/排序与跨域检测？**

这一假设主要源于训练期密集位置标注分配的重要性，而不是“多加一条分支必有效”。

### 1.2 已冻结结果 `[OBS]`

| 协议 | A 原生 | B 普通 FCOS 辅助，λ=.1 | C Mask-guided FCOS 辅助，λ=.1 |
|---|---:|---:|---:|
| PolypDB WLI val，AP50:95（358 图/398 框） | 79.6527 | 79.6504 | 80.0035 |
| PolypDB WLI val，AP75 | 86.78 | 87.17 | 88.28 |
| PolypGen 冻结评测集合，AP50:95（1473 图/1533 框） | 64.64 | 62.91 | 65.58 |
| PolypGen，AP50 | 85.33 | 83.60 | 86.04 |
| PolypGen，AP75 | 71.92 | 69.34 | 72.08 |
| PolypGen，Recall@score0.5 | 77.6% | 78.1% | 76.3% |

因此外部 **C−B = +2.67 pp；C−A = +0.94 pp**（单种子，使用 PolypDB val 选出的 checkpoint）。`C−B` 包括挽回 B 退化的部分，**不能当作 F 预期收益**。域外 AP75 C−A 只有 +0.16 pp，尚不能证实定位质量改善是收益主要来源。

现有 C 的训练期 FCOS 头约增加 4.74M 参数（用户汇报），F 不是 C 的参数压缩，而是改换监督的**受监督张量和责任机制**。F 对比主要应是原生 A，C 作为已有正向的参考。

**数据纪律**：上述 PolypGen 结果已经用于研究决策，不能再把这个集合称为“完全未接触的盲测集”。后续仍可按**冻结、只用 PolypDB val 选模型、不依 PolypGen 调参**的协议报告“外部基准评测”；如要独立最终检验，另外保留真正未参与开发的来源/病例级 lockbox。不得按报告建议改为在 PolypGen 上训练后又把它当外测。

---

## 2. 由官方源码确认的 Mask DINO 接入边界 `[SRC]`

官方仓库 `IDEA-Research/MaskDINO` 的 `maskdino/modeling/transformer_decoder/maskdino_decoder.py` 在 `two_stage=True` 时计算：

```python
output_memory, output_proposals = gen_encoder_output_proposals(
    src_flatten, mask_flatten, spatial_shapes
)
output_memory = self.enc_output_norm(self.enc_output(output_memory))
enc_outputs_class_unselected = self.class_embed(output_memory)    # [B, N, C]
enc_outputs_coord_unselected = (
    self._bbox_embed(output_memory) + output_proposals             # [B, N, 4] unsigmoid
)
topk_proposals = torch.topk(
    enc_outputs_class_unselected.max(-1)[0], self.num_queries, dim=1
)[1]
```

原生选中的 `topk_proposals` 经 gather，得到 `interm_outputs`；`interm_outputs['pred_boxes']` 来自被选中的 `enc_outputs_coord_unselected.sigmoid()`，而类别输出经 `forward_prediction_heads`（使用 `decoder_norm` 再调用共享 `class_embed`）得到。`criterion.py` 对 `interm_outputs` 调用原生 Hungarian matcher 再计算原生 losses。上述两种类 Logit **并非同一张量**，共享分类头但中间存在不同归一化路径。**F 的输入必须选 Top-Q 前真正用于选取的那一套 Logits/Boxes，而不是错误复用 `interm_outputs` 的 Top-Q 张量。**

`INITIALIZE_BOX_TYPE != 'no'` 时 Decoder 的实际初始化参考框可能取自 **预测 mask→box**（且有 `detach`），而不是直接使用选中 Proposal 的原始 box。因此 F 的新回归监督仍可优化共享 `_bbox_embed` / 表征，但不能声称一定通过“优化原始 Proposal box→改善 Decoder 初始框”直接生效。

**必要的真实配置核查：** `TWO_STAGE`、`INITIALIZE_BOX_TYPE`、`NUM_OBJECT_QUERIES`、`NUM_CLASSES`、`SEMANTIC_CE_LOSS`、输入/多尺度 Token 顺序、`mask_flatten` 的真实定义、`DN`、`BOX_LOSS`、各原生 loss 权重、AMP/DDP、Top-Q 排名依据。若 `TWO_STAGE=False` 则 F 按本文设计**不适用，应停止**，不得默默切换配置并沿用 A 的公平比较。

> 官方来源：  
> https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/transformer_decoder/maskdino_decoder.py  
> https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/criterion.py  
> https://github.com/IDEA-Research/MaskDINO

---

## 3. 总体计算图、与原生网络的 Diff

```text
TRAIN ONLY
Image ──► Backbone/Pixel Decoder ──► flattened Encoder features
                           │                       │
                           │          native class/box proposal prediction, N tokens
                           │                       │
                           │                 native Top-Q selector ──► native
                           │                       │                     interm Hungarian
                           │                       └──► original Transformer Decoder
                           │                                        └──► cls/box/mask/native losses
                           │
                           └──► [reuse pre-Top-Q logits & boxes; NO extra head]
                                      │
                              filter valid, non-Top-Q tokens
                                      │
                         GT box + reliable GT mask (train only)
                                      │
                       SCDA per-instance capped assignment
                        ├── cls reliable positive positions
                        └── reg broader effective positions
                                      │
                  positive-only cls + L1/GIoU box aux losses
                                      │
                              native shared parameters

INFERENCE
Image → identical original Mask DINO inference graph / Top-Q / Decoder / masks/boxes/scores
```

**实现 Diff**：

| 项 | Mask DINO A | F-v1 |
|---|---|---|
| 模型已有预测模块 | 原样 | 原样 |
| Top-Q / Decoder / DN / Hungarian / Mask 监督 | 原样 | **严格保留** |
| Encoder N 个位置预测 | 存在，但未作为额外的密集训练目标 | **训练期供 SCDA Loss 使用** |
| 额外可学习 Module | 无 | **无** |
| 额外 GT 位置分配代码 | 无 | **纯函数式 SCDAAssigner，无 Parameter** |
| 原生损失 | `L_native` | `L_native` 原样加 `L_scda` |
| 推理路径/输出 | 原生 | **原生同一路径** |
| 可学习参数唯一数 | P | **P**；DDP、训练和导出前后必须计数 |

零参数只约束 Parameter 数目，**不保证训练时 FLOPs/显存零增长**；允许训练期独立的 Mask 覆盖率张量与稀疏 Proposal Loss 临时张量，需报告峰值显存与单步时长。

---

## 4. SCDA 位置与形状定义：先避免 FCOS 迁移错误

### 4.1 基本记号

- 图像 b、GT j：`class c_j`、增强后的完整有效 GT box `B_j`、可选可靠 instance mask `M_j`。
- Encoder token p 属于实际尺度 level ℓ，空间索引 `(u,v)`；原生编码 Proposal anchor `a_p=(x,y,w,h)`，预测 `\hat b_p=sigmoid(Δb_p+inverse_sigmoid(a_p))`，**精确采用原始实现的参数化，不要重复 inverse_sigmoid**。
- Token 的图像对应区域 `Ω_p`：应按实际特征网格、有效图尺寸、batch pad、mask downsampling 的约定计算，不允许凭假定 stride8/16/32 静态硬编码。
- Mask 覆盖率 `r_j(p)=|Ω_p∩M_j|/|Ω_p|`，值域 [0,1]；`Ω_p` 必须落在**增强后/补齐后**相应像素坐标系，Padding 无效格全部丢弃。
- 合法 token 集合 `V`：`mask_flatten` 有效、原生 anchor 提案合法、预测框/Logits finite、中心处于有效图像范围；必须剔除无效框（宽高≤0、非有限数）和无效 padded proposals。
- `T`：原生已选 Top-Q token index 集合，**只用 `detach` 的 index 作标签过滤**，不改变 Top-Q 本身。
- 可用于新增监督的集合 `U=V\T`（多 GT 冲突时每个 Token 至多被一个 GT 持有）。

> **极其重要**：原生 Proposal anchor 的宽高是多尺度预设，网络会学习回归；其特征点不是 FCOS 的“四边距离”或独立 Centerness 预测。F **不复制**原 FCOS 的 LTRB 尺度阈值、中心度回归或 `r≤0.25 → 全图显式负样本`。

### 4.2 GT 可用性

- `M_j` 必须来源可靠、与 `B_j` 同一增强坐标系，既不读取测试 GT Mask 也不使用模型预测 Mask 代替训练 GT；不能将目录名/类别泄漏到标签中。
- Mask 不可靠/缺失：默认**该实例跳过新增 SCDA**，原生 Mask DINO 训练不受影响；不要在 F-v1 自动换成普通 B 指派造成“有 Mask 与无 Mask”混合机制失控。
- 对 `mask_derived` 框与 official 框分别统计；如二者定义不一致须先按真实标注规范处理，不能用 Mask 推导框替换原生 GT box。
- 若整图无 GT（健康/阴性帧），SCDA 正样本损失为**可微的 0**，交由原生分类负样本机制管理；不能无故从整图制造成千上万个新负样本。

### 4.3 候选集合（最小、可实现、可测试）

对每个可靠 GT j：

1. **几何资格** `G_j = {p∈U: 原生 anchor/特征网格中心位于 B_j 内}`。可用原生 anchor 中心判断；多尺度统一采用归一化坐标。中心位置是候选筛选条件，**不声称中心位置是预测 box 中心或目标外观的真实中心**。
2. **分类候选** `C_j={p∈G_j: r_j(p)≥τ_cls}`。
3. **回归候选** `R_j={p∈G_j: r_j(p)≥τ_reg}`，且 `τ_reg≤τ_cls`。
4. **任务独立筛选**：`P_cls,j=TopK(C_j, k_cls, deterministic_priority)`；`P_reg,j=TopK(R_j, k_reg, deterministic_priority)`。不要求同一位置同时参与两任务。
5. **候选不足**：允许 `|P_cls,j|=0` / `|P_reg,j|=0`，记录实例级率；**不强行从明显背景或其他实例中回退造正样本**。只有在 dry-run 统计表明覆盖足够时才可长训。

推荐优先采用**已有 C 的相同阈值作为预注册起点**：`τ_cls=0.75, τ_reg=0.25`，而非通过 PolypGen 反复找最佳阈值。这两个阈值不是理论最优值；若离线预检查发现很多实例零正样本，应在开训前仅依据训练数据分布重新设计资格，而非训练后改阈值追指标。

**可控正样本数量起点**：`k_cls=2/GT`，`k_reg=4/GT`，并**在训练前**根据每实例候选计数、同类重复 GT 与尺度分布进行无学习离线审核。这两个 K 只是成本/干扰保守的工程初始化，不宣称最优，不在 PolypGen 上搜索。

**确定性排序**（不加可学习参数）：优先高 mask coverage，次级按归一化的预测前 GT box 中心距离较小、最后按稳定的 `(level,u,v,GT id)` 断平局；候选筛选仅用训练 GT、原生固定几何和 `detach` 的索引，不用当前预测分类分数选择正标签，避免把错误初期评分反馈到标签中。可在输入数据预处理环节为每个 GT 的候选额外记录形态特征，但 F-v1 不引入可学习的形状编码器。

**多实例冲突**：如果同一个 p 被多个 GT 争取，应以 `r_j(p)` 优先、随后中心距离/固定 GT index 作为确定性仲裁，每个 token 最多关联一个 GT。真实实现需保证 `P_cls` 与 `P_reg` 对同一 p 的 GT 归属一致（若任务集合都包含 p）；先按统一 owner，再分别选任务候选，不能各自分配成两个 GT。

**位置分布偏置**：不同尺度 `Ω_p` 可能有不同覆盖率，且细小病灶容易失去高覆盖率单元；必须统计 `N_gt`、每个 GT 的可用数、零覆盖、按尺度、GT fill ratio 和面积分组。若细小目标长期不具备 `r>=0.75` 单元，则 F-v1 的前提不满足，停止，不要硬训练。

---

## 5. 真正的双任务监督定义

设 `z_{p,c_j}` 是**原生 Top-Q 选择前**的分类 logit，`\hat b_p` 是该位置原生 `sigmoid(enc_outputs_coord_unselected)` 框；所有 `p` 均为 `U=valid∖TopQ` 内的合法 token。

### 5.1 分类责任：可靠前景正样本（无额外负类项）

对每个 GT：

\[
\ell_{cls,j}=\frac{1}{|P_{cls,j}|}\sum_{p\in P_{cls,j}}
\operatorname{softplus}(-z_{p,c_j}).
\]

这等价于只对 GT 类别计算 positive BCEWithLogits，不新建密集负类标签；若是单类息肉任务，`c_j=polyp`。多类则按真实类索引 gather 对应 logit，**不把别的正类统一改成隐式负类**。若源码启用了 `SEMANTIC_CE_LOSS` 并使用含背景 Softmax，须另行适配为适当的 GT 类正类交叉熵、核对排名背景通道，不能照搬 BCE 公式。

**为何这样做 `[HYP]`**：原生 Matcher 已承担大量一对一匹配与未匹配 Query 的分类监督，新加全图负损失很可能更直接地改变原生共享分类头尺度；低量高纯度正监督更接近“补充覆盖”而非重复一个完整 FCOS 分类器。

**可能失败**：只做正类会提高多个同一目标候选的评分，降低 Top-Q 多样性或增加误报。故用固定小 K、`TopQ` 隔离、完整原生 Loss 和诊断仪表控制；如果出现重复/虚假高分趋势，**停止 F-v1，不临时加入一套硬负例采样以掩盖问题**。

### 5.2 定位责任：稍广集合、完整 Box 监督

\[
\ell_{box,j}=\frac{1}{|P_{reg,j}|}\sum_{p\in P_{reg,j}}
\left[w_{L1}\|\hat b_p-B_j\|_1+w_G\bigl(1-\operatorname{GIoU}(\hat b_p,B_j)\bigr)\right].
\]

`w_L1,w_G` 最初采用原生 Box Loss 比例（从 resolved config 读取，不硬编码）；`\hat b_p` 和 `B_j` 均为有效图尺寸下同约定归一化 `cxcywh`，GIoU 转 `xyxy`。禁止将增强前 Mask 坐标与增强后 Box 坐标混合；如果公式中 `w_L1/w_G` 被复用，须把它们记录为“原生比例复用”而不是声称两类损失数值可直接比较。

**为何不放宽到全部框内**：GT box 内、Mask 外的候选不天然不可用，也不天然可靠。首版只放宽到 `r≥0.25`，避免高成本实验引入大量显著背景位置。**为何不能保证有利**：高 Mask 覆盖率不是定位质量；即便分类和回归职责不同，额外宽松的回归正位置也可能干扰共享 Box Head，需要实验证明。

### 5.3 总 Loss、实例归一化与空实例

记当前 Batch 内能提供至少一个相应任务候选的 GT 集合分别为 `J_cls`、`J_box`：

\[
L_{F}=L_{native}
+\lambda_{cls}\frac{\sum_{j\in J_{cls}}\ell_{cls,j}}{\max(|J_{cls}|,1)}
+\lambda_{box}\frac{\sum_{j\in J_{box}}\ell_{box,j}}{\max(|J_{box}|,1)}.
\]

- 必须先进行**每 GT 内 mean**，再对有效 GT mean。防止大目标产生更多候选从而主导辅助梯度。
- **DDP**：明确全局有效 GT 计数的 `all_reduce` 与梯度平均是否一致，确保有效全局 Batch 大小时的 Loss 缩放一致；不把每卡局部均值无条件当成全局均值。
- 空集要返回连接计算图的零值，例如 `0.0 * logits.sum()`，不能创建 CPU `torch.tensor(0.)` 导致设备不一致、潜在未参与参数不同步或非预期优化器行为。
- `λ_cls` 与 `λ_box` 不能照抄 C 的 `λ=0.1` 后默认优化强度一致，因为 C 的头、标签、分母、目标不同。**建议初版固定低权重起点** `λ_cls=λ_box=0.05`，开训前按相同训练 Batch 的真实共享层梯度检查；若明显过强/数值异常，先在冻结开发协议下重新登记配置与选取依据后再开正式训练。该数值是工程起点，不是理论最优。
- 不做 MSRA 连续权重、IoU 动态分类软标签、负例挖掘、中心度、λ 退火、GT Query 注入或新 Matcher；这些是后续相互独立的候选机制，不能在 F-v1 中一起试。

---

## 6. Top-Q 冲突隔离：F-v1 的关键安全约束

### 6.1 原生一对一 vs 新增一对多

`interm_outputs` 在选中的 Top-Q 上经过 Hungarian，一张图只有少量 GT 时，大量未匹配 Query 会收到背景类别监督；如果 SCDA 又把这些**同位置**设为正类，可能出现任务目标竞争。尽管原生 Top-Q 分类预测和 pre-TopQ 分类 Logits 不是完全相同张量，它们共享 `class_embed`，直接硬叠加仍可能不稳定。

**F-v1 决定 `[DEC]`**：

\[
P_{cls,j}\cup P_{reg,j}\subseteq V\setminus T,
\quad T=\operatorname{TopQ}(z_{pre}).
\]

不删除/降低任何原生 `interm_outputs` Loss；不修改原生 Matcher；SCDA 排除选中 Token 的监督，优先从其余候选中补足训练责任。如此**能避免同一 token 的直接重叠监督**，但不能消除共享参数梯度方向冲突。必须测 `cos(g_SCDA,g_native)`。

### 6.2 防止 Top-Q 重复竞争

仅对少量 per-GT 候选施加正分类，既控制正类梯度，也降低将某个 GT 周围大量位置推入 Top-Q 的可能性；但**不是重复候选的严格约束**，且不能保证无冲突。

必须输出三类量：

1. **Candidate Coverage@Q**：GT 是否被任一 Top-Q 原始 Proposal box 以 IoU≥0.3/0.5 覆盖；阈值固定并与真实引用的候选框种类对应。
2. **Effective Decoder Init Coverage@Q**：若 `mask2box` 启用，进一步按真正送进 Decoder 的 Mask-derived reference box 计算 Coverage，避免把原始 Proposal 质量误当作实际初始化质量。
3. **Redundancy**：每个 GT 的 Top-Q 中 IoU≥0.3 的候选数量及其分布、重叠候选比例；候选覆盖不足与重复聚集的情况分开分析。

定义所有统计为训练/验证阶段**离线诊断**，不可用 val GT 生成推理时 Query，也不能把 GT-driven diversity 干预偷偷接进原生 Top-Q。

---

## 7. 工程实现说明（伪代码，不是现成可运行补丁）

### 7.1 最小源码改动位置

1. `maskdino/modeling/transformer_decoder/maskdino_decoder.py`：在 `two_stage` 分支获得 pre-TopQ 原始输出后，**仅 `self.training and enable_scda`** 时向返回的 dict 附上 `scda_payload`：`raw_cls`, `raw_box_sigmoid`, `base_anchor`（或解码 index→坐标的必要信息）、`topk_indices.detach()`, `spatial_shapes`, `level_start_index`, `valid_token_mask`。不额外调用 `class_embed`、`_bbox_embed`；不在 Eval 构建该 payload。
2. `maskdino/modeling/scda_assigner.py`（独立纯函数文件）：GT box/mask 映射、token 与 mask cell overlap、owner 仲裁、两套 capped 正样本 index，**不得继承 `nn.Module` 并创建任何 `nn.Parameter`**。
3. `maskdino/modeling/scda_loss.py`：对原生 pre-TopQ 张量用 index gather 计算分 GT mean 的正类 BCE、L1/GIoU；**不创建新的分类或框预测层**。
4. `maskdino/maskdino.py` 或训练 wrapper：保留原始 `SetCriterion(...)` 不变，在 criterion 后将 `L_scda` 接入总损失（准确接入点须检查原生 loss dict/weight dict 的加权逻辑，避免误算二次权重）。
5. `configs/...`：新增纯标量配置 `SCDA.ENABLED`, `TAU_CLS`, `TAU_REG`, `K_CLS`, `K_REG`, `LAMBDA_CLS`, `LAMBDA_BOX`，以及审计开关；全部默认 False/原生配置以验证“关闭即原版”。

**注意：用户真实训练仓库未在当前对话提供完整源码，本节是基于官方 upstream 的接入目标，不应未经实际 resolved config 对号入座。**

### 7.2 概念伪代码

```python
# original MaskDINO forward, no changes in eval
raw_cls = class_embed(output_memory)                  # [B,N,C]
raw_box_unsigmoid = bbox_embed(output_memory) + base_proposals
idx_topq = raw_cls.max(-1).indices_or_topk(...)       # call ORIGINAL topk; do not reimplement

if training and scda_enabled:
    payload = {
        'raw_cls': raw_cls,                       # no detach; gradient to shared head
        'raw_box': raw_box_unsigmoid.sigmoid(),   # no detach
        'selected': idx_topq.detach(),
        'spatial_shapes': spatial_shapes,
        'valid': derive_exact_valid_token_mask(), # align with native flatten layout
        'anchor_meta': extract_native_anchor_meta(),
    }
# rest of original decoder/interm/denoising returns untouched

loss_native = original_criterion(original_outputs, targets)
if training and scda_enabled:
    P_cls, P_reg, gt_ids = assign_scda(payload, targets, masks)
    loss_scda = positive_cls_loss(payload, P_cls, gt_ids)
    loss_scda += positive_box_loss(payload, P_reg, gt_ids)
else:
    loss_scda = 0
loss_total = loss_native + loss_scda
```

注：`idx_topq` 行仅示意，**必须调用/保留官方原有 Top-K 原句**。不能从伪代码的 `.indices_or_topk` 生成新选择器。

### 7.3 单元测试必备

- `SCDA.ENABLED=False` 时，相同 checkpoint + 相同输入 Eval 结果与原生完全一致；训练时关闭的 loss_dict/key/梯度也与原生路径对齐。
- `SCDA.ENABLED=True` 但 `model.eval()`：输出 key、tensor shape、Top-Q、boxes、masks、scores 与 A 同权重严格一致（容许按固定精度容忍极微数值差异，记录精度）；辅助路径无任何计算与参数调用。
- `sum(p.numel() for p in model.parameters())` 与原生一致；另按参数唯一 `id` 去重，防止共享模块被重复统计；Optimizer 参数组和冻结状态对齐。
- 单图/不同大小同 batch，`mask_flatten` 实际顺序、GT box归一化维度、Mask resize/pad、各尺度 cell 的几何对应全部对齐。**不能默认 adaptive_avg_pool2d 与原生坐标约定完全等价，需可视化和像素级案例测试。**
- 高覆盖率正样本必在可靠 GT 内，正样本排除 `topk_indices`，多个 GT 的任务 owner 不冲突，无效 Mask/全阴性图安全返回零。
- 原生 Decoder DN query 不进入 SCDA `N` 空间 Token 索引；`Top-Q` 只指其 Encoder 初始候选，避免把 DN/Decoder Query index 当作空间 Proposal index。
- 0/1/多 GT、极小/细长、贴边、空 Mask、无效 Box、Mask derived/official、DDP 多卡各自 positive count 零值等测试均通过。
- FP16/AMP 下 GIoU 与 logits 未出现 NaN/Inf，建议在 FP32 计算 box IoU/GIoU，梯度正常回传。

---

## 8. 不重训诊断 + 短训健康验收：比再设计模块更优先

### 8.1 完整长训前的 P0（必须出表）

**真实配置及路径**：两阶段、初始化类型、preTopQ 与 interm 预测张量、真实参数共享、Loss scaling、AMP/DDP。

**仅统计训练数据**（代表性的真实数据增强 Batch）：每个 GT 的 `N_valid`、`N_cls_eligible`、`N_reg_eligible`、排除 Top-Q 后能否取得 2/4 个位置的比例；按面积、fill ratio、来源、level 分层；多实例冲突率；原生 Top-Q 覆盖及冗余。

**原始损失尺度**：在相同 Batch、相同参数和不同标签分组下，测原生梯度与 SCDA cls/reg 梯度的分层范数与 cosine。重点检查 `class_embed`、`_bbox_embed`、`output_memory`、Encoder 和共享 Backbone，报告哪个子损失造成哪项梯度。不能跨空间比较参数梯度和特征梯度之比。

**失败条件**：Top-Q 排除后大比例 GT 无辅助候选；padding/实例冲突不可可靠处理；新增 SCDA 输出为 NaN/Inf；参数计数或 Eval 路径不符；新增 Loss 没有到共享头的梯度；或原生训练被大幅干扰且无法通过冻结配置预定的小权重修复。失败时报告 `NO-GO`，**不要擅自启动正式训练**。

### 8.2 短训验收

- 50–200 个迭代的 smoke + 若成本允许 5%–10% 日程 pilot；必须与同种子 A 的对应**训练阶段**比较，不用成熟 A 的 best AP 与早期 F 直接判输赢。
- 每隔固定步数记录 `L_native`, `L_scda_cls`, `L_scda_box`, 梯度比、cosine、Top-Q Coverage@.3/.5、候选冗余、AP/Recall 的小型 val 监控。
- **守门而非挑分**：短训用于排除崩溃、过度重复 Query、候选 GT Coverage 系统性坍塌、严重负梯度和数值问题；它不能可靠预测最终 50ep AP。
- 若通过，则锁定代码/配置/Seed、记录 SHA256 与 Git HEAD，才能执行单次 F-seed0 正式训练。

---

## 9. 正式评测、参数预算与止损线

### 9.1 协议完全对齐

- 相同 PolypDB WLI Train/Val（实际完整图/框清单、划分 SHA 和增强配置与 A 对齐），同预训练权重、LR/optimizer/调度、Epoch/iter、batch/AMP/DDP；A/B/C 完整冻结不覆盖。
- 模型 checkpoint 只根据 PolypDB val AP50:95 选 best；记录 best 和 final。
- PolypGen `annotations_coco.json` **1473 图/1533 框**，按原 batch=1、官方 `pycocotools`、相同图像预处理、maxDets、image_id/类别映射/GT，依已冻结协议评估。不以 PolypGen 重新挑阈值、checkpoint、k/τ/λ，更不能直接重训 PolypGen 然后当外测。
- PolypGen 已参与开发，因此应标记为“开发过程可见的跨数据基准”；论文最终更强的结论还需要未参与设计的数据中心或病例级 lockbox。

### 9.2 主结果与机制指标

- Box AP50:95（主）、AP50、AP75、按中心 AP、APs/APm/APL（报告每组 GT 数）、外部固定工作点 Recall / FPperImage、相同 FP/图 Recall / FROC。
- `A vs F` 是零参数模型的主结论；`C vs F` 为当前历史最好方法参考（训练参数量不同；不作为“同结构监督规则”的纯因果对照）。
- 补充 Encoder Candidate Coverage@Q（原始与 Mask-derived Decoder init 两种参考框分开）、Top-Q GT 冗余率、SCDA 正样本分布、score-box IoU 关联、预测 mask quality 与 box 原生分数链路。
- 附部署检查：Eval 参数数=P、FLOPs/Latency 与同权重 A 的差别、实际输出配置 identical；训练成本增量另行报告。

### 9.3 GO / NO-GO

**工程 GO**：零新增参数、原生 eval 一致、归一化与几何对齐、无异常梯度、Coverage 无严重坍塌。

**研究 GO**：F 在 PolypDB 具可解释的变化，PolypGen 相比 A 出现预先定义的实质正向且关键机制指标未恶化，并且有可复验路线。若仅某一个 AP 偶然上升，先按 seed 或病例不确定性分析；不能直接宣称成功。

**研究 NO-GO**：F 的域外净增量不足/负向、或候选覆盖/召回持续恶化、或为获得好数值需要不断改变 Mask/Query/Match/推理模块；应保留现有 C 并完成成对复验，而不是无限迭代 F。

### 9.4 如果 F 首轮成功，需要怎样消融

| 对照 | 作用 | Train 参数 | 当前执行时机 |
|---|---|---|---|
| A 原生 Mask DINO | 必须超过的强基线 | P | 已有 |
| C 旧 Mask-guided FCOS aux | 历史收益与推理无增参考 | P+4.74M | 已有 |
| **F** Top-Q 隔离 + 少量可靠分类正样本 + 较广回归正样本 | 主方案 | P | **优先** |
| **E** 同样 Top-Q 隔离、正类策略、Loss/Normalizer，但 `P_reg=P_cls` | 证明“回归责任解耦”的独立增量 | P | F 有信号后 |
| **D** 同样流程，把 Mask 覆盖筛选换成 Box 内几何筛选 | 证明“Mask 形状条件”的独立增量 | P | F 有信号后 |
| 可选 F w/o `TopQ` exclusion（冲突安全备选单变量） | 验证隔离必要性（高风险） | P | 机制允许才考虑 |

**匹配注意**：E/D 对照必须在 Loss 权重、正样本数上尽量对齐，排除“多监督了 2 个位置，所以性能更好”的伪差异；可固定 per-GT 候选配额但报告实际计数。特别注意 D 的任务集合定义应避免“Box assigner + Mask assigner”附加干扰造成不纯消融。

不同 λ、不同候选策略及多种子都应从训练期开发协议决定，禁止以 PolypGen 结果定方案。

---

## 10. 方法的理论依据、创新边界与失败机制

| 事实/文献 | 可以支持的研究动机 | **不能自动证明** |
|---|---|---|
| Mask DINO 官方两阶段 Proposal + Intermediate Matching `[SRC]` | 有可复用的原生 pre-TopQ 张量；无需额外预测头 | 给所有空间位置增加 Loss 必然提高 AP |
| Co-DETR（ICCV 2023） | 原生 One-to-One 对共享 Encoder 可能造成监督稀疏，一对多监督有价值 | 额外辅助头/GT Query 可无风险整合，或者我们的方案首创一对多 |
| Group DETR（ICCV 2023） | 可在训练中增加正样本监督而部署不改推理 | 必须复制其分组 Decoder 才能成功 |
| Rank-DETR（NeurIPS 2023） | 高分类分数并不一定对应高 IoU；Proposal 排序质量值得监控 | 高 Mask coverage=高 Box IoU，或应该直接改 Top-Q 排名 |
| FCOS / TOOD | 空间位置分配、分类与定位对齐是成熟问题 | FCOS 中心度/尺度/阈值对 Transformer Proposal 直接适用 |
| 当前 A/B/C `[OBS]` | 改变指派规则可能影响跨数据表现；仅添加辅助头不一定有益 | F 一定比 C 更好，或者单种子 +0.94pp 已稳定 |

**可主张的方法贡献（仅在验证后）**：

1. 设计一种**不新增参数**、直接作用于原生 Encoder Proposal 的训练期 GT Mask 形状条件化分类/回归责任分配机制。
2. 提出一种与原生 Hungarian Intermediate 保持兼容的**Top-Q 隔离式少量正样本**辅助学习策略。
3. 通过与原生 A、统一任务责任 E、Box-only D 和现有 C 的比较，检验不同任务责任/Mask 归属是否确实改善息肉外部检测。

**创新性风险**：One-to-Many、Mask-aware 指派、Task Alignment、参数无增量本身已有近似工作；“Top-Q 隔离 + 形状责任”目前也尚未完成详尽 novelty search。不能将 F-v1 宣称为原创基础理论；“仅训练时改变 loss”作为硕士论文工作量是否足够仍依赖完善的消融、统计与问题机制解释。

**潜在失败机制**（不是结果事实）：

- 由于 Top-Q 已覆盖大多数目标，剔除选中位置后的候选质量较差；额外正类梯度将劣质位置推高，造成排序恶化。
- 正类-only Loss 无背景约束，可能让特定物体局部多个 Proposal 评分同时升高、Top-Q 冗余增加。
- `class_embed` 和 `_bbox_embed` 与原生 Decoder 共享，虽然排除了重叠 token，但参数级梯度仍相互竞争。
- 当前 Mask DINO 使用 `mask2box` 时，原始 Proposal 框对 Decoder 初始框的直接作用有限；新增回归收益可能小于分类收益。
- GT Mask 覆盖高的单元可能不具有好的 Box 回归信息；小/细长目标可能无候选。
- 已参与方法开发的 PolypGen 外测结果存在方法选择偏差，不应夸大统计独立性。

---

## 11. 给训练机 AI 的可执行任务单（复制本节即可）

**目标**：基于当前冻结 A/B/C 所用的真实 Mask DINO-R50 代码与配置，实现 `F-v1 MIPS-SCDA`，不用额外的 FCOS 分支，训练与推理的唯一可学习参数数目均严格等于 A。不要替换 A/B/C 的历史文件。仅 GT Mask 训练期参与标签构造；推理与 A 同图。

**先提交 `F_PRECHECK.md`（不跑长训）**：

1. 真实 Git commit、Mask DINO 派生版本、resolved cfg、训练配置/模型参数计数。确定 `TWO_STAGE=True`、原生 preTopQ class/box 的形状、valid token flatten 顺序、Top-Q gather 以及 `mask2box` 状态。若条件不满足先 `NO-GO`。
2. 实现仅训练使用的 `scda_payload` 原始类别/Box/几何/TopQ 索引暴露；实现无参数 SCDAAssigner 和正-only cls+L1/GIoU Loss。默认 `τ_cls=.75, τ_reg=.25, k_cls=2, k_reg=4, λ_cls=λ_box=.05`，**这些仅为首版固定起点，须事先通过训练数据离线覆盖统计和梯度健康检查**。
3. SCDA 的正样本必须来自**可靠 GT Mask 对应、合法空间 Token、且未被 Top-Q 选中**的位置；分配要无多实例冲突、无 Padding/尺度混淆、无预测类置信度自举筛选，分类和回归责任允许不同。
4. 保留原生 Hungarian、`interm_outputs`、DN、mask2box、Decoder、Mask Loss、Top-Q 排名、推理路径；不得偷偷减原生 Loss、加 Query、改变 `INITIALIZE_BOX_TYPE`，也不得新增可学习头。
5. 完成单元/梯度/关闭开关一致性测试；记录全模型 unique 参数数、训练/推理参数数、各损失梯度的层级范数和 cosine、Top-Q coverage/冗余、运行成本。
6. 对真实训练增强后的数据做候选可用性统计；**如显著多 GT 无可用正样本，或新增梯度严重干扰原生，出 `NO-GO` 而非硬训**。
7. 短训通过后，**先征得明确确认再运行 F-seed0 全程长训**；仅 PolypDB val 选 Best，沿用冻结 PolypGen 1473 图/1533 框 batch1 协议单次报告，保留日志/预测/SHA/commit，输出 `F_RESULT_AND_ABLATION_DECISION.md`。

**严禁**：同时上 Co-DETR 辅助 Query、Rank-DETR 排名 Loss、MSRA 软权重、GT Mask→推理 Query、换训练集、改模型主干、改评测协议或依 PolypGen 调参。若希望更改 F 核心设计，先单独提出变更说明与可检验动机。

---

## 12. 参考文献、源码与本设计的来源界限

1. **Mask DINO (CVPR 2023)** — *Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation*：https://arxiv.org/abs/2206.02777 ; 官方源码 https://github.com/IDEA-Research/MaskDINO ；原生 two-stage 提案 https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/transformer_decoder/maskdino_decoder.py ; intermediate Hungarian https://github.com/IDEA-Research/MaskDINO/blob/main/maskdino/modeling/criterion.py 。
2. **Co-DETR (ICCV 2023)** — *DETRs with Collaborative Hybrid Assignments Training*：https://openaccess.thecvf.com/content/ICCV2023/html/Zong_DETRs_with_Collaborative_Hybrid_Assignments_Training_ICCV_2023_paper.html ，https://github.com/Sense-X/Co-DETR 。
3. **Group DETR (ICCV 2023)** — *Group DETR: Fast DETR Training with Group-Wise One-to-Many Assignment*：https://arxiv.org/abs/2207.13085 。
4. **Rank-DETR (NeurIPS 2023)** — *Rank-DETR for High Quality Object Detection*：https://papers.nips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html ，https://github.com/LeapLabTHU/Rank-DETR 。
5. **TOOD (ICCV 2021)** — *TOOD: Task-aligned One-stage Object Detection*：https://arxiv.org/abs/2108.07755 。
6. **FCOS (ICCV 2019)** — *FCOS: Fully Convolutional One-Stage Object Detection*：https://arxiv.org/abs/1904.01355 。
7. **本研究输入**：用户上传的《Mask-guided Dense Auxiliary 四轮审查结论与下一阶段实验决策》、两 AI 整合调研以及 PolypDB WLI/PolypGen 已完成 A/B/C 实验汇报。这里只引用其研究观察，**未访问训练主机真实仓库/代码/权重/日志**；任何涉及本地实现的事实需训练机 AI 复核。

---

## 最后裁决

**GO（开发实现与 smoke/pilot）**：同意采用这版 F-v1 作为下一候选，理由是它利用原生 Pre-TopQ 参数预测，消除了额外头的参数归因因素，并在工程上采取了最小干扰的正样本和 Top-Q 隔离设计。**NO-GO（未经预检直接 50ep 长训）**：不承诺它胜过 C/A，尤其共享分类头的一对多正监督、训练 Mask/Proposal 坐标对应和 `mask2box` 可能引入实质风险。完成前置验收且用户明确批准，才做一次正式实验；若失败，保留已得到正向外测结果的 C，不应无限追加模块。
