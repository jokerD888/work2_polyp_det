# 冻结权重的 Box–Mask 初始化干预

用户批准执行：已有 Baseline、MSSL V1，三条件，PolypGen 全量为主要开发诊断，Kvasir Val 辅助复核；不新增训练，不开发自适应头，不搜索系数/阈值，PolypsSet/PolypDB 保留。

## 固定实现

原生 MaskDINO commit `3831d8514a3728535ace8d4ecc7d28044c42dd14` 和安装源码保持不变。解析该版本真实 forward 的 AST，在 `tgt_mask = None` 前插入唯一参考框赋值；使用原函数 globals 的私有副本与临时实例方法，退出时恢复。原生 Top300、Mask-to-Box、INITIAL_PRED、九层 Decoder、Mask 质量融合与 Top100 均继续执行。

- native：返回原 Mask 参考框 logits 同一张量，不做 sigmoid/inverse_sigmoid 往返。
- encoder：返回对应 Query 的原 Encoder 参考框 logits，与原生关闭 Mask 初始化的表示一致。
- blend05：两份参考框 logits 各经 sigmoid 得到归一化 cxcywh，再计算 `0.5*Encoder + 0.5*Mask`，最后用原生 inverse_sigmoid 转换。

Mask 分支使用原生 sentinel 转换、归一化、inverse_sigmoid 数值处理之后的实际框。空 Mask 和零宽高不加回退；sigmoid后的实际归一化参考框若非有限则终止并记录，不能根据表现选择有利回退。原生 Encoder proposal logits 可为±inf，原生Decoder先sigmoid得到有限边界坐标；保留这一行为，并记录无穷logit与边界坐标数量。

每图每条件重新执行完整检测器。核对 Encoder Top300 索引、Query 内容、Encoder 框和初始预测 Mask logits 逐位一致；最终分类、Mask 和分数随 Decoder 重新计算，不冻结最终排序。

## 验收和队列

入口：`polypbench.analysis.initialization_queue`；工作任务：`polypbench.analysis.initialization_intervention`。

CPU 入口 `scripts/check_initialization_intervention.py` 检查实际原生前后端（仅 attention stub）的参考框挂载、原生条件逐位一致、Encoder-only 与原生 no 路径一致、方法恢复/异常恢复、净得失计数，以及既有阶段/COCO 重复 FP 回归。通过后冻结测试源码 SHA。

本机独立源码快照和有限串行队列，顺序：

1. Baseline GPU Smoke，四张：按 GT 选首张负图、单目标图、双目标图，加一张未重复 C5 图；不按预测成败选样。
2. MSSL 同一固定 Smoke。
3. Baseline 全部1473张 PolypGen，三条件。
4. MSSL 全部1473张 PolypGen，三条件。
5. Baseline 全部120张、130 GT Kvasir Val，三条件。
6. MSSL 同一完整 Kvasir Val，三条件。
7. 汇总报告。

Smoke 另跑未修改的原生 forward，所有捕获的框、类别分数、Mask 质量分数与新原生条件逐位相等；两组 Smoke 全通过才释放全量任务。全量 PolypGen 的新原生条件与上一轮逐图 NPZ 的所有数组逐位核对，包括全300框、各层框、Top100框、分类与 Mask 质量及最终分数。历史数组、图像、GT、权重、实验与源码先冻结 SHA；各任务校验。Kvasir Val 的原生框/最终分数与原训练保存结果按历史固定容差核对。

GPU 空闲门槛沿用显存≤2048MiB且利用率≤20%，间隔30秒连续两次满足。失败即停止后续任务，保留结果，不自动重试或根据 AP 改条件。Windows 隐藏守护仅防自动系统休眠、允许屏幕熄灭，完成/失败后释放。

当前输出：`runs/initialization-intervention-full-20261021-r2/`；首轮失败产物保留在不带r2的目录。各任务保留 progress、verification、三条件 COCO 预测/官方 AP/覆盖汇总、各层300框、Top100分数组分、Mask几何与逐GT得失。密集上游框已在旧全量诊断中保存，三条件不重复长期保存这些不变的大数组。

2026-10-21 15:52（北京时间）已冻结源码与资产并启动独立队列，WSL控制器PID4519；Windows隐藏防休眠PID40744。24项CPU验收通过。具体GPU验收和全量进度以运行目录state/queue/verification为准，未把排队视为运行完成。

首轮在Baseline Smoke第二张(id=1)被过严的新增logit有限性检查阻止，未释放全量，也未修改模型或实验条件。单图复核确认归一化Encoder/Mask参考框均有限，Encoder中928个坐标为sigmoid边界值。检查改为原生实际使用的sigmoid坐标，保留±inf proposal logits；观察器也修复异常展开时return(None)覆盖原始异常的问题。新增边界logit及原始异常保留回归后25项CPU检查通过，重新冻结r2队列；不覆盖首轮源码、日志及结果。

## 评价与停止边界

主指标 bbox AP50:95，辅助 AP50/75/90/95。几何覆盖在 .5/.75/.9 同时报逐GT边际和一对一最大匹配；相对原生最终300框及 Top100 报恢复、新增丢失、净变化，不能只报道原183个初始化丢失 GT 的恢复。按中心、单/多目标、预先固定相对面积四分位分层。

重复兼容 FP 采用既有官方 COCO greedy 匹配分类；它不是临床误报率。全部126张负图另报固定原生融合分数 .5 下非空检测帧数及检测数量，AP 本身不做分数过滤、NMS 或 TTA。

固定权重下的初始化干预存在训练—推理分布差异；失败不直接否定匹配初始化重训。继续研究仍需稳定的净收益或可由推理期自身信息解释的规律；零散 GT 恢复不能自动支持追加训练，不继续搜索融合系数、回退阈值或手工规则。

任何可训练机制的标签只能来自 Kvasir Train。当前 GT 仅在图像前向返回后用于评价，未参与参考框选择、门控或优化。PolypGen 已参与开发，不能称独立盲测。简单融合属于工程对照，不自动构成第二创新。
