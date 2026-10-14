# F-v1 MIPS-SCDA

基于当前固定版本 MaskDINO 的训练期 Proposal 辅助监督。方法放在本目录；通过新的 registry 名称接入，不修改 `third_party/MaskDINO` 或现有基线适配器。

**当前状态：仅完成代码编写，未执行测试、模型导入、前向、反向、推理或训练。不能把本目录视为已通过工程验收。** 预检状态见 [F_PRECHECK.md](F_PRECHECK.md)。原始设计文档保存在 [docs/DESIGN_SPEC.md](docs/DESIGN_SPEC.md)，其中的数据集和验收安排以本次用户确认及预检记录为准。

## 本次确认的实现

- 数据、官方 bbox、划分、预训练、优化器、增强、训练预算及 best-val-AP 规则继承当前 Kvasir 基线配置。没有接入 PolypDB/PolypGen，没有采用文档中的历史实验数值。
- SCDA 只使用**增强前只有一个非 crowd bbox 的图片**对应的裁剪语义 mask。原始多框图片即使裁剪后只剩一框，也全部跳过 SCDA；这些图片仍参与全部原生 MaskDINO 损失。
- 以实际多尺度 feature grid 对应的 padded 像素矩形作为 token 单元，按分段常值像素精确积分 foreground 面积。任何与 padding 像素相交的单元都被排除。
- 中心距离为 `((anchor_cx-gt_cx)/gt_w)^2 + ((anchor_cy-gt_cy)/gt_h)^2`。排序为覆盖率降序、距离升序、`(level,u=x,v=y,GT id)`；先统一 owner，再选两个任务的候选。
- 使用原生 pre-TopQ 类别和框输出，排除原生 Top-Q；不改变其排序、Hungarian、intermediate loss、DN、mask2box、decoder、mask loss 或推理评分。
- 分类仅监督真实 GT 类别的 positive BCE；回归使用完整 bbox、FP32 L1/GIoU，复用 resolved native `BOX_WEIGHT/GIOU_WEIGHT`。分别进行每 GT 内均值和全局有效 GT 均值，最后各乘 `0.05`，在原生加权循环**之后**加入损失。
- 不创建辅助预测层、`nn.Parameter` 或 buffer。参数和 optimizer 对齐仍需后续实际验证。`TWO_STAGE=False`、softmax CE 配置和不匹配的原生源码会明确报错，不自动改配置。

## 文件

| 文件 | 用途 |
|---|---|
| `decoder.py` / `payload.py` | 独立 Decoder 派生类；训练开启时暴露原生 pre-TopQ 输出 |
| `model.py` / `integration.py` | 原生模型派生类与显式注册；辅助 loss 在原生加权之后接入 |
| `mapper.py` / `structures.py` | 相同 LSJ、bbox 变换及过滤；携带可靠性、稳定 GT id 和 padding |
| `scda_assigner.py` | 纯函数：精确覆盖率、合法性、统一归属、确定性 capped 候选 |
| `scda_loss.py` | 正类 BCE、框损失、空集零梯度路径、DDP 归一化 |
| `diagnostics.py` | 候选统计、原始框与真实 decoder 初始化覆盖率、参数和梯度审计接口 |
| `config.py` / `experiment.py` | 原生配置扩展及基线配置继承 |
| `worker.py` / `run.py` | 独立进程与实验入口、源码快照、checkpoint、统一 bbox 评测 |
| `configs/kvasir.yaml` | 本次确认的 F 配置 |

## 配置与环境

`SCDAOptions` 和 `add_scda_config()` 的 `ENABLED` 默认值为 **False**。`configs/kvasir.yaml` 是明确开启 SCDA 的 F 预设；将其 `enabled` 改为 `false` 时，入口保留原生模型/Decoder 名称并采用现有基线 mapper。

当前 F 配置仅继承 `../../../configs/maskdino_kvasir_official.yaml`。它不会直接读取基线正在更新的 checkpoint，也不会恢复/覆盖基线训练。将来启动时会建立新的 run-id，并记录 resolved experiment、基线配置 SHA、方法源文件及数据/仓库快照。推理时校验相同配置、源码与数据。

本机继续使用 **Windows 项目目录 + WSL Linux 模型环境**。正式 Linux 训练机使用相同项目结构和对应已配置的 MaskDINO 环境；路径在运行所在系统内解析。方法没有新增 Python/CUDA 依赖，不安装或编译新算子。

从项目根目录以 source-tree 方式运行；现有 `polypbench` setuptools 包暂未打包 `methods`，所以只复制已安装的 `polypbench` 包不足以运行本方法，应同步完整项目源码。

## 后续入口（本次均未执行）

以下命令供后续完成预检和获准训练后使用。执行前须在 WSL 或 Linux 激活已有 MaskDINO 环境，使 `python` 指向该环境，并进入项目根目录。

```bash
python -m methods.scda_proposal.run \
  --config methods/scda_proposal/configs/kvasir.yaml \
  --mode train --run-id scda-proposal-kvasir-s42

python -m methods.scda_proposal.run \
  --config methods/scda_proposal/configs/kvasir.yaml \
  --mode predict --split val --run-id scda-proposal-kvasir-s42
```

入口明确使用 `methods.scda_proposal.run`；现有 `python -m polypbench train` 仍指向原生基线，不会自动发现本方法。输出为 `runs/<新 run-id>/`，保留 best/final 权重、bbox 预测和 `metrics.json`。同一 run-id 已存在时拒绝重建训练目录。

## 诊断与后续验收

`diagnostics: true` 时，只在训练 forward 中额外采集数据，逐步写入 EventStorage 与 `train/scda_audit.rank<N>.jsonl`。分类/回归每 GT 候选数、配额是否填满、面积、fill ratio、source、所选 level、冲突数量和 Top-Q 原始框/实际初始化框指标分开记录。没有候选的 GT 不被补出正样本。

Top-Q 冗余按已确认的定义报告：IoU≥0.3 的逐 GT 候选数量；与至少一个 GT 匹配的 Top-Q 占比；仅为诊断将匹配候选归给最大 IoU 的 GT（平局按固定 GT 顺序），再计算 `sum(max(owned_count_j-1,0))/Q`。诊断归属不进入模型或 SCDA 分配。

`parameter_audit(model, optimizer)` 输出去重参数数、冻结状态、state_dict shapes 和 optimizer 组。`gradient_audit(model, native_losses, auxiliary, payload)` 通过 `autograd.grad(..., retain_graph=True)` 输出原生与辅助 cls/box 的范数、比例和 cosine；参数空间与 `output_memory` 特征空间分开，包含有/无 lambda 两种尺度，不写入 `.grad`。

梯度诊断默认不执行。授权预检时可在 `diagnostics: true` 下向派生模型的 `scda_audit_callback` 赋值，签名为 `(model, native_losses, auxiliary, payload, targets, assignments)`，在回调内调用 `gradient_audit` 并保存结果。它不应在正式长训里逐步执行。

实际参数对齐、关闭开关/eval 一致性、真实增强后的坐标与覆盖案例、空/多 GT、极小目标、AMP 数值和多卡归一化仍待验收。本次没有生成 GO 结论、正式训练结果或消融决策。文档中的“显著比例”“严重干扰”没有数值定义；后续验收时需要先确认阈值，本实现不替用户设定。
