# Proposal Residual Adapter：DCNv4 / Conv3×3

**2026-10-16 更新：新增 `configs/dcn_freg_quality.yaml`，组合当前有效的 F-Reg-Quality，而不是旧F-v2。** 仍是现有S8残差Adapter，没有FCOS辅助头。历史训练机已报告旧F-v2组合实验，但其batch4/旧补边结果不能替代当前DCNv4-only。新组合本机只完成CPU接口/整网构建验证，真实CUDA验收和正式训练待训练机执行；详见 [新方法说明](../../docs/scda_quality_dcn_boxgate_20261016.md)。下文的“未执行GPU”等状态属于本机/历史阶段记录，不能概括训练机此前已完成的工作。

**状态（2026-10-15）：代码与 CPU 接口检查已完成；未安装/编译 DCNv4，未执行 GPU 验收或新训练。** 用户明确选择先完成 CPU 阶段，GPU 阶段待正在运行的 F-v2 结束后再进行。

已增加普通 Conv3×3 对照。当前合计 **26 项 CPU 检查通过**，其中 Conv 使用真实 `torch.nn.Conv2d`，不依赖 DCNv4 扩展；DCNv4 部分仍使用明确标注的 CPU 测试替身。新增检查见 `checks/conv_control_cpu_report.json`；首次18项结果与 manifest 作为 `ee1f5fb` 阶段记录保留。

## 实现版本

继承固定版本的 `MaskDINOHead`，在 `pixel_decoder.forward_features()` 和 `predictor()` 之间，仅增强 **S8、256 通道**特征：

```text
Backbone → 原生 Pixel Decoder / MSDeformAttn
         → [S8: F + 0.1 × DCNv4(F), S16, S32, S64]
         → 原生 Predictor / Proposal / Top-Q / Decoder
         → 原生分类、框、掩码输出
```

- DCNv4：3×3、group=4、stride=1、pad=1、dilation=1、offset_scale=1；使用官方默认的 value/output pointwise projection，无额外 depthwise conv 或 center feature scale。
- `operator: conv3x3` 使用普通 `Conv2d(256,256,3,padding=1,groups=1,bias=True)`，weight/bias 均为零初始化，无额外 Norm/激活/投影。它与 DCNv4 使用相同 S8、通道、alpha 和 FP32 执行策略；第一次反向可更新卷积参数。
- `alpha=0.1` 是固定 Python 标量，不增加可学习门控；`alpha=0` 提供完全跳过算子的旁路。
- 在 CPU RNG 隔离上下文初始化新增参数，避免其随机初始化改变后续原生参数或数据增强的随机流。
- 输入从 `[B,C,H,W]` 适配到 `[B,H×W,C]`，显式传入 `(H,W)`；非方形与非连续输入已作 CPU 路由测试。
- DCNv4 参数保持 FP32，并在关闭 autocast 的上下文执行；残差输出恢复到原特征 dtype。禁止直接 `model.half()` 改变本分支的参数精度。
- **DCNv4 在训练和推理中都会执行**。不是零部署开销；F-v2 监督仍只在训练时使用。
- `mask_features` 和其他三个特征对象原样传给 Predictor。掩码/框共享下游参数，因此不能声称掩码预测完全不受影响。
- 当前 pinned source 的实际输出为 `[S8,S16,S32,S64]`，512 画布时是 `[64²,32²,16²,8²]`。实现从 Pixel Decoder 的真实 stride 元数据确定索引，并检查与 Backbone 同 stride 特征的尺寸；不依赖 `low2high` 名字的直觉含义。

没有修改 `third_party/MaskDINO`、原生 Backbone、MSDeformAttn、Hungarian、DN、mask2box 或 `methods/scda_proposal/`。普通卷积必须由 `operator: conv3x3` 或 `--operator conv3x3` 显式选择；DCNv4 缺失仍明确报错，不自动回退到其他算子。`dcnv4_adapter.py` 保持首次实现不变。

## 独立配置与当前真实配方

| 配置 | 残差算子 | F-v2 | 继承 |
|---|---|---|---|
| `configs/dcn_only.yaml` | DCNv4 | 关 | 当前基线 YAML |
| `configs/dcn_fv2.yaml` | DCNv4 | 开 | 已确认的 `scda_proposal/configs/kvasir_v2.yaml` |
| `configs/conv_only.yaml` | Conv3×3，groups=1 | 关 | 当前基线 YAML |
| `configs/conv_fv2.yaml` | Conv3×3，groups=1 | 开 | 已确认的 `scda_proposal/configs/kvasir_v2.yaml` |

配置支持关闭 DCNv4，关闭时保留原生 Head 类名并且不构造 DCNv4。F-v2 使用**同一份** `SCDAOptions`、Mapper、Decoder、Assigner 和 Loss；回归仍为实际分类核心点＋中覆盖率点，不另写或修改规则。

`enabled` 全局默认 False；`operator` 默认 `dcnv4`，旧 YAML 未写该键时保持默认行为。`group: 4` 是 DCNv4 专用字段；Conv3×3 始终 groups=1，本轮没有实现 groups=4 卷积模式。为兼容现有 DCNv4 state_dict，Head/参数前缀继续叫 `ProposalDCNv4Head` / `dcnv4_adapter`；Conv 模式内实际子模块为 `.conv.*`，不得按前缀误判算子。

本地仓库为 `D:/code/work2_polypy_det`，不是此前训练机 B 的 `work2_det`。当前继承的是 Kvasir 官方 880/120、原始框 941/130、512 输入、batch4、seed42、100ep/22000iter；当前权重源为 **ImageNet R50**，不能称作完整 COCO 联合预训练。新目录不改上述配方，也不恢复已有 F-v2 检查点。

`dcnv4` 字段与 `MODEL.PROPOSAL_DCNV4.*` 对应。基础配置、F-v2 配置和数据协议使用原项目解析器读取；原生 overrides、种子、日程保持一致。新增入口：

```bash
# 在准备好的隔离模型环境、项目根目录执行。以下不是本轮已经执行的训练。
python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/dcn_only.yaml \
  --mode train --run-id proposal-dcn-only-kvasir-s42

python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/dcn_fv2.yaml \
  --mode train --run-id proposal-dcn-fv2-kvasir-s42

python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/dcn_fv2.yaml \
  --mode predict --split val --run-id proposal-dcn-fv2-kvasir-s42
```

命令保留项目已有的 best-val-AP 选择、best/final 保存、官方 bbox 评测、禁止覆盖 run-id 及数据/源码指纹校验。源码快照包含本目录及复用的 SCDA 文件。`worker.py`/`run.py` 从现有入口派生为独立副本，原入口未修改；无自动训练或后台启动器。

### 普通卷积配置与命令行覆盖

```bash
# 专用普通卷积配置
python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/conv_fv2.yaml \
  --mode train --run-id proposal-conv-fv2-kvasir-s42

# 或显式覆盖 DCNv4 配置。两种入口使用各自的 run-id。
python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/dcn_only.yaml \
  --operator conv3x3 --mode train --run-id proposal-conv-cli-kvasir-s42

# 预测必须使用训练时相同的配置和覆盖参数；遗漏/错用会提前拒绝。
python -m methods.proposal_dcnv4.run \
  --config methods/proposal_dcnv4/configs/dcn_only.yaml \
  --operator conv3x3 --mode predict --split val --run-id proposal-conv-cli-kvasir-s42
```

实际选择记录在 `experiment.json` 的 `dcnv4.operator` 和 `operator_selection`（配置值、有效值、来源），参与 `config_fingerprint`，并写入 `manifest.json`、`command.json`、模型日志与 `method.json`。模型报告以 `adapter_operator` / `adapter_backend` / `adapter_parameter_count` 标注实际算子；Conv 模式下历史 `dcnv4_parameter_count=0`，避免把普通卷积参数误称 DCNv4。

配置来源也参与指纹。通过命令行启动的实验，预测时应重复同一覆盖参数，不改用另一份名称不同的 YAML。

### 参数与可解释边界

| 算子 | 新增参数 | 证据 |
|---|---:|---|
| 当前官方 DCNv4 | 160,368 | 固定源码静态计算；真实 CUDA 后端待构造核验 |
| Conv3×3，groups=1 | **590,080** | CPU 真实模块及完整模型差值实测 |

这是“固定采样普通卷积增强”对照，**并非参数/FLOPs匹配，也未隔离 DCNv4 内部 pointwise 投影与动态聚合权重等其他差异**。即使 DCNv4 更优，也应表述为“相对该普通卷积 Adapter 的优势”，结合其他证据分析动态采样作用。groups=4 卷积对应147,712参数可作为后续单独方案，当前没有启动或默认添加。

## DCNv4 来源与环境

官方仓库：<https://github.com/OpenGVLab/DCNv4>

固定 commit：`4b848f7dd7da74ff03f7d278f902c6fd05b391b5`。

`backend.py` 检查官方 Python 模块和 Function 的规范化源码 SHA256，并记录实际扩展二进制 SHA256。参数规格的静态推算为 **160,368 个新增参数**；真实算子安装后，由 P0 和模型参数报告实测，CPU 测试替身的参数数不能替代该结果。

目前没有安装该扩展。未来编译应在**单独的 venv/环境**中进行，不降级或向现有 `polypbench-maskdino` 环境安装包。推荐先建立同版 Python/PyTorch/cu128 与 Detectron2 的环境，再在专用依赖缓存中：

```bash
# 仅为未来操作说明；本轮未执行。
git clone https://github.com/OpenGVLab/DCNv4.git /path/to/isolated/DCNv4
git -C /path/to/isolated/DCNv4 checkout 4b848f7dd7da74ff03f7d278f902c6fd05b391b5
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0  # 本机 RTX 5070；其他机器应使用其真实架构
MAX_JOBS=2 python -m pip install --no-build-isolation /path/to/isolated/DCNv4/DCNv4_op
```

这些命令不是已确认可编译的配方。官方历史源码在 PyTorch 2.7.1/CUDA12.8/Blackwell 上的兼容性**待核实**；若需补丁，单独记录源码 diff、编译环境与重验结果，不默默放宽源码身份检查。

## 已运行 CPU 检查

在现有 WSL 模型环境中，只使用 CPU，未安装任何新依赖：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m methods.proposal_dcnv4.check_cpu \
  --output methods/proposal_dcnv4/checks/conv_control_cpu_report.json
```

26 项通过（首次18项＋本轮8项），详见 `VALIDATION.md` 和 JSON。Conv 模式使用真实普通卷积，完整模型构建期间拦截任何 `DCNv4` 官方包导入且全部通过。DCNv4 模式明确注入 `TokenProbe`，仅替代不可用的 CUDA 算子以检查形状、路由、梯度和配置；其他 Head/完整模型/优化器接口均使用真实 MaskDINO/SCDA 实现。**不据此宣称官方算子正确、整网 GPU 等价或训练健康。**

## 待执行的验收

1. **P0**：独立环境编译、真实官方算子 FP32/AMP 前后向、非方形、输入和 offset/mask 权重方向有限差分。
2. **P1**：原生权重加载和遗漏键清单；真实 DCNv4 的模型参数/优化器检查；原生损失与新增参数梯度有限；同一模型 alpha=0 时完整预测/损失与原生对齐；512及非方形特征。
3. **P2**：独立 run-id 的 50–200iter 短跑；记录 loss、梯度、显存、吞吐、Top-Q 覆盖与冗余；比较相同训练阶段，不拿短跑与成熟 best 比较。

训练结束且隔离环境准备好后，P0 有显式 GPU 开关：

```bash
python -m methods.proposal_dcnv4.doctor --allow-gpu \
  --output runs/proposal-dcn-p0/operator.json
```

`doctor` 只实现 P0，尚未经 GPU 执行；P1/P2 仍需在实际环境验收，不能用 P0 代替。本轮不自动等待训练、不编译、不启动这些任务。

后续若进入效果实验，A/F/DCN/DCN+F 是主消融；普通残差卷积+F 是容量/变换对照。形态分组、DCNv4 偏移是否被使用、实际成本均需由实验给出，当前没有 AP 提升结论。

完整待运行矩阵为 A、F、DCN、DCN+F、Conv、Conv+F。重点比较 DCN vs Conv、DCN+F vs Conv+F，以及各架构内 F 的增量；当前只实现并检查入口，没有训练任何新增候选。
