# SCDA V3: independent auxiliary regression + mask-aware assignment

本目录基于 `31167af` 的研究现状独立实现。旧 `scda_proposal/`、`proposal_dcnv4/` 和原生 MaskDINO 均不改写。本轮不引入 DCNv4 或分类辅助，不自动启动正式训练。

## 两个版本

| 项目 | V3-A | V3-B |
|---|---|---|
| `assignment_policy` | `legacy_quality_iou` | `mask_multiscale` |
| 辅助头 | 原生 `_bbox_embed` 的独立 deepcopy，三层 MLP | 相同 |
| 回归参数化 | `sigmoid(aux(memory) + native_unsigmoid_proposal)` | 相同 |
| 回归监督 | `.05 × 每GT均值/全局有效GT均值 × (5 L1 + 2 GIoU)` | 相同 |
| 候选合法性 | 可靠原始单框、非空有效mask、完整非padding单元、base-anchor中心在GT框内、coverage≥.25、排除Top-Q | 相同 |
| 指派 | **原生 Raw Box** IoU Top4，量化1e-4及旧tie-break | SDF分区 + mask尺度选层 + 原生Raw Box IoU |

Aux MLP只前向被选中的token。选点不读取aux预测。主模型参数初始化及RNG不因deepcopy改变；ImageNet权重加载完成后、首次训练step前再次同步aux头，预测/载入训练checkpoint时不执行同步。

新增 **132,612** 个参数，默认基础 LR=`1e-4`，原生参数52,017,477个，总计52,150,089。Aux分支只在train执行，eval直接继承原生路径；checkpoint保留aux参数，但它们不参与推理。

### 已确认的 V3-B 几何规则

1. 对增强后的mask与有效像素取交集，加一圈零边框，计算 `EDT(mask)-EDT(~mask)`，去除边框并除以 `sqrt(mask面积/π)`。SciPy CPU EDT无梯度、确定性；本机现有环境为SciPy1.15.3，无新增CUDA算子。
2. 在实际padded feature-grid单元的像素中心双线性采样（像素中心为x+.5/y+.5，`align_corners=False`）。裁剪截断处属于**可见mask边界**，不推测裁剪外的形状。任何含padding像素的单元仍排除。
3. `D=2sqrt(mask面积/π)`；每层 `sx=Wpad/Wlevel, sy=Hpad/Hlevel, s=sqrt(sx*sy)`。按 `abs(log2(D/(8s)))` 排序，平局取固定level顺序；取前两个有合法候选的层。
4. 每GT、每层的合法池按SDF升序切半：前半Boundary、后半Core；奇数Boundary多1个，单候选归Boundary。SDF平局保持既有质量/距离/level-u-v稳定次序。
5. 每个优选层分别选Core和Boundary各最多1个，区内按原生IoU降序、距离及level/u/v稳定排序。无重复token。缺额先在两优选层剩余池按IoU补，再从其他合法层补；总候选不足4则保留实际数量。
6. 每步每rank记录GT、token、层、coverage、IoU、SDF、实际几何区域、选取角色、层内/跨层补位及缺额。两个原始GT均不可靠，不能在crop过滤后变成可靠单实例。

Core/Boundary 是**候选内部的相对SDF分区**，不等同于固定物理厚度边界带。Mask coverage和框内约束仍在，不能声称监督了所有轮廓/所有外部背景。尺度8格、二分及配额均为待实验验证的既定设计，不代表最优。

## 配置和启动入口

`configs/base_kvasir.yaml` 显式固定低LR协议，不继承旧的4e-4默认值：seed42、B16、5500iter/100epoch、AdamW、LR1e-4/backbone×.1、wd.05、AMP、full-model clip .01、warmup125/factor.001、steps[4400,4950]、LSJ512/.5–1.5/hflip、workers4、每275步验证、best val AP选权重。学习率衰减节点没有改成另一套日程。

训练机优先从**已完成低LR F-Reg-Quality的实际experiment.json**准备新配置链，保留该机数据路径和实际配方：

```bash
python -m methods.scda_v3.prepare \
  --reference-experiment runs/scda-freg-quality-pad0-a6000-b16-s42-lr1e4/experiment.json \
  --output runs/_launcher/scda-v3-a6000-s42
```

该入口仅生成配置、检查协议，无模型/子进程；拒绝覆盖已有目录，输出V3-A GPU0 / V3-B GPU1。reference的实际run-id请按训练机文件核对；如上述文件不存在，应定位真实低LR F-Reg-Quality产物，不能拿高LR或DCNv4组合替代。其余参数不符会拒绝准备。

CPU验收：

```bash
CUDA_VISIBLE_DEVICES= python -B -m methods.scda_v3.check_cpu \
  --output runs/scda-v3-precheck/cpu.json
```

空闲GPU上进行**无optimizer step**的完整DN、FP32/AMP、真实图验收：

```bash
python -B -m methods.scda_v3.precheck \
  --config runs/_launcher/scda-v3-a6000-s42/v3a.yaml \
  --output runs/scda-v3-precheck/gpu --samples 128 --allow-gpu
```

正式训练接口已实现，以下为后续执行入口，**不表示本轮已经开训**：

```bash
python -m methods.scda_v3.run --config runs/_launcher/scda-v3-a6000-s42/v3a.yaml \
  --mode train --run-id scda-v3a-kvasir-a6000-b16-s42-lr1e4
python -m methods.scda_v3.run --config runs/_launcher/scda-v3-a6000-s42/v3b.yaml \
  --mode train --run-id scda-v3b-kvasir-a6000-b16-s42-lr1e4
```

每组应先完成训练机B16短跑和检查，再由可靠后台pipeline串联train、`--mode predict --split val`及统一汇总。本目录runner拒绝覆盖已有run/stage，保存解析配置、源码/配置哈希、数据验证、checkpoint选择和统一COCO bbox评测。正式run源码冻结后不可边训边改。

## 验收边界与初始化梯度

原生Box MLP末层全零初始化，deepcopy后aux末层也为零；**初始aux损失对memory/encoder/backbone的梯度为零是链式法则的预期结果**。初始aux最后层可以获得梯度。梯度路由验收另在一次性测试模型上把aux末层权重置为非零，验证后续非零权重状态的共享梯度路径；这不是正式初始化，也没有optimizer step或继承到正式run。

CPU全模型构建保留完整R50结构；CPU前反向测试使用较少queries/layers并关闭CUDA专属DN，不能代替完整CUDA DN/AMP验收。所有报告区分验证范围，不把CPU autocast当成CUDA FP16 AMP。

CPU native-native重复反向也有舍入差异。损失独立精确比较，梯度native-native和native-V3使用相同固定逐参数L2容差；不通过放宽单个V3分支标准掩盖差异。

本机实测与未完成事项见 [VALIDATION.md](VALIDATION.md)。当前没有V3 AP结果，不声称独立头、多尺度或Mask几何已带来增益；后续Box-Geometry control和多seed需独立实验。
