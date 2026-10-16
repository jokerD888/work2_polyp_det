# F-Reg-Quality + DCNv4 / Box-Gated IoU（2026-10-16）

## 当前基线和实验进度

研究基线采用 **F-Reg-Quality**：`quality_iou`、λ_cls=0、λ_box=.05；Mask coverage≥.25 是候选资格，预测框-GT IoU 是排序。回归在同一个池内取至多4点，**不再保留旧 F-v2 的 core2+transition2 配额**。分类、Hungarian、DN、mask2box、原生 mask loss 与推理规则保持原样。

训练机报告记录：Clean A s42=65.069 AP，F-Reg-Quality s42=66.143、s43=66.010，Q-BoxOnly s42=64.751；本机未独立读取这些远端原始产物。不能把 s43 与 A s42 当成配对复现，也不能把路径上的增益百分比当作独立机制贡献。历史 batch4 的 D+F/Conv+F 受旧补边管线及协议差异限制，不能代替当前 DCNv4-only。

本轮用户确认只实现 **F-Reg-Quality + DCNv4** 与 **Box-Gated IoU**。没有当前协议 DCNv4-only 结果；后续补齐。本轮未启动正式训练、GPU 校准、真实 DCNv4 CUDA 验收。

## F-Reg-Quality + DCNv4

- 新配置：`methods/proposal_dcnv4/configs/dcn_freg_quality.yaml`。
- 复用现有 S8、256通道残差：`S8 + 0.1 * DCNv4(S8)`；3×3、group4、dilation1、offset_scale1、官方 pointwise projections，算子 FP32。
- 不引入 FCOS/辅助预测头；DCNv4参与训练和推理，SCDA只在训练时生效。
- 新加载器明确接受 F-Reg-Quality（λ_cls=0、λ_box=.05），仍支持旧 F-v2，拒绝把 Q-BoxOnly 或其它未声明变体悄悄混入组合。

## Box-Gated IoU：用户确认的新规则

对相同可靠 GT，使用原生 base-anchor 中心 c 和增强后 GT 的中心 g、宽高 w,h：

```text
g_x - rho*w/2 <= c_x < g_x + rho*w/2
g_y - rho*h/2 <= c_y < g_y + rho*h/2
```

门控额外保留全 GT Box 中心包含规则、有效 token、完整无 padding 单元、Top-Q 排除。之后复用 F-Reg-Quality 的 detached FP32 IoU（量化到1e-4）降序、归一化中心距离升序、level/u/v升序，取至多4点；不补位。仅回归，λ_box=.05，每GT内均值/全局有效GT均值及原生L1/GIoU系数不变。

空间门控只用 Box。为控制监督GT集合，仍保留与参考方法相同的**原始单框且增强后有非空有效 mask**资格判断；这是标注有效性的 Boolean 检查，mask 的像素位置/面积不决定 token 资格或排序。原生 MaskDINO 各组仍使用 mask 监督，不能把本组称为“仅 bbox 标注训练”。诊断模式可在选点确定后读取 mask coverage 作分析，不反馈到门控。

ρ 是一个跨图、GT、尺度层共享的固定数：

1. 用参考 seed 的固定128张 **train** 图、固定LSJ增强、相同 ImageNet R50 初始化，不更新参数。
2. 在完整合法性和 Top-Q 排除后，统计 Mask coverage≥.25 的候选池总数 M，与 full-box 总数 N。
3. 在预先固定的 rho 网格 .01,.02,…,1.00 中，最小化 `abs(total_box_pool-M)`；平局取较小rho。等价于匹配总体保留率 M/N；不使用 val AP、验证图或训练好权重。
4. 保存原始计数、样本ID、数据/源码/初始化哈希和残余差值，之后冻结rho。初始化SHA必须为既定 ImageNet R50。

**只匹配校准样本上的总体候选池数量**，不声称逐图、逐层、最终选点数、梯度范数或整个训练阶段都相同。报告同时给出按层数量、capped选点数和缺额。模型变化导致Top-Q改变时也不重新拟合rho。该消融使用了全局训练mask统计来校准ρ，因此不是“从未利用任何mask信息”的方案；运行时空间门控不读各GT形状。

新配置 `methods/scda_proposal/configs/kvasir_box_gated_iou.yaml` 必须绑定校准JSON；无默认ρ、禁止YAML直接填写ρ绕过证据。数据、源码或协议身份变化后旧校准会被拒绝。证据嵌入 experiment.json，并另存到run，参与配置指纹。

## 在训练机准备配置

先核对空闲GPU和当前进程。以下仅是执行入口，不表示本机已经运行。进入真正的项目根目录（这台训练机是 `/home/ouc1/code/work2_polyp_det/work2_polyp_det`），激活原 `maskdino` 环境。

优先从已经完成的 F-Reg-Quality s42 **真实 experiment.json** 冻结新链，保留它的实际数据路径、worker数、优化器、B16等全部配方，而不是照搬 Windows 路径或重新猜参数：

```bash
python scripts/prepare_quality_experiments.py \
  --reference-experiment runs/scda-freg-quality-pad0-a6000-b16-s42/experiment.json \
  --output runs/_launcher/quality-boxgate-a6000-s42
```

该脚本只写新配置，不启动任何进程；已有目标目录会拒绝覆盖。会生成 reference_base/reference_quality、dcn_quality（GPU0）、box_base/box_gated_iou（GPU1）及准备记录。解析配置的 runtime.visible_gpus 会覆盖外层环境变量，必须核对实际PID与物理GPU。

固定128图校准，**只有空闲GPU上才执行**：

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/calibrate_scda_box_gate.py \
  --config runs/_launcher/quality-boxgate-a6000-s42/reference_quality.yaml \
  --output runs/_launcher/quality-boxgate-a6000-s42/calibration.json \
  --samples 128 --batch-size 4 --allow-gpu
```

此处batch4只是无梯度校准前向的显存选择；正式训练仍是B16。校准不创建optimizer，不调用val evaluator，不加载成熟checkpoint。

完成真实GPU一致性、辅助梯度与独立短跑验收后，正式训练入口为：

```bash
# GPU0：组合。使用可靠后台调度；不要让普通终端退出中断实验。
python -m methods.proposal_dcnv4.run \
  --config runs/_launcher/quality-boxgate-a6000-s42/dcn_quality.yaml \
  --mode train --run-id proposal-dcn-freg-quality-pad0-a6000-b16-s42

# GPU1：新空间门控。先审阅真实校准JSON及数量差异。
python -m methods.scda_proposal.run \
  --config runs/_launcher/quality-boxgate-a6000-s42/box_gated_iou.yaml \
  --mode train --run-id scda-boxgate-iou-pad0-a6000-b16-s42
```

每组完成后用同一入口/配置/run-id、`--mode predict --split val` 预测best，并统一COCO bbox评测。完整B16日程：100ep/5500iter，LR4e-4，warmup125，steps4400/4950，每275步验证选best。两组均从相同ImageNet初始化，不能续训旧模型；启动前同名产物/进程检查、源码配置冻结、状态/PID与自动预测链仍需由训练机执行者落实。

## 本机验证与限制

CPU回归入口：

```bash
CUDA_VISIBLE_DEVICES= python -B -m methods.scda_proposal.check_quality_cpu \
  --output runs/quality-boxgate-cpu/report.json
```

本机最终实测：**103项全部通过，cuda_context_created=false**。原始JSON及逐项日志在 `runs/quality-boxgate-implementation-20261016/cpu-report-final.json` / `.log`。首次101项检查有两处新快照接口错误，失败报告保留；修复后101项通过，补齐实际参考配置准备检查后最终103项通过。

测试覆盖旧方法回归、rho几何及匹配目标、Top-Q/部分padding/空GT/可靠性、IoU排序、mask形状不变性及零分类梯度、实CPU整网初始化/参数/optimizer一致性、配置继承、校准身份拒绝和快照。

DCNv4 CPU 路由使用明确的 TokenProbe 测试替身；不等同于官方DCNv4 CUDA验收，也不能据此提供真实ρ或AP。旧precheck原先强制双分支非零梯度、分类集合包含于回归集合；现在按实际lambda与策略检查，未放宽活跃分支的有限/非零要求。

另修正了通用多GT API的一处错误：IoU排序不再误用上一循环最后一个GT框。当前可靠原始单框策略下历史有效GT只有一个，这项修复不用于解释既有AP收益。Box-Gated不擅自扩展可靠多GT owner策略，遇到这类未确认输入会报错。

首次整套CPU回归的两处快照接口错误已保留并修复，记录位于 `runs/quality-boxgate-implementation-20261016/`；最终测试结果以该目录最新报告为准。
