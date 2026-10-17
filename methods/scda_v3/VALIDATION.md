# V3 实现验收（2026-10-17，本机 WSL / RTX 5070）

实现基于研究基准 `31167af`，仅新增 `methods/scda_v3/`。本轮未改写旧方法、baseline配置或原生源码，未安装依赖，未执行optimizer step、短跑或正式训练，未取得V3 AP。

## 已通过

| 检查 | 实测 |
|---|---|
| V3 CPU验收 | **23/23通过**，无跳过，`cuda_context_created=false` |
| 旧方法/仓库回归 | **112/112通过**，无跳过，`cuda_context_created=false` |
| 完整R50参数 | 原生52,017,477；aux新增132,612，总计52,150,089；native/aux参数ID和存储不共享 |
| 初始化及optimizer | 同seed共有state逐位一致，deepcopy不消耗RNG；全部可训练参数恰好入组一次，aux按1e-4优化 |
| V3-A指派 | 固定payload及真实GPU批次的reg token/GT与旧F-Reg-Quality完全一致；仍由原生Raw Box IoU排序 |
| 回归目标 | 相同预测下与旧辅助Box Loss一致，保留proposal先验/坐标缩放及每GT均值 |
| V3-B形状信息 | 同Box、同原生预测、**同mask面积且每cell coverage相同**，只改变像素轮廓即改变选点 |
| SDF/尺度/边界 | 非方形网格、像素中心双线性采样、可见截断边界、零padding、尺寸偏好、空/极小GT、候选缺额 |
| DDP | 两个CPU Gloo rank：某rank零候选、全局零候选，梯度与串行有效GT均值一致；aux参数有合法零梯度 |
| 真实数据 | 128张真实LSJ增强图，与baseline的image/padding/bbox/mask/class逐位一致；单类别映射为0，多框原图仍跳过辅助 |
| 原生损失不变 | 完整DN、FP32/CUDA AMP、真实增强batch1：λ_aux=0与native逐项loss相同；梯度通过既有固定逐参数L2容差，native-native也用相同标准 |
| 推理不变 | 同主模型参数，FP32及CUDA AMP各自与native eval字段逐位一致 |
| 初始aux预测 | deepcopy并同步后，相同memory和proposal先验下，选中token的aux与native raw box预测逐位一致 |
| 梯度路由 | V3-A/B，FP32和CUDA AMP：aux头、memory/encoder、pixel decoder、backbone有非零梯度；原生共享box头、class/mask头及Decoder专属参数无直接aux梯度 |
| 数值 | FP32和CUDA AMP完整前反向有限；选择到非有限memory/head输出会显式报错；空候选及λ=0有有限零梯度 |
| CLI/隔离 | train/predict入口可解析；配置准备保留实际reference数据/模型/协议，拒绝覆盖、错误LR/workers/成熟初始化；保存源码/数据/配置指纹 |

GPU预检峰值PyTorch分配 **2,163,079,680 bytes（约2.01 GiB）**，为batch1验证占用，不是B16训练显存估计。环境为现有Python3.10.12、torch2.7.1+cu128、SciPy1.15.3；正式训练机应复用自身已验收环境。

## 原始证据

- `runs/scda-v3-implementation-20261017/cpu-report-final.json` / `.log`：23项V3测试，逐项结果和源码哈希。
- 同目录 `legacy-regression.json` / `.log`：112项旧方法/协议回归。
- 同目录 `gpu-precheck-1/report.json`：128图mapper、完整DN/FP32/AMP、初始等价、A/B梯度路由及源码哈希。
- 同目录 `gpu-precheck-1/native_repeat_{False,True}.json`、`zero_lambda_{False,True}.json`：全部参数的梯度对比，不隐藏逐元素失败计数。
- `gpu-precheck-1/resolved_V3.yaml`：完整解析后的配置；`optimizer_steps=0`，`training_started=false`。

这些runs产物属于本机证据，不进入Git。GPU验收结束时，全部记录的源码哈希与当前实现一致；上述文档不改变运行源码。

## 数值比较与初始梯度的边界

首次CPU逐元素梯度检查未通过。补做原生自身重复反向发现，同一位置也有舍入差异；固定逐参数L2比较中，CPU native-native与native-V3最大相对L2均约 **1.017e-6**，均在预设 `atol1e-6 + rtol1e-5 × 参数梯度范数` 内。逐项loss仍要求精确相同。

完整GPU测试复用旧预检标准：FP32逐参数L2 `rtol1e-4/atol1e-5`，AMP `rtol5e-3/atol5e-4`。损失差值均为0；native-native及native-V3都保留逐元素舍入差异，不宣称梯度逐位相同。eval另行要求逐位相同，实测通过。

原生最后一层Box MLP全零初始化。初始aux损失对aux头的梯度范数和约0.453，而对共享特征是零，这是预期的链式法则结果。为检查后续梯度路由，仅在一次性预检模型将aux末层权重置为.001：A/B在FP32和AMP下共享路径梯度均非零，且原生box/decoder无直接aux梯度。这不是训练初始化、没有执行optimizer step，也没有保存可被正式run继承的训练checkpoint。

## 未完成、不可据此声称的事项

尚未做训练机A6000的B16显存/步时短跑、正式5500iter训练、AP比较、多seed或外部数据集评测。真实GPU梯度验收使用1个可靠增强训练样本；128图检查指mapper几何与类别对齐，不代表128图都做了梯度验收。

CPU前反向用原生CPU deformable-attention reference fallback及缩减query/layer配置、DN关闭；它与另行完成的**完整CUDA DN/AMP真实图**检查分别记录，不混淆验证范围。

正式开训前应在训练机确认同协议物化配置、检查同名run/GPU进程、用独立run完成B16健康短跑，再从原始ImageNet初始化新run；本轮未启动这些操作。没有任何数据证明V3-A或V3-B优于Clean A，后续仍需按设计完成机制消融。
