# 实例掩码 LSJ 补边修复（2026-10-15）

## 错误与修复范围

Detectron2 固定版本的 `FixedSizeCrop` 默认 `seg_pad_value=255`，用于语义分割的 ignore 标签。项目使用 RLE 实例掩码，变换后由 `BitMasks` 转成 bool，因此补边 255 被当成前景。同时，对全 1 有效图做相同变换再按 bool 取反，会把这些补边误认为有效图像。

原生 `BitmaskLSJMapper` 和 `SCDALSJMapper` 都存在此错误。现统一调用 `polypbench/adapters/instance_lsj.py::build_instance_lsj_augmentations`，在新建的原生增强对象上明确设置 `FixedSizeCrop.seg_pad_value=0`。有效图使用 uint8 的 0/1 表示。

以下内容保持原样：图像 resize/flip/crop/pad 的几何与像素补边值、权威检测框及过滤规则、类别、数据划分、模型结构、SCDA 分配规则和损失权重、推理和评测路径。没有修改 `third_party/MaskDINO` 或 Detectron2。

此修复自动覆盖原生、SCDA、DCNv4-only、DCNv4+F、Conv-only、Conv+F 的相应训练 mapper。三个 runner 的源码哈希和快照均加入共享构造函数文件。旧预检的源码哈希不再匹配，不能直接沿用其验收身份。

## 验证

所有验证仅用 CPU，没有训练、完整检测器前向或占用 GPU；Adapter 测试包含 CPU 模块前向和反向。

- 项目测试：`python -B -m unittest discover -s tests -v`，22 项通过。
- SCDA 与 Adapter CPU 测试：`python -B -m unittest methods.scda_proposal.tests.test_cpu methods.proposal_dcnv4.tests.test_cpu methods.proposal_dcnv4.tests.test_conv_control -v`，46 项通过。DCNv4 CPU 测试包含替身算子，不是 CUDA 扩展验收。
- mapper 的 8 项测试覆盖缩小补边、横/竖非正方形、多实例、空实例、触边掩码、放大裁剪、权威框保留、SCDA 跨补边单元排除和源码快照依赖。
- 新增可重放脚本 `scripts/check_lsj_instance_padding.py`，默认取训练集 seed42 的固定 128 图，用原生变换的全 1 标签图独立识别真实补边，以相同 RNG 对照旧管线与修复后 A/F。

实图输出见 [CPU 审计 JSON](checks/lsj_instance_padding_cpu_20261015.json)：

| 项目 | 结果 |
|---|---:|
| 抽查图像 | 128 |
| 真实存在补边的图像 | 77 |
| 旧管线标出补边的图像 | 0 |
| 真实补边像素，逐图累计 | 7,903,200 |
| 旧掩码中的补边前景，逐实例累计 | 8,224,878 |
| 旧前景像素，逐实例累计 | 13,174,932 |
| 修复后前景像素，逐实例累计 | 4,950,054 |

程序断言全部通过：补边位置与独立几何参考一致；修复后每个实例在补边上的前景均为零；A/F 的图像、padding、框、类别与掩码逐位一致；与旧管线比较，图像、检测框、类别以及非补边位置的掩码逐位不变。两个像素统计分母不同：多实例图的同一补边位置会在多个实例掩码中重复计数。

Linux／WSL 模型环境中复核：

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -B scripts/check_lsj_instance_padding.py \
  --config methods/scda_proposal/configs/kvasir_v2.yaml \
  --samples 128 --seed 42 --output runs/lsj-padding-audit-new.json
```

脚本拒绝覆盖已有输出；使用本机 Kvasir 训练图和标注，不下载数据。输出记录数据和实现哈希。扩大抽查可调整 `--samples`；当前交付只声称上述 128 图验证。

## 历史实验与下一轮边界

- 本机 `maskdino-official-s42` 与 `scda-proposal-v2-kvasir-s42` 的训练 mapper 快照已核对，与出错版本一致。历史 AP、日志、权重、预测和源码快照保留，未覆盖。
- 67.5271 / 65.7202 AP 是旧掩码管线下的历史结果，不是修复后结果。不能据此推算修复后的 AP，也不能断言补边错误解释了 F−A 的全部差距。
- 远端 DCNv4/Conv 的原始训练产物仍未同步，本机不能独立确认其具体污染量；若使用同一旧代码和依赖，也会经过此错误路径，应在远端复核。
- 新训练必须使用新的 run-id，例如 `maskdino-official-pad0-s42`、`scda-v2-pad0-s42`。A 和 F 均从原约定初始化重新训练；不要只修 F 后与旧 A 比，也不要从旧错误监督下的 checkpoint 续训来替代新对照。
- runner 会拒绝用变化后的源码身份直接预测旧 run；需要重评旧实验时应恢复旧代码/快照。未放宽该保护。
- 本轮未启动任何训练或 GPU 冒烟。修复本身已经完成 CPU 验证；F 是否提点仍需后续受控训练回答。
