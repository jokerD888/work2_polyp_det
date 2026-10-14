# PolypBench

跨仓库目标检测实验入口。目前已接入官方 MaskDINO，统一配置、运行记录、COCO bbox 评测和结果汇总。

首轮使用 Kvasir-SEG 官方 **880/120 train/val** 划分，以及原始 `kavsir_bboxes.json` 的 **1,071 个框**（train 941，val 130）。框仅转换为 COCO xywh 格式，保留原始坐标与数量；训练时不从 mask 重算 bbox 标签。val 用于权重选择和 val AP 汇报。

## 目录

```text
configs/       正式实验配置，目前保留 maskdino_kvasir_official.yaml
data/          官方划分和原始 bbox 转换后的 COCO 数据
docs/          实验协议、验证摘要
env/           本机依赖版本记录
methods/       自研方法，目前包含 F-v1 MIPS-SCDA
polypbench/    运行、数据校验、模型适配、评测与可视化代码
runs/          正式实验输出及后台任务控制文件
scripts/       Linux/WSL 安装、启动与预检工具
tests/         在系统临时目录生成测试数据的测试
third_party/   固定版本的 MaskDINO 官方源码和许可证
.venv/         Windows 数据处理、评测环境
```

参数与跨方法比较约定见 [实验协议](docs/experiment_protocol.md)，本机验证结果见 [验证摘要](docs/validation.md)。调试日志、旧数据转换、预检配置和短试跑权重已清理。

F-v1 MIPS-SCDA 的实现和独立入口见 [方法说明](methods/scda_proposal/README.md)。2026-10-15 已通过 CPU/Gloo、A/F参数对齐、真实GPU和50iter健康检查；发现回归阈值放宽没有改变当前样本的选点，正式长训等待方法定义决策。沿用 Kvasir 配置，SCDA 仅使用原始单框图片，详见 [预检记录](methods/scda_proposal/F_PRECHECK.md)。

Git 仓库保存源码、配置、文档和依赖记录；`data/`、`weights/`、`runs/`、虚拟环境及 `third_party/` 均不提交。克隆后按下文重新准备数据和模型环境，安装脚本会获取固定版本的 MaskDINO 源码并应用兼容补丁。

## 本机 Windows 运行

训练通过已有 Ubuntu-22.04 WSL2 使用本机 RTX 5070。模型环境为 `~/.venvs/polypbench-maskdino`，Python 3.10、PyTorch 2.7.1/cu128、torchvision 0.22.1，CUDA Toolkit 12.8。

在项目根目录运行 PowerShell：

```powershell
# 校验图片、官方框坐标、RLE、划分和训练预算
./.venv/Scripts/python.exe -m polypbench validate --config configs/maskdino_kvasir_official.yaml

# 检查 GPU，以及 CUDA 算子的前向和反向
./scripts/run_wsl.ps1 doctor --config configs/maskdino_kvasir_official.yaml

# 正式配置：R50、512 输入、总 batch 4、100 epochs、seed 42
./scripts/run_wsl.ps1 train --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42
./scripts/run_wsl.ps1 predict --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42 --split val
./scripts/run_wsl.ps1 summarize
```

预测默认加载最高 val AP50:95 的 `model_best.pth`；保留 final 和最近两份常规 checkpoint。重新训练使用新的 run-id，已有实验不会覆盖。当前没有断点恢复入口。

WSL 的配置使用项目相对路径或 Linux 路径。WSL 发行版在 `scripts/run_wsl.ps1` 设置，模型环境可通过 `POLYPBENCH_ENV` 修改。

## 数据转换与安装

当前 `data/kvasir_official_bbox_coco` 已准备好，原始数据在 `D:\dataset\kvasir-seg\Kvasir-SEG`，转换过程中只读。重新准备项目时：

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[data]"
./.venv/Scripts/python.exe -m polypbench convert-kvasir --source D:/dataset/kvasir-seg/Kvasir-SEG --split-dir D:/dataset/kvasir-seg/Kvasir-SEG/Data-split --bbox-source provided --output data/kvasir_official_bbox_coco
```

转换不会覆盖已有输出。源 bbox JSON 的副本和哈希、划分文件、转换规则、坐标校验结果保存在转换目录。辅助 mask 是原始语义 mask 在对应 bbox 内的前景，RLE 保留孔洞；它不决定检测标签。MaskDINO 使用 bbox+mask，跨模型表格需注明监督条件。

首次创建模型环境（本机已有环境可跳过）：

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/d/code/work2_polypy_det/scripts/bootstrap_maskdino.sh
```

脚本要求已有 Python venv、g++ 和 CUDA Toolkit 12.8；在隔离环境安装依赖、获取固定源码、编译 CUDA 扩展并运行 doctor。环境版本记录在 `env/requirements-maskdino-local.lock.txt`；包含本机路径，用于审计，迁移时使用安装脚本重新构建。

Linux 服务器复制代码与 `data/kvasir_official_bbox_coco` 后运行：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
bash scripts/bootstrap_maskdino.sh
source ~/.venvs/polypbench-maskdino/bin/activate
python -m polypbench validate --config configs/maskdino_kvasir_official.yaml
python -m polypbench train --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42
python -m polypbench predict --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42 --split val
```

双卡运行时修改 `runtime.num_gpus: 2`、`visible_gpus: "0,1"`，总 batch 4 保持不变。不同 GPU 架构重新编译 CUDA 扩展。

## 预检、测试与产物

需要重新验证显存时，在输出目录生成独立的短配置：

```powershell
./.venv/Scripts/python.exe scripts/make_probe_config.py --config configs/maskdino_kvasir_official.yaml --output runs/preflight/config.yaml --iterations 2
./scripts/run_wsl.ps1 train --config runs/preflight/config.yaml --run-id preflight-check
./.venv/Scripts/python.exe -B -m unittest discover -s tests -v
```

正式产物保存到 `runs/<run-id>/`：配置、数据与代码哈希/快照、环境、日志、checkpoint、最佳权重选择记录、COCO 预测和统一指标。`summarize` 读取完成的预测结果生成 `runs/results.csv`。没有已完成的实验时不会生成汇总表。

评测使用所有 GT 图片、原始类别 ID、原图坐标、bbox AP50:95、`maxDets=[1,10,100]`；不额外增加置信度过滤、NMS 或 TTA。JSON 指标为 0–1，CSV 转为百分数。预测可脱离模型环境使用 `evaluate` 重算；`preview` 可生成 GT 与预测框对照图，参数见 `python -m polypbench --help`。

## 上游

- [MaskDINO](https://github.com/IDEA-Research/MaskDINO)，固定提交 `3831d8514a3728535ace8d4ecc7d28044c42dd14`，保留上游 Apache-2.0 许可证。
- [Detectron2](https://github.com/facebookresearch/detectron2)，固定提交 `1e3e13bbf607b54f62205c4c33922521822fb298`。
- `scripts/patch_maskdino_ops.py` 仅替换旧 ATen dispatch API，适配 PyTorch 2.7；数学计算保持原样。bbox-preserving RLE mapper 和 checkpoint 加载兼容处理位于模型适配层。
