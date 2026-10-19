# MSSL V1：已完成五组实验的证据归档

归档日期：2026-10-20。只整理既有结果，没有训练、推理或修改方法。

统一说明见 [发布索引](../../docs/mssl_v1_release_20261020.md)。

## 内容

- `manifest.json`：五组 run-id、权重路径/哈希、历史源码清单、训练更新数和外部文件指纹。
- `sources/`：三个经过原协议逐文件 SHA256 核对的历史源码 ZIP；不以工作区新版代码代替旧训练版本。
- `models/`：五组实际实验配置、解析配置、选权记录、环境与源域/PolypGen指标。
- `protocols/`：三个原始执行协议的逐字节副本。
- `evidence/`：方法契约、五组结果、配对区间、行为分析、数值/GPU验收与完成状态。
- `results.csv`：两份数据上的五组 AP/AP50/AP75/AP90/AP95；单位为0–100 AP点。
- `checksums.json`：归档内容校验清单；该文件自身由Git版本记录保护。

## 校验

在项目根目录执行（Python 3.10+，仅标准库）：

```powershell
python scripts/archive_mssl_v1.py verify
python scripts/archive_mssl_v1.py verify --local
```

第一条校验已归档文件及ZIP内部源码；第二条还重新读取原位置的权重、预测和证据文件。

权重、完整预测、Bootstrap抽样向量和数据未复制进Git，仍在原来的 `runs/`、`data/` 路径。
保留这些本地目录；仅克隆Git仓库不包含模型权重，也不构成异机备份。

## 源码与模型对应

| 模型 | 源码包 | 选定迭代 | 成功更新数 |
|---|---|---:|---:|
| baseline | initial.zip | 2750 | 2740 |
| mssl | initial.zip | 2750 | 2740 |
| box_cdf | attribution.zip | 2750 | 2740 |
| mask_static | attribution.zip | 2750 | 2740 |
| ciou_replace | ciou.zip | 2750 | 2740 |

所有组按Kvasir Val最大bbox AP50:95选权重，同分取最早；均为Seed42。
源码包包含项目自有实现，官方MaskDINO和Detectron2依赖需按固定版本另行准备。
真实环境、第三方版本和初始化哈希保存在 `models/*/train/`、`evidence/dependencies.json`。
