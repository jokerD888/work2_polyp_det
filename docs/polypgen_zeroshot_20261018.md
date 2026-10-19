# Kvasir → PolypGen 单帧零样本框检测

用户提供数据：`G:/ChromeDownload/PolypGen2021_MultiCenterData_v3`。
来源：[官方仓库](https://github.com/DebeshJha/PolypGen)、
[Scientific Data 数据论文](https://www.nature.com/articles/s41597-023-01981-y)。
这里的 zero-shot 指跨数据集直接评估，不是开放词汇检测。

## 冻结的评估条件

- 模型：已完成的 Mask DINO 对照与 MSSL，seed 42。
- 权重：各自已有的 Kvasir val 最佳 bbox AP checkpoint；不根据目标数据选择权重。
- 不在 PolypGen 训练、微调、更新归一化统计或搜索阈值。
- 模型和配置读取训练时冻结的源码，严格加载完整 checkpoint。
- 预处理继承源域推理：800 短边／最大 1333，100 detections；保留原生 Mask 打分，
  不加额外置信度截断、NMS 或 TTA。
- 先与各自已保存的源域单图预测核对，框 atol=0.001 像素、score atol=1e-6。
  推理入口仅传图像，不传目标域 GT。
- COCO bbox AP50:95、AP50、AP75、AR100；pooled 与六中心分别汇报。
  六中心 AP 算术平均另列，不能与 pooled AP 混称。

## 目标数据与标注审计

六个中心的原始单帧共 1537 张，全部保存预测。主指标按已有合法原始框文件覆盖的
1473 张／1533 个框计算，其中 126 张具有空框文件，作为真实负样本保留。

| 中心 | 主评估图像 | 框 | 原始空框负样本 |
|---|---:|---:|---:|
| C1 | 256 | 277 | 5 |
| C2 | 301 | 295 | 32 |
| C3 | 393 | 464 | 1 |
| C4 | 227 | 155 | 81 |
| C5 | 208 | 244 | 2 |
| C6 | 88 | 98 | 5 |

C3 另有 64 张缺框文件且存在 Mask 前景；不将“缺文件”解释成“无息肉”，
也不临时用 Mask 生成框混入原始框基准。这些图像仅导出预测，排除原因逐图记录。
因此主结果应称为 **PolypGen 本地原始框有效标注单帧子集**，不是完整 1537 张标注结果。

C5 发现两对像素完全相同但框不一致的图像：`50000140/50000141` 和
`50000284/50000285`。主结果保留发布标注；另将这四张全部排除，给出 1469 张的
敏感性评估。不依据模型得分决定删图或选择其中一份标注。

原始图像和标注只读，未更改。`imagesAll_positive`、concatenated 副本、叠框图和视频序列
不重复计入这次单帧主评估。视频序列不属于当前指标范围。

### 框坐标

包内 `codes/extract_PolypBoxes.py` 的实际生成路径使用 `regionprops.bbox`，并直接输出
零起点、右／下边排他的 xyxy。尽管文件被描述为 VOC，不机械减一或加一。
转换为 COCO 时使用 `[xmin,ymin,xmax-xmin,ymax-ymin]`，不重算、放缩、裁剪或过滤小框。
检查未发现越界／退化框或同图重复框。

### 与源数据的重合

对全部 1537 张目标单帧与 Kvasir train+val 1000 张比较编码 SHA256、解码 RGB SHA256，
未发现重合；256-bit dHash 距离≤12筛查未发现候选。
这不是患者／视频完全独立的证明；没有患者标识时不能把跨数据集声称为已验证患者隔离。

### 负样本诊断

除不截断分数的标准 COCO AP，附固定 score≥0.5 的负样本误报图像比例与每图误检数。
0.5 在推理前固定，只用于诊断，不用于 COCO AP，也不根据 PolypGen 优化。

## 产物和复现

- [冻结协议](../runs/polypgen-zeroshot-20261018/protocol.json)
- [实时状态](../runs/polypgen-zeroshot-20261018/state.json)
- [最终汇总](../runs/polypgen-zeroshot-20261018/summary.md)（全部完成后生成）
- [CSV 指标](../runs/polypgen-zeroshot-20261018/results_percent.csv)（百分数单位）
- [数据审计](../data/polypgen_zeroshot_stills_v2/audit.json)

各模型目录保存完整 1537 张预测、源域预测一致性检查、严格 checkpoint 加载结果、
各中心／pooled／敏感性指标和负样本诊断。原始框、图像、权重与代码均有 SHA256 记录。
新外部推理／评测源码快照在评估输出 `source/`，原始训练源码及实验目录不修改。

准备命令（Windows 数据环境）：

```powershell
./.venv/Scripts/python.exe -m polypbench.polypgen --source G:/ChromeDownload/PolypGen2021_MultiCenterData_v3 --output data/NEW_DIR --source-annotations data/kvasir_official_bbox_coco/train.json data/kvasir_official_bbox_coco/val.json
```

启动命令（WSL 模型环境，项目根目录）：

```bash
python -m polypbench.zeroshot launch --data data/polypgen_zeroshot_stills_v2 --image-root /mnt/g/ChromeDownload/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3 --output runs/NEW_RUN
```

输出目录拒绝覆盖。后台独立会话按基线、MSSL 顺序运行，完成两组推理和全部评估后退出。
三个数据转换专项测试涵盖坐标边界、多框、空标注、缺框排除与源文件不变性。
