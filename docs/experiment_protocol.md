# 第一轮检测对比协议

## 数据与汇报

采用用户指定的 Kvasir-SEG `Data-split/train.txt` 和 `val.txt`，880/120 图，保留文件顺序。只有 train/val，没有 test。官方 val 同时用于权重选择和最终 val AP 汇报，不称为独立测试结果。此选择于 2026-10-14 确定。

统一检测标注：直接读取原始 `kavsir_bboxes.json`，只把 `[xmin,ymin,xmax,ymax]` 转为 COCO `[xmin,ymin,xmax-xmin,ymax-ymin]`。输出 `data/kvasir_official_bbox_coco/train.json`、`val.json`，941/130 个框，总数 1,071，坐标逐项与源 JSON 校验。没有重算、过滤或修正官方框。原始 JSON 在转换目录中保留只读副本和校验值；后续检测器使用这些相同的 bbox。

MaskDINO 的辅助监督使用原始语义 mask（灰度 >127）在每个官方 bbox 范围内的前景，并保存为 RLE；重叠 bbox 可共享前景，相关像素数量记录在 `conversion.json`。这是语义 mask 到框对应辅助 mask 的适配规则，不声称数据集额外提供了实例级 mask 真值。`training_boxes: annotations` 保证训练加载后只对官方 bbox 做几何变换，不再从辅助 mask 重算 gt_boxes；bbox 评测面积按框宽×高。先前 mask-derived 标注及其预检仅用于历史联调，不用于正式检测比较。

## MaskDINO 首轮参数

`configs/maskdino_kvasir_official.yaml` 为正式候选配置：

| 参数 | 设置 |
|---|---|
| 模型 | 官方 MaskDINO R50，2048 FFN，300 queries，原生损失/匹配/后处理；mapper 使用官方框监督 |
| 预训练 | ImageNet R50 backbone；不加载 COCO 完整检测器 |
| 监督 | bbox+mask；在跨模型表格单独注明 |
| 训练输入 | 512×512 LSJ 输出；保持比例缩放、裁剪/填充 |
| 训练增强 | 水平翻转，LSJ scale 0.5–1.5；暂不加额外颜色、Mosaic 或 MixUp |
| 评估输入 | 保持比例，最长边不超过 512，模型内部按 stride 填充；TTA 关闭 |
| 总 batch | 4，所有 GPU 合计；AMP 开启 |
| Optimizer | AdamW，学习率 1e-4，backbone 1e-5，weight decay 0.05 |
| Warmup | 500 iterations，linear，factor 0.001 |
| 训练预算 | 100 epochs，220 iterations/epoch，22,000 iterations |
| LR 衰减 | 第 80、90 epoch，倍率各 0.1：17,600 / 19,800 iterations |
| 梯度裁剪 | 原生 full-model L2 norm 0.01 |
| 验证 | 每 5 epochs（1,100 iterations），以及训练末尾 |
| 权重选择 | 最高 val AP50:95，分数相同保留较早 checkpoint；保留 best 和 final |
| 常规 checkpoint | 每 5 epochs，滚动保留最近 2 份；best/final 单独保留 |
| 随机种子 | 首轮 42；完整表建议 42、43、44，报告均值与样本标准差 |
| 主要指标 | bbox AP50:95；附 AP50、AP75、AR100，pycocotools maxDets=[1,10,100] |

这是结合小数据集和本机资源制定的起始方案，不是论文原始 COCO 训练方案，也不保证已达到最佳性能。官方模型配置保持结构、损失、优化器类型和原生后处理；输入分辨率、增强范围、batch 和训练日程为项目设定。

本机已使用当前官方 bbox 完成 2-step 短试跑及完整 val 评估，512、batch 4、AMP 的日志峰值为 `7146MiB`（约 7.0GiB），best 权重保存、加载和统一评测通过。短试跑产物已清理，验证摘要保存在 `docs/validation.md`；100-epoch 长训练尚未启动。

880 / 4 = 220 iterations/epoch。Detectron2 的 loader 持续采样，因此这里的 epoch 表示期望图像曝光次数，不保证每个 epoch 恰好遍历一遍所有图像。切换 Linux 双卡时总 batch 仍为 4（每卡 2），步数和学习率保持；关闭按 GPU 数自动缩放。若改变总 batch，需要一起重新计算 MAX_ITER、STEPS、EVAL_PERIOD 和 training_budget，不能只改 GPU 数或 batch。

## 后续方法如何比较

固定条件：图像划分、目标标注、类别、评估图像全集、输入分辨率协议、预训练数据来源分组、训练预算、验证频率、checkpoint 选择、种子集合和评测器。

各模型分别使用合理的优化器、学习率和调度方式；不强制 YOLO 与 MaskDINO 共用 AdamW/1e-4。首轮给各模型同样 100 epochs 的图像曝光预算，记录总 batch、优化更新次数和训练时间。这是统一预算比较，不能代替各模型充分收敛后的性能比较。若在预算结束仍持续上升，为相关模型设置统一且预先规定的延长预算（例如 200 epochs），单独列为更长训练实验，不只给某一个模型延长并混入原表。

调参只使用 val，对每个方法给相近的搜索次数；不得把单模型多次调参后的最佳值与其他模型未经调参的值描述为严格公平。每 5 epochs 作为可选权重候选，减少不同验证频率导致的选择机会差异。

预训练分组必须记录实际数据来源：ImageNet backbone、COCO 全检测器、其他额外数据不能悄悄视为相同初始化。若后续 YOLO 使用 COCO pretrained，应将本组标注为预训练条件不同的实用性能比较，或者补充匹配预训练来源的实验。

MaskDINO 使用 bbox+mask，纯检测器通常只使用 bbox。跨模型表格记录监督列；你的方法与原始 MaskDINO 的机制消融应保持 bbox+mask、backbone、初始化、增强、batch 和日程一致，才便于归因到方法变化。

## 启动

PowerShell：

```powershell
./scripts/run_wsl.ps1 train --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42
./scripts/run_wsl.ps1 predict --config configs/maskdino_kvasir_official.yaml --run-id maskdino-official-s42 --split val
```

训练结束后默认预测加载 `model_best.pth`。`best_checkpoint.json` 记录选择指标、iteration 和 ties；`selected_checkpoint.json` 记录选择出的权重哈希。最终 val AP 通过统一评测器重新计算，不直接把 loss 当作模型选择依据。

Linux 上激活模型环境后用相同 `python -m polypbench ...` 命令。仅在服务器确有两张可用 GPU 时修改 `runtime.num_gpus: 2`、`runtime.visible_gpus: "0,1"`；总 batch 4 保持不变。
