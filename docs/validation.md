# 本机验证摘要

验证日期：2026-10-14（Asia/Shanghai）。运行环境为 Windows 的 Ubuntu-22.04 WSL2、RTX 5070 12GB、Python 3.10.12、PyTorch 2.7.1+cu128、torchvision 0.22.1、CUDA Toolkit 12.8。模型环境依赖检查通过。

- 原始 bbox JSON 与转换后的全部坐标逐项一致，源文件哈希保持不变；880/120 图，941/130 个框，共 1,071。校验记录保存在 `data/kvasir_official_bbox_coco/bbox_verification.json`。
- 训练 mapper 保留官方 bbox 的几何变换结果，辅助 mask 不覆盖框标签；测试覆盖框与 mask 范围不一致、极小官方框不被过滤、原文件只读。
- CUDA 算子前向与输入/采样位置/注意力权重梯度，同 PyTorch 参考实现比较通过。
- 使用当前官方框配置，512、global batch 4、AMP，完成 2-step GPU 训练、两次完整 120 图验证、最佳权重加载和统一评测。日志训练峰值 `7146MiB`，约 7.0GiB；仅代表这次短试跑。
- 最佳 checkpoint 文件哈希与推理实际加载文件一致，统一 AP 与原生选权重时的 AP 对齐后相同。
- Linux 模型环境 15 项测试通过；Windows 14 项通过，1 项因需要模型环境而跳过。

短试跑权重、日志、预测和旧标注数据已按用户要求删除。保留这份验证摘要与环境版本记录；后续正式实验保存在 `runs/`。

以上预检不代表模型收敛或论文性能结果。正式 100-epoch 训练 `maskdino-official-s42` 已于 2026-10-14 22:12（Asia/Shanghai）在本机 WSL GPU 启动；实时进度见对应 `runs/` 目录。完成后后台流水线自动使用最佳权重做 val 推理、统一评测和 CSV 汇总。独立 Linux 训练服务器尚未部署。
