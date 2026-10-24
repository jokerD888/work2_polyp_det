# 训练机同步补充：研究分支、累积训练器及受限网络权重

研究代码位于 **`codex/candidate-stage-diagnostics`**，默认`main`不包含所需MSSL目录和冻结证据。2026-10-24已检查两端远程分支；请显式取得研究分支，而不是仅对main执行pull。

- Gitee：<https://gitee.com/JokerMao/work2_polyp_det>
- GitHub：<https://github.com/jokerD888/work2_polyp_det>

## 安全取得代码

训练机先查看`git status`。若当前工程已有修改/正在被任务使用，推荐在现有Git仓库内执行：

```bash
git fetch https://gitee.com/JokerMao/work2_polyp_det.git codex/candidate-stage-diagnostics
git worktree add ../work2_polyp_det-comparison-20261024 FETCH_HEAD
```

这是独立的detached checkout，不会覆盖现有修改；目标目录若已经存在，应使用新的明确目录。若没有本地Git仓库，可直接：

```bash
git clone --branch codex/candidate-stage-diagnostics --single-branch \
  https://gitee.com/JokerMao/work2_polyp_det.git work2_polyp_det-comparison-20261024
```

在新checkout中检查：

```bash
git log -1 --oneline
git ls-files methods/mssl repro/mssl_v1_20261020
python scripts/archive_mssl_v1.py verify
```

应存在：

- `methods/mssl/accumulation.py`、`data.py`、`tests/`
- `repro/mssl_v1_20261020/`（含三个历史源码ZIP及证据）
- `docs/mssl_v1_release_20261020.md`
- `docs/mssl_v1_method_20261019.md`
- `docs/comparison_training_machine_prompt_20261024.md`

`data/`、`weights/`、`runs/`、`third_party/`和环境在gitignore中，拉代码不会得到这些大文件。新worktree使用训练机已有数据/MaskDINO的显式绝对路径或经过检查的链接，不覆盖或挪走原任务资产。

## 关于C1累积训练器

现在提供的是同一个项目内的真实MSSL工具源码，不需要另一份“MSSL仓库”。优先直接复用`methods.mssl.accumulation.AccumulatingAMPTrainer`，保持microbatch1×累积16、2750计划迭代和计划迭代驱动scheduler的语义。当前文件在原训练语义上增加日志，精确历史文件在`repro/mssl_v1_20261020/sources/initial.zip`中。

C1仍由作者DI-MaskDINO模型和criterion计算Loss；导入累积工具并不启用MSSL。不要用MSSL配置入口替代DI架构，不新增MSSL或local contrast Loss。不同Detectron2/Torch接口允许有证据的适配，保存patch并验证平均梯度、step频率、AMP跳步、恢复及图像独立预测。

Mask DINO原来的训练数据政策也已提供：原始单框用完整语义Mask，原始多框保留转换辅助Mask，Box来自官方提供框；不能因为目录未拉到而另建一套标签政策。

## 关于受限网络权重

两份具体资产都有无需Hugging Face的官方入口：

| 方法/资产 | 官方下载地址 | 已核查依据 |
|---|---|---|
| D-FINE HGNetv2-B2 | [PPHGNetV2_B2_stage1.pth](https://github.com/Peterande/storage/releases/download/dfinev1.0/PPHGNetV2_B2_stage1.pth) | [D-FINE骨干源码](https://github.com/Peterande/D-FINE/blob/master/src/nn/backbone/hgnetv2.py)，GitHub Release元数据列有该文件，大小24,362,501字节 |
| ECC TorchVision R50 | [resnet50-11ad3fa6.pth](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) | [ECC加载代码](https://github.com/yuncheng97/ECC-PolypDet/blob/main/resnet.py)、[TorchVision上游](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)，对应ImageNet1K V2 |

本次核实了源码指向和HGNet Release资产元数据，没有声称已在训练机成功下载，也没有编造完整文件SHA。训练机应完成真实下载、文件类型/完整性验证，登记SHA后加载。

若实际下载仍因域名重定向失败，允许由能联网机器取得同一官方原文件并离线复制；双端SHA必须一致。官方另一入口也可使用，必须验证资产身份。只改封装/键名需要逐张量数值一致证据。

不能静默换骨干、换随机初始化、换COCO/Objects365/息肉任务权重；Detectron2的R50转换pkl也不是HGNetv2或ECC TorchVision文件的任意替身。ECC V2与现有Mask DINO转换pkl的具体初始化差异需公开记录，不能只因都叫R50就写为相同权重。

无法确认资产时等待该方法的权重，其他独立实现继续。完整任务仍以更新后的执行Prompt为准；其长训开关默认保持false。
