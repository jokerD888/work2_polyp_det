# 候选阶段诊断：实现、验收与训练机入口

这次只做Baseline/MSSL V1只读阶段追踪和训练标签资格审计。E/F继续使用既有冻结队列，未加入本机GPU任务；未实现或训练局部对比模块。

## 已实现

- `polypbench.analysis.candidate_stage`：观察原生函数返回帧，不替换forward、TopK、mask2box、Matcher或推理分数。导出完整密集池、原生合法性、实际图像有效范围、真实Top300索引、Mask-to-Box原值/实际初始化、9层Decoder框及Top100。
- 原生aux[0]框与实际初始化逐张量一致，aux[1]才是第一层Decoder。空Mask的原生转换是`[1e8,1e8,0,0]`哨兵框，保留原值、边界处理后的实际初始化及空/单行/单列标记，不能用零框替换。
- 逐GT最大IoU及阶段变化、一对一最大基数覆盖和同Query轨迹分别保存。原生完整Mask转换不预先去padding或清碎片；有效图像面积统计仅作观察。
- 每图先普通原生前向、再带观察器前向，框与最终分数必须逐位相等；再核对历史全300框/分类分数及Top100框/融合分数/Query ID。跨GPU历史参照采用`rtol=1e-5`，框`atol=1e-3`像素、分数`atol=1e-6`，Query ID仍要求相同；任何不一致停止队列，不自动放宽。
- `candidate_stage_jobs`：准备前核对固定权重、72图内容、标注、Python/YAML原生源码及bundle哈希；顺序执行两次只读诊断，成功后生成对照报告和失败/恢复示例。
- `supervision_eligibility`：真实训练mapper与固定LSJ，CPU执行；保留原始多框身份限制。stride4为融合Mask特征，当前候选路径为stride8/16/32/64，不能把不同路径的可用区域混称候选监督。

GT只在前向返回后参与统计，不送入模型；没有优化器或训练更新。密集候选池中“原生合法”和“有效图像中心内”另报：原生某些尺寸会关闭padding mask，不能私自过滤后重新选Top300。

## 本机验收边界

7项新增CPU检查通过：真实原生Decoder forward、TopK、mask2box、pred_box及MaskDINO后处理均执行；仅骨干特征与注意力层使用小型测试替身，因此不能替代真实权重GPU验收。检查了前向输出/RNG不变、初始框对应、padding坐标、负图、退化框、匹配与边际覆盖差别、语义Mask不强拆实例，并实际生成临时报告。既有9项质量诊断回归通过。

全部880张训练图、941个GT、3个固定增强种子已做CPU标签审计；结论见`candidate_stage_supervision_report_20261021.md`。原始少量工程重放和第一次未包含stride64的全量结果保留，正式报告采用补全尺度的`lsj-full-v2`，没有改动区域或资格阈值。

本机已通过完整资源预检：两份固定checkpoint、72张原图、历史预测、标注和原生Python/YAML源码匹配。预检只生成job，**没有调用本机GPU前向**。真正的72张×两模型GPU诊断尚须在训练机运行。

## Git内容与额外资源

Git包含观察器、审计程序、队列入口、72张清单、两份精确实验JSON和对应的历史预测NPZ（两份共约480 KB）。bundle按字节保留，跨平台不转换换行。

权重及图像不通过Git提交。训练机必须另有以下两份本机checkpoint，各625,285,006字节；不能用旧512/100epoch或其他同名模型替代：

| 资源相对目录（相对于`CANDIDATE_ASSETS_ROOT`） | SHA256 |
|---|---|
| `maskdino-mssl-control-s42-20261018T025946/train/model_best.pth` | `9489d529cac7d6c73fe746b00f630d1773547758588669b54e1e02b0926ea775` |
| `mssl-official50-s42-20261018T025946/train/model_best.pth` | `c0ea44a7fdfd0e673d32729b128cbbf8916c313b206e5ab334335331c52a0ae0` |

本机文件在工作区`runs/`的上述目录。若训练机没有它们，需要单独复制，共约1.25 GB；不需要复制整个旧训练目录，也不需要重新训练。实验JSON与历史预测已在Git的bundle中。

沿用训练机可用的MaskDINO环境与固定原生版本`3831d8514a3728535ace8d4ecc7d28044c42dd14`，CUDA扩展须已为训练机GPU编译。本机原生CUDA源有既有`value.type()`→`value.scalar_type()`兼容补丁，数学算子未改；本次不远程安装或重编译。Python/YAML源码严格核对，扩展兼容性及跨GPU数值差异由真实前向/历史预测门槛检查。

## 训练机启动

同步本次Git分支后，在项目根目录、已激活模型环境的终端运行。设置实际数据目录；其中应直接包含`data_C1`至`data_C6`。

```bash
export POLYPGEN_ROOT=/实际路径/PolypGen2021_MultiCenterData_v3
# 以下默认值若与训练机不符才设置：
# export POLYPGEN_ANNOTATIONS=/实际路径/provided_boxes.coco.json
# export CANDIDATE_ASSETS_ROOT=/实际路径/装有上述两个run目录的根目录
# export CUDA_VISIBLE_DEVICES=0
bash scripts/run_candidate_stage_diagnostics.sh --training-machine
```

如果终端会断开，可用`nohup bash scripts/run_candidate_stage_diagnostics.sh --training-machine > candidate-stage-launch.log 2>&1 &`启动；上述环境变量须先export。只顺序占用一个GPU，不触发任何训练。输出目录已存在时拒绝覆盖；失败后保留记录，用新的输出目录重新准备，不盲目重试或修改阈值。

## 回传产物与决策

回传完整`runs/candidate-stage-remote-*`目录，尤其是：`queue.json`、两组`verification.json`、`summary.json`、`per_gt.csv`、`images/*.json`和`images/*.npz`、`report/report.md`、`comparison.json`与`examples.jpg`。若失败，回传日志、state、job和`last_parity_actual.npz`；失败结果不进入方法决策。

报告展示密集池→Top300→实际初始化→每层Decoder→Top100的连续IoU和覆盖变化；示例同时包括退化与恢复。72张是平衡机制面板，不外推总体故障比例，也不自动宣布唯一主因。神经特征可分性及局部关系是否有效仍未被这项几何诊断证明。

两份报告共同决定第二机制的挂载位置；在此之前不排局部对比完整训练，不通过本次PolypGen分析继续搜索超参数。
