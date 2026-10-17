# V3 自审记录（2026-10-17）

结论：**未发现阻止提交的实现问题**。本轮补充审查记录，仅清理Decoder副本末尾多余空行，没有修改运行语义；不表示已证明检测性能收益，也不表示已完成训练机B16训练验收。

审查覆盖配置解析、独立aux初始化及训练checkpoint重载、Decoder payload、V3-A旧指派等价、V3-B SDF/尺度/区域与补位、Loss/DDP归一化、worker/model集成、runner产物隔离和源码快照。

## 主要核对

- Decoder训练forward的AST与旧SCDA副本一致，增加memory/proposal的行为封装在V3 payload中；原生Top-Q、Mask2Box、DN及Decoder计算不改写。
- Aux head为独立参数/存储，加载ImageNet初始化后才同步native head；predict路径不调用同步、不执行aux forward。辅助回归保留原生unsigmoid proposal先验。
- 指派只读取原生raw boxes；V3-B的SDF和层级/区域选择无梯度。原始多框图不能因增强过滤成为可靠辅助GT，padding和Top-Q排除沿用确认规则。
- 辅助Box Loss使用实际每GT选点数均值及全局有效GT归一化，系数只施加一次；没有分类辅助，空候选/零lambda保留aux参数的合法零梯度路径。
- 两个配置共享低LR协议：B16、LR1e-4、workers4、5500iter、warmup125、steps[4400,4950]。配置准备保留本机reference实际数据和模型路径，拒绝混入高LR/未确认日程/成熟初始化/DCNv4。
- train/predict使用独立worker、run/stage拒绝覆盖、完整源码指纹和统一COCO bbox评测；没有修改旧方法/协议模块。原生及新头参数恰好入optimizer一次，新头按非backbone LR优化。

## 验证证据

复核原有 **23项V3 CPU + 112项旧回归**，以及完整CUDA DN/FP32/AMP真实图预检。报告均通过。清理末尾空行前，源码哈希与验收记录一致；该格式调整后，重新检查Decoder的forward AST仍与已验收副本一致。原始证据哈希保留，不回填成新哈希。GPU预检只做前反向，optimizer_steps=0；其范围和舍入误差标准见 `VALIDATION.md`。

本轮另做两项无训练的CPU补充验证，证据为本机 `runs/scda-v3-review-20261017/supplemental_checks.json`：

1. **跨层补位**：两优选层各仅1个候选、其他层有更高IoU候选，仍先保留两个优选层候选，再从其他层补2个；无重复，补位计数准确。
2. **checkpoint重载**：在一次性模型中让native head和aux head具有不同值，经实际DetectionCheckpointer保存/加载后全部state逐位一致，aux未被native覆盖；eval期间aux forward调用次数为0。此验证没有optimizer step，临时checkpoint随临时目录自动清理。

本机原有 `docs/reviews/` 未跟踪文件不属于V3提交，保留原样；data/weights/runs/环境及训练产物不上传。

## 保留的限制

尚未做A6000 B16健康短跑或5500iter正式训练。完整GPU梯度路由验证是batch1，128图指真实mapper对齐。两rank验证使用CPU Gloo。初始aux末层与native一样为零，因此初始共享特征梯度为零；后续非零末层状态的路由另在一次性测试模型验证，正式初始化未更改。

V3-A的audit沿用旧指派记录，可能包含legacy分类候选的诊断字段；实际 `assignment.cls` 为空，训练只返回 `loss_v3_box`，这些历史诊断字段不是分类辅助损失。
