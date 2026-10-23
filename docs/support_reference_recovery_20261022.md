# 三条件完整实验中断恢复

2026-10-22 20:22（北京时间）用户要求继续时，发现原有限队列状态仍为running，但WSL中的控制器PID3297和训练包装进程PID1851951都不存在；GPU约1%利用率、约1GiB显存。第三组metrics最后写入19:25:37、第2319迭代，外层日志最后19:26左右。Windows上次启动时间为10月20日，现有证据不能把退出归因为Windows重启，也不能确定是人为停止、WSL退出或其他原因。

前两组训练、Kvasir Val预测与正式门禁完整：blend05选中模型Val AP62.695，geometry64.002；这些不是PolypGen结果。原生Baseline源域AP66.072。恢复不依据中途AP修改模型、配置或预算。

第三组最近周期检查点`model_0002199.pth`含model、trainer、iteration；trainer内有optimizer、grad_scaler和LRScheduler状态，但未保存完整随机/增强/数据加载器状态。可以进行非逐位一致的优化器续训，但本次为保护受控对照，采用**只重跑第三组、从同一原始预训练起点完整训练**。原中断run及其最佳模型全部保留、标明selection_eligible=false；其Val结果不参与新run的checkpoint选择，不跨两次训练挑选最佳。

恢复程序为`scripts/recover_support_reference_queue.py`，三项回归检查通过：只替换未完成run、不改冻结配置/源码清单；错误阶段拒绝恢复；缺少已完成对照时拒绝跳过。启动前校验原协议源码和引用SHA、前两组正式冻结训练以及train/predict_val完成状态。恢复程序自身也记录SHA。训练和评估继续从原始冻结source导入，未修改原协议文件、冻结源码或损失函数。

- 原队列：`runs/support-reference-full-20261021/`，标为interrupted并记录新队列位置。
- 恢复队列：`runs/support-reference-recovery-20261022/`，保留原任务顺序及已完成的11个任务；三组PolypGen推理、评分与最终比较在第三组完成后执行。
- 第三组新run：`reference-support-s42-20261021T154208Z-restart-20261022T122625Z`。
- 20:26启动新控制器PID2617；重新启用隐藏Windows防自动睡眠进程PID21412。
- 剩余任务预计需要约7～9小时，实际以新run吞吐和后续评估为准；不沿用中断前预计完成时间。

没有新增实验条件、系数搜索或MSSL组合；PolypGen仍为主要外部开发基准，PolypsSet和PolypDB保留。当前未完成的新训练和评估不能记为已通过，也没有新的第二机制有效性结论。
