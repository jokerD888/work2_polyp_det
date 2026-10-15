# 实现与验收记录

日期：2026-10-15。基于项目 commit `96cb1f3` 新增独立目录。

## 本轮执行范围

用户确认：**先完成实现和 CPU 检查；GPU 验收待当前训练结束。**

未安装或编译 DCNv4，未更改模型环境，未启动 GPU 前向/反向或新训练。

## CPU 结果

### 首次 DCNv4 接入（ee1f5fb）

`checks/cpu_report.json`：18 项，全部通过，`cuda_context_created=false`。

1. 配置参数与通道分组合法性。
2. 默认关闭、固定 alpha=0.1。
3. DCN-only 继承原生配方，组合配置逐项继承实际 F-v2。
4. 非方形、非连续 BCHW/B(HW)C 往返与残差数值。
5. 输入和测试算子参数梯度，alpha 非参数/非 buffer。
6. alpha=0 精确旁路、算子不执行。
7. CPU autocast 下算子输入 FP32、输出恢复 dtype。
8. 非法形状拒绝。
9. 非 FP32 算子参数拒绝。
10. 模块注册不导入 CUDA 扩展。
11. 真实输出尺度元数据的 low2high/high2low 解析。
12. 真实派生 Head 仅修改目标尺度；Mask Feature、其他尺度和调用参数原样传递；训练/推理均执行。
13. alpha=0 的真实 Head 与原生 Head 输出和输入梯度逐位相同（Pixel/Predictor 为路由 fixture）。
14. DCN/SCDA 四种独立开关对应正确 registry 类。
15. 缺失官方扩展时明确失败，无近似卷积回退。
16. 真实原生/SCDA 全模型 CPU 构建、共有 state_dict 初始化逐位一致、原生优化器无遗漏无重复、尺度为8/16/32/64；仅 DCNv4 被明确注入测试替身。
17. 全程未创建 CUDA context。
18. 新增参数初始化保持调用方 CPU RNG 状态不变。

这些验证不包含真实 CUDA 动态采样的数值正确性。参数数160,368为官方源码结构静态计算，正式值待真实后端构造验证。

### 本轮普通卷积对照

`checks/conv_control_cpu_report.json`：合计26项，全部通过；未创建 CUDA context。

原18项回归通过，新增8项覆盖：

1. 算子枚举与默认值；旧 YAML 未指定 operator 时仍选择 DCNv4。
2. **真实**普通卷积参数590,080、groups=1、weight/bias全零、初始恒等、首步非零梯度、更新后输出变化；CPU RNG状态不变。
3. Conv 在 autocast 下使用 FP32，恢复输入dtype，alpha=0 完全旁路，错误半精度参数拒绝。
4. 四份配置的数据、初始化权重、增强、batch、日程、seed一致；两份组合配置的 F-v2 规则完全一致。
5. `--operator` 经真实 CLI 解析进入有效配置，改变运行指纹；重复同一覆盖生成相同指纹。
6. 训练／预测算子不符时，在建立预测目录或启动 worker 之前明确拒绝。
7. 真实 Conv Head 在 train/eval 仅增强所选 S8，其他尺度与 Mask Feature 原对象传递。
8. 原生和 Conv-only／Conv+F 的**真实完整模型** CPU 构建；共有参数初始化逐位一致；完整模型增加590,080参数；优化器无遗漏无重复。全程安装导入拦截器，任何 `DCNv4` 包导入都会失败，测试通过。

原 `checks/implementation_manifest.json` 与 `cpu_report.json` 作为初始阶段记录保留。本轮新增 `checks/conv_control_manifest.json`，记录当前代码、配置和本轮报告哈希。`dcnv4_adapter.py`、独立 F-v2 目录、基线配方保持原样。

## 待验收

| 项目 | 状态 |
|---|---|
| 官方 DCNv4 在 RTX5070 / torch2.7.1+cu128 上编译 | 未运行 |
| P0 官方算子 FP32/AMP/有限差分 | 未运行；doctor.py 提供入口 |
| P1 原生真实网络 GPU 前向、损失和梯度 | 未运行 |
| P1 真实 DCNv4 参数数量/权重加载/旁路 | 未运行 |
| P2 50–200iter 短跑与成本 | 未运行 |
| 正式训练和检测 AP | 未运行 |

基线、原 F-v2 和 third_party 源文件不在本次修改范围内。当前仅能判定“CPU 接口检查通过”，不能判定“GPU 可运行”或“研究有效”。
