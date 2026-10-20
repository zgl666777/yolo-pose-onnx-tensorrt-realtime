# 显式有限混合精度实验

结果：**严格验收仍未通过**。`pose-mixed-explicit.engine` 的全部 11,936 帧 raw 坐标/分数通过既有阈值，但有 123 帧后处理未通过。该结果不能写为 FP16 严格精度通过，也不是纯 FP16 模型。

## 可复核构建

```powershell
.\.venv\Scripts\python.exe -m poseguard build-engine --onnx models/yolo11n-pose.onnx --output models/pose-mixed-explicit.engine --precision fp16-explicit
```

新模式先创建可检查的 `pose-mixed-explicit.explicit-fp16.onnx`。图中仅三个末端人物分数卷积的输入和常量权重转换为 FP16，卷积输出马上转回 FP32。其他算术保留原 ONNX FP32，TensorRT 使用 `STRONGLY_TYPED`，不启全局 FP16 或每层 `OBEY_PRECISION_CONSTRAINTS`。引擎 inspector 保存了实际 Half 卷积及 tactic，证明引擎确有 FP16 计算；它仍可能改变 FP32 tactic/融合，不承诺与旧 FP32 引擎逐位一致。

`fp16-explicit-lowres` 是另一个实验模式，仅末端最低分辨率人物分数卷积使用 FP16，其余图运算 FP32。该模式也未通过后处理，不作为运行默认。

原 FP16、旧“strict”全部 FP32、全局约束混合引擎及原始失败产物均保留，新产物没有覆盖旧模型。

## 实际结果

| 范围 | raw 坐标失败 | raw 分数失败 | 后处理失败 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 三个分数卷积 explicit，原 321 失败帧 | 0 | 0 | 114 | 未通过 |
| 单个低分辨率卷积 explicit，同 321 帧 | 0 | 0 | 113 | 未通过 |
| 三个分数卷积 explicit，全部 11,936 帧 | 0 | 0 | 123 | 未通过 |
| FP32 对照，全部 11,936 帧 | 0 | 0 | 45 | 姿态证据不足，未通过 |

全量 123 帧失败中有 45 帧可见点不足、23 帧可见性变化、67 帧可见关键点最大距离超过 3 像素、2 帧人数差异；分类可重叠，没有 IoU 失败。原始输出最大坐标绝对差约 0.06218、分数绝对差约 0.00186。raw 差异满足原 `coordinate_atol=0.1, score_atol=0.005, rtol=0.001`，而微小分数/FP32 tactic 差异仍可能影响 NMS 选择的候选姿态，因此不能仅凭 raw 通过宣称端到端一致。

同一完整帧集的 FP32 对照 `precision-fp32-all11936.json` 已完成：11,891 帧通过，45 帧可见点不足；没有 raw、人数、可见性或关键点距离失败。最大 raw 坐标绝对差约 0.007889、分数差约 0.0001743。该对照表明混合策略仍引入额外后处理差异，不能将所有 123 帧失败解释为数据证据不足。

结果文件：

- `runs/remediation/precision-mixed-explicit-all11936.json` 与同名 `.log`
- `runs/remediation/precision-mixed-explicit-failure321.json`
- `runs/remediation/precision-mixed-explicit-lowres-failure321.json`
- `runs/remediation/build-mixed-explicit.log`、`runs/remediation/build-mixed-explicit-lowres.log`
- `models/pose-mixed-explicit.engine.json` 与 `.engine.layers.json`
- `models/pose-mixed-explicit-lowres.engine.json` 与 `.engine.layers.json`

全量有 9,395 帧提供匹配人物和至少四个可见关键点证据。验证还增加了每个候选引擎至少一帧真实可见姿态证据的总门槛：空场景 raw 对比可以通过，但全无人物的图片集合会明确记录 `insufficient_pose_evidence`，不会总体验证通过。证据不足帧仍保留为原有后处理未通过，未删除样本或放宽阈值。
