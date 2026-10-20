# 第二轮逐项修复与验证

本页为第二轮历史记录；最新环境与精度结论见 [当前验收状态](acceptance-status-2026-10-09.md)。后续已补三段跌倒告警回归、有限混合精度抽样对齐、Ubuntu CPU 四视频和真实摄像头 RTSP 停流恢复；混合精度全量检查失败，人工独立最终测试、Linux NVIDIA 和物理 USB 仍待完成。

## 已完成

| 问题 | 实现与验证 |
| --- | --- |
| 人工真值被训练特征覆盖、漏检被删除 | 保留完整人工真值、原框和源哈希；两人漏一人计 FN、框误差不被覆盖已有回归 |
| 离线评分绕过可见性拒识 | 实时预测、验证集选参、训练报告、最终评分使用同一个可见性门槛；unknown 计入混淆矩阵及拒识数量 |
| 下一帧预测被上一帧抢占 | 明确帧 ID 优先，时间匹配先保留精确帧，再用稀疏一对一最优匹配；回归覆盖掉帧和模糊邻近时刻 |
| 空场景无法评分 | 增加真值帧账本、预测帧账本和人数核对；零人帧误检计 FP，完全空预测可评分；未标注预测帧阻止完整评估 |
| 启动与后端恢复 | 从启动起保存失败摘要；后端关闭/重建失败继续有限重试；已有回归 |
| 关键点和轨迹指标盲区 | 检查基线可见点、丢失点和最低可比较点；报告轨迹碎片、ID 切换和漏检后重获 |
| 告警只检查媒体时限 | 支持 media/emitted 两种时限；输出超时计 FP/FN；缺失输出延迟时拒绝 emitted 评分；独立回执记录 events.csv flush 完成时刻 |
| C++ NMS 顺序不同 | 共享原生预后处理，先模型空间 NMS 再裁剪；有限值过滤、稳定排序、top3000、max100、关键点置信度裁剪对齐 Python；实际编译和跨语言对照通过 |
| 原生驱动读帧阻塞 | 实时采集使用独立进程、共享帧缓冲和看门狗；可强制终止和重启；重启丢弃旧流帧并重置跟踪；真实子进程阻塞/恢复/退出测试通过 |
| 仅姿态变化即报警 | 平躺分支要求近期支撑和髋部下降/躯干倾斜运动证据；记录运动特征；快速坐姿跌倒、慢速躺下、静态平躺、低可见性、多人物回归通过 |
| 人工标注缺少工具 | tools/annotate.py 提供框、姿态、跨视频身份依据、空帧、事件和审核者；SQLite 保存；未完成帧不能导出，falling 帧须关联事件 |
| 验收脚本缺少保护 | 子进程有超时和进程组清理；完整评分须有帧覆盖、真实四类模型、冻结划分及训练/模型/测试哈希对应；空场景视频单独冻结分配 |

## 实际证据

- GPU/C++ 全部测试：**71 passed**，日志 `runs/revised-final-tests.log`，机器报告 `runs/revised-final-tests.xml`。移除了被真实进程测试替代的模拟重连用例，不能将测试数量减少理解成验证减少。
- 实际 C++ 构建通过：`runs/revised-cpp-build.log`。新增原生预处理/后处理对照可执行文件 `pose_geometry_check`。
- Windows RTX 4070 Laptop、真实摄像头、FP16 连续运行 **901.42 秒 / 22,378 帧**：系统 **24.83 FPS**，主机延迟 P50 **21.48 ms**、P95 **33.23 ms**，0 推理失败、0 重连、采集线程和进程均正常退出。
- 同次运行丢帧比例 **16.92%**，RSS 峰值 **505.87 MB**，前后段增长 **12.50 MB**；其中测试和五后端回放同时占用 GPU，不能当独占设备基准。15 分钟和可见内存增长不能证明长期无泄漏。主进程 RSS 不包括独立采集进程。完整证据 `runs/live-camera-revised-900s/`。
- 公开 URFD 已有 70 段视频。检查发现 MP4 是深度/RGB 拼接预览，RGB 位于右半幅；新增 source_roi，统一标注与回放坐标。`data/urfall-review-dense-rgb/` 导出 **11,936** 帧待审核任务，使用 RGB 媒体时间并保留同步源时间。该素材是 320x240 RGB 预览，不是原始分辨率 PNG。
- 裁剪 RGB 后已完成 ONNX / TRT FP32 / TRT FP16 / C++ FP32 / C++ FP16 的 4 段实际视频回放及耗时报告：`runs/revised-windows-rgb/`。严格精度检查仍有失败，原始结果保留。
- 标注界面已通过显示检查，未保存任何假真值。computer-use 进一步操作被用户按 Esc 中止，之后不再调用该工具。

## 后续更新与剩余范围

1. **真实人物独立四类数据与最终报告**：公开页面不提供可核实的跨视频演员身份、RGB 人工框及 standing/sitting 四类细分；11,936 帧仍是待审核任务。不能使用模型输出替代人工真值，也不能把不同视频假定为不同人物。完整标注入口已准备好，尚无已审核独立训练/验证/测试集。
2. **真实跌倒检出能力**：本页早期 RGB 回放没有告警；后续 Windows FP32 与 Ubuntu CPU 四段回归均为三段跌倒各一次、ADL 0 次。该改善不代表冻结独立测试达标。主动快速躺下、慢速跌倒、多人和长遮挡仍需真实标注与负例验证。
3. **FP16 精度**：纯 `pose-fp16.engine` 的严格失败继续保留。旧 `pose-fp16-strict.engine` 实际将全部浮点层固定 FP32，不能算 FP16 通过。新 `pose-mixed-strict.engine` 仅三个末端人物分数卷积使用 FP16、骨干/姿态头 FP32，inspector 有 Half 层；20+70 张抽样通过，但全 11,936 帧有 321 帧失败、整体未通过。它是有限混合精度，严格验收仍未完成。
4. **目标 Linux/GPU**：VMware Ubuntu CPU 的基准和四视频回放已有 `acceptance.json` 完成证据。Linux NVIDIA/C++ TensorRT 仍缺 GPU 直通或原生 Linux 硬件。
5. **物理 USB / RTSP 故障**：完整校验后的 MediaMTX 已用于实际协议测试；Ubuntu 已读取 Windows 内置摄像头的真实 RTSP 流并完成 90 秒停流恢复。外置 USB 拔插、独立 RTSP 硬件摄像头和物理网络断线仍缺设备，不由软件停流替代。
6. **长时间验收**：Windows 内置摄像头与 Ubuntu RTSP 均已完成一小时稳定性测试，最新数据见当前状态；8 小时仍未完成。主进程 RSS 不包含采集子进程，不能以增长阈值通过宣布无泄漏。

## 标注和后续验收入口

```powershell
.\.venv\Scripts\python.exe tools/annotate.py --tasks data/urfall-review-dense-rgb/review-tasks.jsonl
```

对每帧画原 RGB 框并填写身份/姿态及身份依据，确认空帧，记录事件边界和审核者后完成审核。未完成任务会阻止导出；不要为未知人物随意填写 identity_verified。

```text
poseguard prepare-splits --annotations data/manual/manual.jsonl --frames data/manual/frames.jsonl --events data/manual/events.csv --output data/frozen
poseguard evaluate --ground-truth data/frozen/test.ground-truth.jsonl --ground-truth-frames data/frozen/test.frames.jsonl --predictions runs/final/predictions.jsonl --prediction-frames runs/final/frames.csv --events runs/final/events.csv --truth-events data/frozen/test.events.csv --event-delivery runs/final/event-delivery.jsonl --deadline-scope emitted --output runs/final/evaluation.json
```

最终 Linux 验收提供 `--split-manifest`、`--training-report`、`--ground-truth-frames` 和已锁定真实分类器；视频集合须恰好对应冻结 test 集，裁剪从人工帧账本读取。媒体时限与实际输出时限应分别报告。原生 pose_camera 是姿态显示示例，监督采集和跌倒全流程使用 `poseguard run --backend cpp`。
