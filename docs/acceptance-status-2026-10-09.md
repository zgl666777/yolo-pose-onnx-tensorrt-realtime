# 当前验收状态

状态依据已经落盘的证据；探索回放、数值一致性和硬件验收分别记录。四段视频参与规则修复后的回归，不是冻结的独立最终测试集。

| 项目 | 当前结果 | 可审核证据与边界 |
| --- | --- | --- |
| Windows 真实视频告警 | fall-01、fall-15、fall-30 各一次，adl-01 无告警 | `runs/remediation/replay-fp32-v2/`；不代表 30 段跌倒或人物独立准确率 |
| 全 70 视频探索回放 | 11,936 帧、0 调度丢帧；TP 17、FP 4、FN 13，召回率 56.67%，Precision 80.95% | `runs/remediation/urfd-all70-fp32-v2/official-coarse-report.json`；ADL30/32/33/40 各误报，13 段跌倒漏报；人工身份、RGB 框和事件边界尚未审核，不能验收准确率 |
| VMware Ubuntu CPU | ONNX 基准、四段视频回放及性能汇总完成；3 次跌倒告警、ADL 0 次 | `runs/remediation/ubuntu-evidence/runs/linux-vmware-cpu/acceptance.json`；GPU/C++/独立 PyTorch 精度阶段为 skipped |
| Ubuntu 最新代码测试 | 162 passed、13 skipped，退出码 0 | `runs/remediation/ubuntu-final-tests-v2.log`、`.xml`、`.exitcode`、`.source.json`；65 个源码文件哈希全部匹配，跳过 GPU/C++ 等环境相关项，测试来源与长测启动 provenance 分开记录 |
| Windows GPU 完整回归 | 175 passed、0 skipped，退出码 0 | `runs/remediation/final-windows-gpu-tests-v4.xml`、`.log`、`.exitcode`、`.source.json`；65 个源码文件哈希全部匹配，覆盖当前 Windows GPU/C++ 环境 |
| 有限混合精度严格对齐 | 20+70 张抽样通过，但全 11,936 帧 321 帧失败，验收未通过 | `runs/remediation/precision-mixed-all11936.json`；抽样报告保留，不是完整 FP16 或人工事件精度 |
| 显式混合精度全量对齐 | 11,936 帧 raw 坐标/分数全部通过，123 帧后处理未通过 | `runs/remediation/precision-mixed-explicit-all11936.json`；只有三个人物分数卷积使用 FP16，严格验收仍未通过 |
| FP32 全量对照 | raw 坐标/分数全部通过，45 帧因可见关键点不足未通过后处理 | `runs/remediation/precision-fp32-all11936.json`；保留证据不足帧，不将其删除或写成总体验收通过 |
| 真实摄像头 RTSP 停流恢复 | Ubuntu 客户端完成 90.36 秒、635 帧、5 次重连、0 推理故障，线程/进程退出 | `runs/remediation/linux-rtsp-fault-correlated.json`、`ubuntu-evidence/runs/linux-live-rtsp-fault/` 与 `runs/windows-camera-rtsp-linux-soak-v3/faults.jsonl`；详情见 [故障结果](fault-results-2026-10-09.md)，不是独立 RTSP 硬件或物理网络拔线 |
| Windows 小时级长测 | 3601.42 秒、76,499 帧、0 推理故障、0 重连 | `runs/remediation/live-camera-1h/`；主进程 RSS 增长 10.03 MB，丢帧 29.12%，不代表 Linux/NVIDIA 性能或无泄漏 |
| Linux 小时级长测 | 3600.38 秒、22,173 帧、0 推理故障、0 重连，稳定性门槛通过 | `runs/remediation/ubuntu-evidence/runs/linux-live-rtsp-1h/`；帧覆盖 3598.46 秒、最大间隔 0.891 秒；RSS 增长 5.92 MB，6.16 FPS、丢帧 79.46%、全帧 P95 321.47 ms，不代表实时性能达标 |
| 人工身份/四类姿态/框/事件 | 审核工具与 11,936 帧建议草稿已具备，人工完成 0 帧 | `data/urfall-review-draft-suggestions/`，9,633 个框建议；17 段错误视频的 480 帧优先审核队列在 `error-analysis/priority-review-v2/`；模型建议不能升级为人工真值 |
| Linux NVIDIA、物理 USB | 缺对应硬件证据 | 普通 VMware Workstation 3D 加速不是 CUDA 直通；内置摄像头与虚拟断连不是外置 USB 物理拔插 |

全量错误证据在 `runs/remediation/urfd-all70-fp32-v2/error-analysis/`，每段误报/漏报保留六帧 RGB、框及运动诊断。旧粗评分把 ADL 的官方 0 临时姿态当作跌倒事件，会错误隐藏四次误报；现评分使用 `transition` 粗姿态，仅 `fall-*` 序列定义探索正事件，ADL 告警均计 FP。默认规则不能可靠区分主动蹲跪/躺下与意外失去支撑，不应通过降低阈值直接声称修复。

另下载 fall-17、fall-23 的官方原始 640×480 RGB PNG，完整 ZIP CRC、解包路径与图像哈希已校验；用同一 FP32 模型和参数逐帧比较时，fall-23 原图检出一次（预览版漏报），fall-17 两版仍漏报。[原图比较证据](../runs/remediation/original-rgb-comparison.md) 说明预览分辨率确实限制部分场景，但两段诊断不能代替全量独立准确率。原图审核任务仍是待人工草稿。

## 精度声明纠正

旧 `models/pose-fp16-strict.engine` 的实现把所有浮点层约束为 FP32。其 20 帧通过报告 `precision-fp16-strict-v2.json` 保留作为历史结果，但不能证明 FP16 骨干或 FP16 严格验收。

新 `models/pose-mixed-strict.engine` 只将 `/model.23/cv3.0/cv3.0.2/Conv`、`cv3.1/cv3.1.2/Conv`、`cv3.2/cv3.2.2/Conv` 三个末端人物分数卷积设为 FP16；骨干、姿态头与解码仍为 FP32。构建报告 `models/pose-mixed-strict.engine.json` 和 layer inspector `models/pose-mixed-strict.engine.layers.json` 保留实际 Half 层与哈希。该策略是有限混合精度，不能称为 FP16 骨干、完整 FP16，或沿用纯 FP16 的历史 FPS。

两份抽样报告均使用原严格阈值 `coordinate_atol=0.1`、`score_atol=0.005`、`rtol=0.001`，未放宽阈值。20 帧与每视频一帧的 70 帧均通过，但全量 `precision-mixed-all11936.json` 为 `passed:false`：11,615 帧通过、321 帧失败。失败项目有重叠，包含 185 帧 raw 坐标、4 帧分数与 135 帧后处理失败；其中有 45 帧可见关键点不足、25 帧可见性变化、2 帧人数变化。原纯 FP16 失败报告继续保留。

`precision-fp32-failure-control.json` 保留这 321 帧的历史对照。旧全 FP32 约束 `pose-fp16-strict.engine` 有 253 帧失败，包含 170 帧 raw 坐标、4 帧分数与 82 帧后处理失败。因此约束导致的 TensorRT tactic 差异也影响对齐，不能把全部误差归因于 FP16。

最新全量 `precision-fp32-all11936.json`：11,891 帧通过、45 帧未通过，raw 坐标与分数全部通过，45 帧均因参考可见关键点不足四个而无法验证姿态；没有人数、可见性或关键点距离失败。最大 raw 坐标绝对差 0.007889、分数差 0.0001743。保留这 45 帧的未通过状态，不删除困难样本。

显式 ONNX Cast + TensorRT `STRONGLY_TYPED` 策略已实际构建并检查 Half 层：全部 11,936 帧 raw 坐标和分数通过原严格阈值，但 123 帧后处理未通过。包含 45 帧关键点证据不足、23 帧可见性变化、67 帧关键点距离超过 3 像素及 2 帧人数变化，分类有重叠。`fp16-explicit-lowres` 的 321 帧失败子集也未通过，未升级为默认模型。详情见 [显式混合精度实验](explicit-mixed-precision-2026-10-09.md)。这仍是有限混合精度，完整 FP16 和严格端到端验收均未完成。

## 长测判定

`tools/soak_report.py` 默认描述统计。显式设置时长、内存、丢帧或帧间隔阈值才启用验收状态与失败退出码，同时检查 completed、0 推理故障、无清理错误、采集线程/进程停止及 summary 已记录的退出码。一小时门槛还要求帧时间覆盖至少 3595 秒、时间单调及默认最大间隔 5 秒，避免两帧相隔一小时被算通过；`--max-frame-gap-seconds` 可显式调整并记录。RSS 只覆盖主推理进程，未记录采集子进程 RSS；显存是整卡统计。

```bash
python tools/soak_report.py --run runs/linux-live-rtsp-1h \
  --output runs/linux-live-rtsp-1h/soak.json \
  --min-seconds 3600 --max-rss-growth-mb 32
```

上述时长与内存设置只用于稳定性门槛，丢帧率和延迟仍按实测描述，不宣布实时性能通过。需要性能门槛时应显式追加 `--max-drop-ratio` 并记录选定值；不能用无阈值描述报告或尚未结束的进程宣布通过。多人交叉、长遮挡、慢速跌倒、主动快速躺下、分辨率变化和 8 小时验证仍需扩展真实证据。

Ubuntu 长测使用 2 个虚拟 CPU、约 4 GB 内存的 ONNX Runtime CPU，输入为 Windows 内置摄像头经 FFmpeg/MediaMTX 发布的真实 RTSP。运行期间主机同时做精度验证，访客安装依赖并执行测试，因此该记录是并发负载下的稳定性证据，不是隔离性能基准。FFmpeg 存在重复帧，发布 30 FPS 不能当作 30 个独立摄像头图像。原始启动 `provenance.json` 保留当时源码与环境，后续测试来源哈希分开保存。

## 机器可审核汇总

`tools/remediation_report.py` 从现有 JSON、JUnit XML 和帧账本生成项目相对路径、文件大小、SHA256 与各目标 `completed`、`partial`、`blocked` 状态。模型/RTSP 关联原始文件哈希也会检查；文件变更、只通过抽样、阈值放宽、缺人工见证、描述性长测或尚未完成的进程都不能提升状态。

```bash
python tools/remediation_report.py --output runs/remediation/delivery-manifest.json \
  --windows-tests runs/remediation/final-windows-gpu-tests-v4.xml \
  --linux-tests runs/remediation/ubuntu-final-tests-v2.xml \
  --windows-soak-report runs/remediation/live-camera-1h/soak-frame-coverage.json \
  --precision runs/remediation/precision-mixed-explicit-all11936.json \
  --precision-control runs/remediation/precision-fp32-all11936.json \
  --precision-control runs/remediation/precision-mixed-all11936.json \
  --manual-audit runs/remediation/review-audit-empty-v2.json
```

最终清单为 6 项 completed、2 项 partial、3 项 blocked，`final_acceptance_complete:false`。completed 的范围分别是 CPU 软件验收、两平台代码测试、RTSP 发布端故障恢复和两平台一小时稳定性；partial 为探索准确率、混合精度；blocked 为人工真值、Linux GPU 与物理 USB。它不代表全部项目目标完成。

默认回归参数仍指向历史 XML，上述最终命令显式选择最新版本。附带 `.source.json` 时逐文件核对当前源码哈希；缺文件、非法哈希或源码变化会降为 partial，旧来源仍保留。长测须落盘当前显式阈值报告后重新生成汇总；Windows 单独用 `soak-frame-coverage.json` 保留历史 `soak.json`。命令退出码 0 仅代表成功生成清单，探索粗标签不能完成独立人工准确率验收。

测试服务已自然退出，MediaMTX/FFmpeg 进程均已结束；访客临时 RTSP 地址 `169.254.12.38/16` 已移除，恢复原 `192.168.63.128/24`，证据为 `runs/remediation/ubuntu-network-final.txt`。
