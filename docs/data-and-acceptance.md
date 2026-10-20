# 真实数据与验收流程

## 公开视频与使用范围

原始发布方：[UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)。包含 30 段跌倒、40 段日常活动。项目下载 camera 0 RGB MP4、逐帧同步 CSV、原始深度姿态标签，保存于 `data/urfall/`。

许可为 CC BY-NC-SA 4.0，限非商业学术使用；商业用途需联系发布者。引用：Kwolek & Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014, 489–501。

```bash
python tools/download_urfall.py
python tools/prepare_urfall_review.py --output data/urfall-review-complete
```

`download-manifest.json` 保存来源 URL、许可、各文件 SHA256、帧数及尺寸；`official-page.html` 是原网页快照。`video-registry.csv` 中人物身份留空，防止把同一个演员在不同视频中误当成不同人物。

官方标签：-1=非平躺，0=临时过渡姿态，1=地面平躺，来自深度帧。官网说明 0 是不用于其姿态分类的 temporary pose；日常活动的主动坐下、跪下或躺下也含 0，因此不能将其直接转换成本项目的意外跌倒 `falling` 真值。-1 不能转换成“站立”，因为它也可能包括坐姿。官方页面未给出本项目所需的跨视频人物身份、RGB 人工框；RGB/深度同步也需复核。不得用 YOLO 框或分类结果补成评估真值。

`review-tasks.jsonl`、`events.provisional.csv` 是待审任务，不是已完成标注。稀疏抽帧用于标注启动；最终测试需要覆盖全部预测帧及全部人物，也需要空场景帧审计。暂不能声称已经建立人物独立测试集。

本机已生成 `data/urfall-review-draft/` 作为可审核草稿：11,936 个任务、70 段视频，保留官方粗标签作为 `suggested_posture`，身份、RGB 框和四类姿态全部留空；`draft-manifest.json` 明确标记 `authoritative:false`。草稿不会被训练/评估命令接受，人工完成后必须通过 `tools/annotate.py` 的 ReviewStore 导出。

完整建议草稿在 `data/urfall-review-draft-suggestions/`，11,936 帧全部用 TRT FP32 推理，9,441 帧含共 9,633 个人框建议，2,495 帧没有检测建议（需人工排查遗漏）。`suggested_people` 的 bbox 来源是 `model_suggestion`，姿态仅为未训练规则提示，`candidate_local_actor_id` 仅限视频内轨迹，不是已核实人物身份。所有行仍为 `human_review_pending:true`。旧草稿里的 `falling_candidate` 表示官方 0 过渡提示，不是意外跌倒标注；未来草稿使用 `transition_candidate`。保留已有草稿及来源文件哈希，不把历史产物改写成真值。

新版草稿为 `data/urfall-review-draft-v2/`，修正草稿图片路径并保留来源任务 SHA256、完成度与验收阻塞要求。机器可读审核命令如下：

```bash
poseguard review-audit --tasks data/urfall-review-draft-v2/draft-review.jsonl \
  --database data/manual-review.sqlite --output runs/manual-review-audit.json
```

审计不会创建不存在的审核数据库，也不会把草稿变成真值。`ready_for_export:false` 时返回退出码 2；每个任务（含空场景）都完成、身份及证据齐备、跌倒帧均被人工事件覆盖后才允许导出。已保存的图片、时间戳或裁剪若与当前任务变更不一致，也需要重新审核。当前 `runs/remediation/review-audit-empty-v2.json` 记录 0/11,936 完成，因此人工标注目标仍未验收。

## 标注定义

| 标签 | 定义与边界 |
| --- | --- |
| standing | 身体由脚部支撑，包含站立和行走；弯腰但未失去支撑仍归此类 |
| sitting | 臀部由椅面等支撑，包含坐姿倾斜；不将坐下动作当跌倒 |
| falling | 意外失去站立或坐姿支撑，向地面下降的过程；需结合前后帧判定 |
| lying | 身体主要由地面/床面等承托；主动躺下和跌倒后平躺都属于此姿态 |

跌倒事件起点为首次可确认意外失去支撑的帧，包含从椅子滑落或坐姿侧倒。事件结束为恢复稳定支撑或视频结束。主动坐下、主动躺床、跪地、弯腰捡物是负例；不能只由“平躺”推导跌倒。遮挡或无法判断的帧保留审核问题，不能悄悄删掉困难样本。标注协议由至少两人复核事件边界与争议样本，保存审核者和版本。

当前时序规则允许站立或坐姿→平躺持续满足阈值触发。仅凭静态姿态不能可靠区分主动躺下和意外跌倒；该规则必须在上述负例上调参和验收，坐姿回归通过不等于真实坐姿跌倒检出率达标。

修复后的时序检测会保留跌倒冲击前捕获的运动证据 1.5 秒，在短暂关键点遮挡后用运动强度和落地姿态触发；慢速坐下没有足够运动强度，不会触发。Windows 回放证据见 `runs/remediation/replay-fp32-v2/`：fall-01、fall-15、fall-30 各 1 次告警，adl-01 为 0 次。

人工 JSONL 每人每帧一行：

```json
{"image":"frames/v01-00001.jpg","video_id":"v01","person_id":"subject-A","timestamp":0.0,"bbox":[10,10,100,200],"label":"sitting","annotation_source":"manual","identity_verified":true}
```

同一人跨视频使用同一个 person_id；多人视频中所有人都标注。身份无法核实的样本不能假定人物独立。bbox 必须是原图人工 xyxy，评估不需要关键点。

## 严格分离训练与测试

1. 先完成身份、姿态、框与事件人工审核，再按共享人物或视频的连通组冻结划分。
2. 特征提取只产出 `pose_features`；保留 `annotation_bbox`、源标注哈希和匹配 IoU，完整人工真值另存，所有漏检进入审计。
3. 训练仅拟合 train，C=0.1/1/10 仅用 validation 的 macro F1 选择。测试记录仅做输入格式验证及保存，不参与拟合、预测和评分。
4. 锁定模型及参数后单独执行最终评分，核对模型与测试特征哈希；全过程分类/检测/事件另用完整人工真值评估。

```bash
poseguard prepare-splits --annotations data/manual.jsonl --events data/manual-events.csv --output data/frozen
poseguard extract --annotations data/manual.jsonl --config configs/camera.yaml --output data/features.jsonl
poseguard train --dataset data/features.jsonl --split-manifest data/frozen/split-manifest.json \
  --output models/real-classifier.joblib --report-dir runs/real-training
poseguard score-classifier --report-dir runs/real-training --output runs/real-final-classifier.json
poseguard evaluate --ground-truth data/frozen/test.ground-truth.jsonl \
  --predictions runs/final-test/predictions.jsonl --events runs/final-test/events.csv \
  --truth-events data/frozen/test.events.csv --output runs/final-test/evaluation.json
```

切勿用 `features.jsonl` 当真值，工具会拒绝。旧版本未标记的提取文件无法自动判断来源，应回到原始人工标注。模型/报告已有内容时需使用新的实验目录。提取可以在整份人工标注上运行以复用推理，但划分必须预先冻结；训练读取划分与原始框/标签/来源哈希。

`score-classifier` 仅评估成功提取的姿态，不能代表漏检后的端到端指标。评估报告同时保留检测 TP/FP/FN、人工框 IoU、每人轨迹碎片及 ID 切换；身份纯度不能代替这些指标。

## Linux/GPU 验收

本机 Windows 已联入 VMware Ubuntu 64 位虚拟机，Linux CPU 四视频回放与真实摄像头 RTSP 软件停流恢复已有证据，见 [当前验收状态](acceptance-status-2026-10-09.md)。该虚拟机没有 NVIDIA GPU 直通，不能把 Windows Engine 或 CPU 回放标记为 Linux GPU 验收。

### VMware Ubuntu 64 位路径

VMware Workstation 中的 Ubuntu 22.04/24.04 64 位虚拟机可作为 `linux-cpu` 验收目标。将项目连同 `models/yolo11n-pose.onnx` 和回放视频放入虚拟机后，先安装基础依赖：

```bash
sudo apt-get install -y python3-venv libgl1 libglib2.0-0
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

用下列预检保存虚拟化、系统、依赖和能力证据：

```bash
python tools/acceptance.py --output runs/linux-vmware-preflight \
  --profile linux-cpu --preflight-only
```

`--profile linux-cpu` 只运行 ONNX Runtime CPU 基线、回放和软件长测，并将 TensorRT/C++ GPU 阶段记为 `skipped`。这是 VMware 默认可靠路径，不需要伪设 GPU 直通。CPU 环境未安装 `torch`/`ultralytics` 时，独立精度对比也会明确记为 skipped；需要 ONNX/PyTorch 对比时再安装 `python -m pip install -e '.[export]'` 并提供 `--weights`。只有当 Ubuntu 能证明 `nvidia-smi`、TensorRT 和引擎重建都可用时，才使用 `--profile linux-gpu --rebuild`；普通 VMware 3D 加速不等于 CUDA 或 PCIe 直通。

CPU 回放的可执验命令：

```bash
python tools/acceptance.py --output runs/linux-vmware-cpu \
  --profile linux-cpu --video data/urfall/fall-01-cam0.mp4 \
  --stage-timeout 1200
```

虚拟机已有摄像头或 RTSP 源时，新建一份与 `configs/camera.yaml` 格式相同的配置，设置其 `source`，再加 `--camera-config configs/ubuntu-source.yaml --soak-seconds 3600`。没有实际源的配置不能当长测通过证据。

Ubuntu 虚拟机里的 USB 设备必须在 VMware 菜单中选择 **VM > Removable Devices > Connect**；内置摄像头不构成物理 USB 拔插证据。RTSP 需要桥接网络并记录真实上游停止/恢复时间；只有回放或本机 MediaMTX 注入仍是协议回归，不替代真实摄像头故障。

脚本会写入 `acceptance.json` 的 `profile`/`capabilities`/`virtualization` 字段；每个被跳过的阶段保留原因。因此 VMware CPU 结果可作 Linux 软件证据，物理 USB、独立 RTSP 硬件与 Linux/NVIDIA GPU 必须各自保留对应实测记录。

安装目标环境依赖、CUDA/TensorRT/OpenCV 开发包后，在目标 Linux 执行：

```bash
python tools/acceptance.py --output runs/linux-acceptance --profile linux-gpu --rebuild \
  --video data/test-video-1.mp4 --video data/test-video-2.mp4 \
  --classifier models/real-classifier.joblib --ground-truth data/frozen/test.ground-truth.jsonl \
  --ground-truth-frames data/frozen/test.frames.jsonl \
  --split-manifest data/frozen/split-manifest.json --training-report runs/real-training/training.json \
  --truth-events data/frozen/test.events.csv \
  --camera-config configs/usb-test.yaml --camera-config configs/rtsp-test.yaml --soak-seconds 28800
```

工具在 `linux-gpu --rebuild` 目标设备重新构建 C++ 和 FP32/严格混合精度 Engine，以 ONNX/PyTorch 为基线验证真实帧，跑五种后端的基准与全流程回放，保存原始日志、命令、环境和哈希。默认 `fp16-strict` 仅将末端人物分数卷积 `cv3.*.2/Conv` 设为 FP16，骨干、姿态头和解码保留 FP32；构建报告与 inspector 记录实际 Half 层。旧 `pose-fp16-strict.engine` 是全部浮点 FP32 约束，不能作为 FP16 证据；新 `pose-mixed-strict.engine` 才是有限混合精度。需要纯 FP16 探索时显式传 `--fp16-build-mode fp16`。Windows 探索须显式 `--allow-windows`；不会被标记成 Linux 成绩。完整 test 真值必须对应指定视频及其源时间戳，不能把仅子集视频的预测与整个测试真值混算。

实时验收应记录：USB 物理拔出/插回、RTSP 上游进程停止/恢复、网络中断/恢复、持续坏流、分辨率变更；每次写入 UTC 故障时刻，关联 `health.jsonl` 的打开/读取失败/重建/恢复记录。USB 驱动阻塞不能仅由 mock 证明恢复；不要在测试时假称软件关闭设备等于物理断连。

`tools/rtsp_fault_test.py` 可对本机 MediaMTX+FFmpeg 做实际 TCP/RTSP 停止/重启注入。它验证真实协议故障，仍不是目标 Linux 上真实 RTSP 摄像头/网络验收。

默认长测 8 小时。`frames.csv` 保留逐阶段耗时、队列等待、主机延迟及主进程 RSS；采集子进程 RSS 未记录。`gpu.csv` 每秒保存整卡显存/利用率/温度，显存含其他应用，不能当本进程分配量。`soak_report.py` 对全量 CSV 计算 P50/P95、内存趋势；默认只描述，显式提供 `--min-seconds`、`--max-rss-growth-mb`、`--max-drop-ratio` 才有验收通过/失败与退出码，短回放不能通过一小时门槛。主机延迟从读取完成计时，未包含摄像头/网络上游缓存。事件媒体延迟与加上主机处理时间的输出延迟分别报告。

验收脚本始终明确保留人工独立测试、物理故障及长测的待核实状态，不以工具执行成功自动宣布生产验收完成。

当前设备边界：Windows 内置摄像头可经 RTSP 提供真实帧到 VMware Ubuntu；Linux CPU 回放、90 秒协议停流恢复和一小时稳定性测试已完成，Windows 内置摄像头一小时也已有完成证据。Linux 一小时为 3600.38 秒、22,173 帧、0 故障，6.16 FPS、丢帧 79.46%，未通过实时性能验收。Linux NVIDIA、独立 RTSP 硬件/物理网络断线及外置 USB 拔插仍缺对应设备；不同环境的证据分别保存。
