# YOLO Pose + ONNX + TensorRT 摄像头实时姿态识别

PoseGuard 是一个以 Linux + NVIDIA GPU 为目标环境的实时人体姿态与跌倒检测项目。包含 Python 摄像头流水线、ONNX Runtime 基线、TensorRT 10 后端、C++ TensorRT 动态库及摄像头示例、CPU 姿态分类器训练、转换一致性验证和性能评估工具。

**实现范围与验证范围不同。** 仓库提供完整部署代码和复现命令；真实准确率、GPU FPS、FP16 精度损失和连续运行结论必须用目标设备及标注数据实测。无模型 demo 使用合成骨架；默认规则分类器未经过训练。它们不能代表真实跌倒检测效果。

2026-10-09 当前状态：[验收状态与证据](docs/acceptance-status-2026-10-09.md)。Windows FP32 与 VMware Ubuntu CPU 四段探索回放均为 3 段跌倒各告警一次、ADL 0 次。Windows 与 Ubuntu RTSP 均完成一小时稳定性测试；Ubuntu 6.16 FPS、丢帧 79.46%，未认定实时性能达标。显式有限混合精度全 11,936 帧 raw 对齐通过，但 123 帧后处理未通过。人工独立最终测试、完整 FP16 严格精度、Linux NVIDIA 和物理 USB 仍未完成。

扩展到全部 70 段、11,936 帧的顺序回放后，当前规则基线为 **TP 17 / FP 4 / FN 13**，事件召回率 56.67%。[全量报告与逐帧证据](runs/remediation/urfd-all70-fp32-v2/official-coarse-report.json) 和 [误报漏报分析](runs/remediation/urfd-all70-fp32-v2/error-analysis/README.md) 明确保留局限：官方 0 为临时过渡姿态，ADL 仍是负例；这些探索数据不能代替独立人工最终测试。

## 1. 安装与快速体验

推荐 Python 3.11/3.12，Linux Ubuntu 22.04/24.04。基础模式也支持 Windows。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
poseguard doctor
pytest -q

# 无需权重、摄像头或 GPU，默认运行 14 秒
poseguard demo --output runs/demo
# 无桌面环境
poseguard demo --headless --save-video --output runs/demo-headless
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m poseguard demo --output runs/demo-windows
```

当前工作目录已安装 `.venv` 并生成 `models/yolo11n-pose.onnx`，可直接启动实际摄像头：

```powershell
.\.venv\Scripts\python.exe -m poseguard run --config configs/camera.yaml --source 0 --output runs/my-camera
```

实际 ONNX 检测画面在 `runs/onnx-preview.jpg`，合成多人告警演示视频在 `runs/demo-final/annotated.mp4`。本机具体检查结果见 `docs/validation.md`。

本机现已补齐 CUDA 12.8、TensorRT 10.8、MSVC 与 OpenCV C++ SDK，并生成 FP32/FP16 Engine 和 C++ DLL。直接使用 GPU：

```powershell
.\.venv\Scripts\python.exe -m poseguard run --backend trt --model models/pose-fp32.engine --source 0 --output runs/my-gpu-camera
# 使用 C++ TensorRT 推理
.\.venv\Scripts\python.exe -m poseguard run --backend cpp --model models/pose-fp32.engine `
  --library build-gpu/cpp/Release/pose_trt.dll --source 0 --output runs/my-cpp-camera
```

Python 自动发现用户目录下的 CUDA/TensorRT DLL；SDK 换位置时设置 `POSEGUARD_SDK_ROOT`，或 `CUDA_PATH`/`TENSORRT_ROOT`。C++ 程序先执行 `. .\tools\gpu-env.ps1`，重新构建执行 `.\tools\build-gpu.ps1`。GPU/C++ 集成测试先设置 `$env:POSEGUARD_TEST_GPU='1'`。

2026-10-05 本机预热后的历史基准为 FP32 约 264 FPS、FP16 约 368 FPS（包含传输，不含前后处理）；41 项历史测试通过。纯 FP16 的严格 raw 检查仍未通过，不能用该速度代表新的有限混合精度引擎。最新精度与环境证据见 [当前验收状态](docs/acceptance-status-2026-10-09.md)；真实跌倒准确率仍需独立人工数据集验证。

可视化窗口中按 `q` 或 `Esc` 退出；命令行可按 `Ctrl+C`。demo 中第一人从站立转为平躺并触发告警，第二人保持站立，随后第一人起身恢复。它用来检查逐人状态和运行记录。

## 2. PyTorch → ONNX

```bash
python -m pip install -e '.[export]'
poseguard export --weights yolo11n-pose.pt --output models/yolo11n-pose.onnx --imgsz 640

# 验证集应包含横屏、竖屏、奇数尺寸、小目标和多人图像
poseguard verify --weights yolo11n-pose.pt --onnx models/yolo11n-pose.onnx \
  --images data/verification_images --output runs/torch-vs-onnx.json
```

首次使用 Ultralytics 的已知权重名称会下载预训练模型。可改为本地 `yolov8n-pose.pt` 等单类别 COCO 17 点模型。预训练权重由原作者提供；使用及分发前需检查 Ultralytics 和权重的许可条款。

模型契约：输入静态 FP32 `[1,3,640,640]`；输出原始 `[1,56,N]`，由 `4 个 xywh + 1 个人体置信度 + 17×3 关键点` 组成。导出设置为 `nms=False`、`dynamic=False`、`half=False`。不支持端到端 NMS 输出或自定义关键点类别。

预处理顺序为等比缩放、值 114 的居中补边、BGR→RGB、除以 255、HWC→NCHW。图像尺寸取整遵循 Ultralytics letterbox 规则；保存实际左/上补边量，后处理按它们还原 bbox 和关键点。输出置信度直接使用原始模型分数，不额外加 sigmoid。

一致性验证使用同一份输入张量分别执行 PyTorch 和 ONNX/TRT。报告区分坐标通道误差与置信度通道误差，同时对后处理人物进行 IoU 匹配，统计可见关键点误差。默认后处理要求 IoU ≥ 0.9、关键点最大偏差 ≤ 3 像素。失败时返回非零退出码；空图像上的全通过不能替代含人的验证集。

## 3. TensorRT FP32/FP16

安装匹配目标 NVIDIA 驱动的 CUDA Toolkit 和 **TensorRT 10.x**。Python TensorRT 包不等于 C++ SDK；C++ 编译另需头文件、`libnvinfer`、CUDA 开发工具及 OpenCV 开发库。按 NVIDIA 安装文档配置库路径后：

```bash
python -m pip install -e '.[trt]'
poseguard build-engine --onnx models/yolo11n-pose.onnx \
  --precision fp32 --output models/pose-fp32.engine
poseguard build-engine --onnx models/yolo11n-pose.onnx \
  --precision fp16 --output models/pose-fp16.engine
poseguard build-engine --onnx models/yolo11n-pose.onnx \
  --precision fp16-strict --output models/pose-mixed-strict.engine

poseguard verify --onnx models/yolo11n-pose.onnx \
  --engine models/pose-fp32.engine --engine models/pose-fp16.engine \
  --images data/verification_images --coordinate-atol 2 --score-atol 0.03 \
  --rtol 0.01 --output runs/trt-consistency.json
poseguard verify --onnx models/yolo11n-pose.onnx \
  --engine models/pose-mixed-strict.engine --images data/verification_images \
  --output runs/mixed-strict-consistency.json

poseguard benchmark --backend onnx --model models/yolo11n-pose.onnx --output runs/ort-benchmark.json
poseguard benchmark --backend trt --model models/pose-fp32.engine --output runs/fp32-benchmark.json
poseguard benchmark --backend trt --model models/pose-fp16.engine --output runs/fp16-benchmark.json
```

FP32 构建禁用 TF32；`fp16` 允许内部层使用 FP16，输入输出仍为 FP32。`fp16-strict` 当前只将末端人物分数卷积 `cv3.*.2/Conv` 设为 FP16，其余骨干、姿态头和解码为 FP32，是有限混合精度。`pose-mixed-strict.engine.layers.json` 保存实际 Half 层检查。旧 `pose-fp16-strict.engine` 将所有浮点层约束为 FP32，其通过报告不能作为 FP16 通过证据。全局约束混合策略全量失败 321 帧；新 `fp16-explicit` 用显式 Cast/强类型构建，raw 全量对齐通过但后处理仍失败 123 帧，见 [实验结果](docs/explicit-mixed-precision-2026-10-09.md)。阈值未放宽，运行默认 FP32，完整 FP16 严格验收仍未完成。

Engine 旁的 JSON 保存 ONNX 哈希、TensorRT 版本、GPU 和精度信息。Engine 需在部署 GPU/软件环境构建，不应当作跨硬件通用文件。项目使用 TensorRT 10 的 `set_tensor_address`/`execute_async_v3`，CUDA stream 同步后读取输出，以 CUDA events 分别测 H2D、GPU、D2H。

## 4. 摄像头与视频

```bash
poseguard run --config configs/camera.yaml --source 0 --output runs/camera-onnx
poseguard run --config configs/camera.yaml --backend trt --model models/pose-fp32.engine \
  --source 0 --save-video --output runs/camera-trt

# 文件按源时间戳与帧率回放；video-id 必须对应标注数据
poseguard run --source data/test.mp4 --video-id test-video --headless \
  --output runs/video-test

# RTSP：将 source 设为视频流地址，建议通过配置文件避免凭据进入命令历史
```

配置相对路径以**命令执行目录**为基准。在项目根目录运行命令。每次运行使用新的输出目录，防止覆盖评估记录。

采集线程使用有限队列，满队列时丢最旧帧；消费者过滤超过 `max_frame_age` 的帧。live 时间戳采用单调时钟，文件采用媒体时间戳；延迟另使用 `perf_counter_ns`。网络流使用 FFmpeg 的打开/读取超时，断连后有限次数重连并增加流代数，消费者随之清空人物状态。推理异常有限次数重建后端；不可恢复错误终止并写入失败摘要。USB 摄像头读取能否及时退出受操作系统驱动影响，摘要记录未退出的采集线程。

人物关联采用带位置预测、关键点距离及门限的 Hungarian 全局匹配。每个 track 独立保留时间窗口、告警和恢复状态。消失超过 TTL 或流重连时清空状态；窗口同时有数量上限，避免长时间运行积累历史。此 tracker 适合有限遮挡，长时间遮挡和多人交叉仍可能发生 ID 切换，需要用实际场景验证。

规则基线分站立/坐姿/平躺。静态平躺默认不视为跌倒；近期站立/坐姿→平躺，或训练分类器输出 `falling`，持续满足阈值后才告警，恢复站立/坐姿后解除。输出 `unknown` 的低质量骨架不会单独触发告警。主动躺下仍可能触发此姿态规则，需结合真实负例验证，参见 [真实数据与验收流程](docs/data-and-acceptance.md)。

## 5. 标注、特征与 CPU 分类器

特征为 17 点的 bbox 内归一化 x/y/置信度，以及躯干倾斜、头脚倾斜、bbox 对数宽高比、可见点比例、相对躯干长度，共 56 维。不可见点坐标置零，保留置信度。分类器采用 `StandardScaler + LogisticRegression(class_weight='balanced')`；持久化特征版本和关键点阈值，运行时检查一致性。

标注图像 JSONL 格式示例（bbox 为原图 xyxy；同帧每个人单独一行）：

```json
{"image":"frames/000001.jpg","video_id":"video-01","person_id":"subject-01","timestamp":0.033333,"bbox":[110,40,310,490],"label":"standing"}
```

`image` 相对于标注 JSONL 所在目录。姿态标签为 `standing`、`sitting`、`lying`、`falling`，由人工标注。**person_id 应跨视频保持真实人物身份**，不能直接使用视频内部的 tracker ID。提取工具只按人工 bbox 做一对一 IoU 匹配，不以模型预测替代真值：

```bash
poseguard prepare-splits --annotations data/annotations.jsonl --events data/manual-events.csv --output data/frozen
poseguard extract --annotations data/annotations.jsonl \
  --config configs/camera.yaml --output data/poses.jsonl
poseguard train --dataset data/poses.jsonl --split-manifest data/frozen/split-manifest.json --output models/pose_classifier.joblib \
  --report-dir runs/training
poseguard score-classifier --report-dir runs/training --output runs/final-classifier.json
poseguard run --config configs/camera.yaml --classifier models/pose_classifier.joblib \
  --output runs/learned-camera
```

人工标注还需 `annotation_source:"manual"` 和 `identity_verified:true`，在人物身份审核后冻结划分。提取后每行带 `record_type:"pose_features"`、预测 bbox/keypoints、原始 `annotation_bbox`、源哈希和 IoU。完整人工真值保存到 `poses.ground-truth.jsonl`，含所有漏检人物，审计保存到 `poses.extraction.json`；这三个文件不能互相覆盖。

划分以“同人物或同视频形成的连通组”为单位，近似 60/20/20，至少 5 个独立组；连通组太大时比例会偏离目标。只在 train 拟合 scaler 与分类器，validation 选择正则参数；训练报告不评测试集，最终通过 `score-classifier` 核对模型与测试文件哈希后独立评分。

生成仅用于接口检查的合成样本：

```bash
python tools/generate_sample_dataset.py --output data/synthetic-poses.jsonl
poseguard train --dataset data/synthetic-poses.jsonl --output models/synthetic-classifier.joblib \
  --report-dir runs/synthetic-training
```

合成样本成绩不能用于简历或真实效果报告。joblib 模型只加载可信来源文件。

## 6. 评估与性能

预测 JSONL 来自 `run`；真值使用完整人工框/标签 JSONL，不需要关键点，必须覆盖预测帧中全部人物。工具拒绝提取的 `pose_features` 充当真值。评估按视频/时间戳及人工 bbox IoU 一对一匹配，漏检计 FN、未匹配预测计 FP；报告人工框 IoU、轨迹碎片、ID 切换、分类指标及身份纯度。

真实跌倒事件 CSV 格式：

```csv
video_id,person_id,start,end
video-01,subject-01,3.0,8.0
```

```bash
poseguard evaluate --ground-truth data/frozen/test.ground-truth.jsonl \
  --predictions runs/video-test/predictions.jsonl \
  --events runs/video-test/events.csv --truth-events data/test-falls.csv \
  --output runs/evaluation.json
```

告警按发生时刻空间匹配关联人物；同事件只匹配一次，多余及未关联告警计 FP。告警必须在事件起点后、结束前且不超过 `max-alarm-delay`。报告事件 Precision/Recall/F1、媒体 P50/P95 延迟和加上 capture-to-event-write-call 时间的输出延迟，不包含上游摄像头/网络缓存。身份纯度之外还检查轨迹碎片和 ID 切换。

每次运行生成：

| 文件 | 内容 |
| --- | --- |
| `config.yaml` | 本次参数 |
| `provenance.json` | 配置/模型/分类器/动态库/输入/代码哈希、软硬件环境 |
| `health.jsonl` | 启动、读取故障、重连、重建及失败的原始记录 |
| `gpu.csv` | GPU 后端每秒的整卡显存、利用率与温度 |
| `predictions.jsonl` | 每人每帧 bbox、17 点、类别、概率、状态与时间戳 |
| `events.csv` | 逐人 fall/recovered 事件 |
| `frames.csv` | 采集读取、队列等待、预处理、推理/传输、后处理、分类跟踪、渲染、输出、延迟、RSS |
| `summary.json` | FPS、P50/P95、内存、丢帧、异常、重连统计 |
| `preview.jpg` / `fall_preview.jpg` | 首帧及第一次告警画面 |
| `annotated.mp4` | 可选叠加视频 |

`model_fps` = 后端平均耗时的倒数，包含传输，不包含采集/预后处理；`system_fps` = 处理帧数 / 运行墙钟时间。P50/P95 使用最近 20000 帧的有界样本；均值/最大值涵盖全程，CSV 可离线算全量分位数。RSS 是主机进程内存，不是 GPU 显存。H2D/GPU/D2H 仅由 TensorRT 后端测量，其他后端对应字段为空。模型基准有预热；摄像头摘要包含首次推理等运行开销。

`captured_ns` 为 OpenCV 返回图像的时间，因此 latency 是**主机收帧到输出**，不包括相机曝光、硬件缓存或网络上游缓冲。MP4 为固定源 FPS 写出，丢帧后播放会缩短，分析以日志时间戳为准。

连续运行示例：

```bash
poseguard run --config configs/camera.yaml --backend trt --model models/pose-fp32.engine \
  --duration 3600 --headless --output runs/soak-1h
python tools/soak_report.py --run runs/soak-1h --output runs/soak-1h/soak.json
```

长测报告提供 RSS 首尾段变化、P95 延迟、丢帧比、错误和重连。结合 `nvidia-smi` 外部记录显存与功耗，主动断开/恢复摄像头检查恢复行为。日志按帧写磁盘，长期无人值守需为 `runs` 配置清理策略。

## 7. C++ 推理集成

```bash
sudo apt-get install build-essential cmake libopencv-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTENSORRT_ROOT=/opt/TensorRT-10
cmake --build build -j

# 纯 C++ YOLO pose 摄像头示例：显示 bbox/关键点和推理耗时
./build/cpp/pose_camera models/pose-fp32.engine 0 640

# 完整姿态/跌倒/重连/评估流水线：Python 调用 C++ TensorRT 动态库
poseguard run --backend cpp --model models/pose-fp32.engine \
  --library build/cpp/libpose_trt.so --source 0 --output runs/cpp-camera
poseguard benchmark --backend cpp --model models/pose-fp16.engine \
  --library build/cpp/libpose_trt.so --output runs/cpp-benchmark.json
```

`Engine` 通过 RAII 管理 runtime、engine、context、device buffers、stream/events，显式校验输入输出容量，串行保护 context；C API 将异常转换为错误码与线程内错误信息。Python `ctypes` 接入后复用同一预后处理、分类、跟踪和告警逻辑。**纯 C++ 示例只做 pose 检测，完整跌倒识别通过 `--backend cpp` 集成运行。**

## 8. 目录

```text
configs/camera.yaml       默认摄像头配置
src/poseguard/            Python 包与 CLI
cpp/include/ cpp/src/     C++ TensorRT 库、C API、摄像头例程
tools/                   合成数据与长测报告工具
tests/                   几何、跟踪、泄漏、事件、ONNX 与视频流水线测试
models/ data/ runs/       权重、数据、运行输出（默认不入版本控制）
docs/                    设计与本机验证记录
.github/workflows/       Linux CPU 测试任务
```

详见 [系统设计](docs/architecture.md) 和 [本机验证记录](docs/validation.md)。项目代码提供研究和工程验证基础，实际安全告警应用还需验证场景覆盖、误报漏报及人工处置流程。
# 最新修复状态

2026-10-09 历史第二轮结果见 [逐项结果](docs/revised-results-2026-10-09.md)，最新证据见 [当前验收状态](docs/acceptance-status-2026-10-09.md)。已补 Windows 一小时内置摄像头、Ubuntu CPU 四视频和真实摄像头 RTSP 停流/恢复记录；这些结果不构成 Linux NVIDIA、物理 USB、独立人物最终准确率或完整 FP16 精度验收。
