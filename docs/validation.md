# 本机验证记录

本页保留 2026-10-05 的 Windows 历史验证记录；后续 Ubuntu CPU 与有限混合精度结果见 [当前验收状态](acceptance-status-2026-10-09.md)。目标生产部署环境仍为 Linux + NVIDIA GPU。

验证日期：2026-10-05（Asia/Shanghai）。测试与合成演示不代表真实摄像头准确率或 TensorRT 性能。

## 环境

- Windows 11；项目独立 `.venv`，Python 3.12.13。
- CPU：Intel Core i9-13900HX；GPU：NVIDIA GeForce RTX 4070 Laptop GPU，8 GB。
- ONNX Runtime 1.20.1 CPUExecutionProvider、OpenCV 4.10.0、NumPy 1.26.4。
- scikit-learn 1.5.2、PyTorch 2.5.1+cpu、torchvision 0.20.1、Ultralytics 8.3.40、ONNX 1.17.0。
- CUDA Toolkit 开发组件 12.8.1，`nvcc` 12.8.93，包含 nvcc/cudart/CCCL/NVRTC，已编译并运行 CUDA kernel。
- TensorRT 10.8.0.43 Windows CUDA 12.8 SDK，Python 3.12 wheel，cuda-python 12.6.2.post1。
- C++ 编译器：Visual Studio 2022 MSVC 19.44.35228；Windows SDK 10.0.26100.0；OpenCV C++ SDK 4.10.0。
- SDK 安装目录：`C:\Users\zgl666\.local\poseguard-sdk`。现有 NVIDIA 驱动保留，项目启动时自动发现本机 CUDA/TensorRT DLL。

可复现版本在 `requirements-tested.txt` 与 `requirements-export-tested.txt`。其他传递依赖仍由包管理器解析；完整锁定时应同时保存部署环境的 `pip freeze`。

## 已执行

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| Python/GPU 回归测试 | 41 项通过，含 4 种尺寸与官方 LetterBox 逐像素一致、FP32/FP16 Python/C++ 输出一致性、GPU 重复调用与缓冲区检查 | `runs/gpu-tests.xml` |
| YOLO11n Pose 导出 | 静态 FP32 640×640，输出 `[1,56,8400]`，opset 17 | `models/yolo11n-pose.onnx.json` |
| PyTorch/ONNX 一致性 | 3 张含人图像变体通过；raw 坐标最大绝对误差 0.003510，分数最大误差 0.00001437；可见关键点最大误差 0.000173 原图像素 | `runs/torch-vs-onnx.json` |
| 实际 ONNX 预览 | bus 示例图检测出 4 人并绘制骨架，低质量骨架为 unknown | `runs/onnx-preview.jpg` |
| ONNX CPU 基准 | 20 次预热、100 次测量；call FPS 40.83；后端均值 23.93 ms、P50 23.75 ms、P95 25.58 ms | `runs/ort-benchmark.json` |
| 真实模型视频回放 | bus 图片重复成 20 帧 10 FPS 视频，全部处理，0 丢帧/异常/告警；系统 FPS 9.71，模型 FPS 41.03，主机输出延迟 P95 44.63 ms | `runs/onnx-video-smoke/summary.json` |
| 14 秒合成多人演示 | 421 帧；第一人 4.967 秒告警、12.133 秒恢复；第二人无告警；视频已解码及目视检查 | `runs/demo-final/` |
| 60 秒合成连续运行 | 1801 帧，30.01 FPS；0 丢帧/异常；4 次跌倒及 4 次恢复；主机延迟 P95 4.18 ms，RSS 峰值 80.84 MB，首尾段差约 0.61 MB | `runs/demo-soak-60s/soak.json` |
| CPU 分类器训练/加载 | 800 条合成样本，按 10 个人物/视频组划分 480/160/160；保存、加载及推理通过 | `runs/synthetic-training/` |
| C++ C API 语法 | `g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only -I cpp/include cpp/src/c_api.cpp` 通过 | 本地编译器 |
| CMake 完整构建 | MSVC Release 构建成功，生成 TensorRT 动态库与纯 C++ 摄像头程序 | `build-gpu/cpp/Release/pose_trt.dll`、`pose_camera.exe` |
| CUDA 编译和运行 | nvcc 编译 sm_89 kernel，GPU 输出通过检查 | `build-gpu/cuda_smoke.exe`、`tools/verify-cuda.ps1` |
| TensorRT FP32 Engine | 20,202,060 字节，禁用 TF32，在本机 RTX 4070 构建和加载成功 | `models/pose-fp32.engine.json` |
| TensorRT FP16 Engine | 11,124,604 字节，内部 FP16、I/O FP32，构建和加载成功 | `models/pose-fp16.engine.json` |
| FP32 与 ONNX 一致性 | 3 张变体完整通过默认阈值；raw 坐标最大误差 0.00232 | `runs/trt-fp32-consistency.json` |
| FP16 与 ONNX 一致性 | 后处理 3 张变体均通过，人数相同；最小 IoU 0.99643，可见关键点最大偏差 0.5022 px；raw 坐标检查未通过，详见下方说明 | `runs/trt-consistency.json` |
| FP32 GPU 基准 | 20 次预热、100 次测量；call FPS 264.15；GPU 均值 2.619 ms；backend P95 3.823 ms | `runs/trt-fp32-benchmark.json` |
| FP16 GPU 基准 | 20 次预热、100 次测量；call FPS 368.06；GPU 均值 1.651 ms；backend P95 2.667 ms | `runs/trt-fp16-benchmark.json` |
| C++ FP16 基准 | 20 次预热、100 次测量；call FPS 359.55；GPU 均值 1.695 ms；backend P95 3.040 ms | `runs/cpp-fp16-benchmark.json` |
| 纯 C++ 运行 | `pose_camera.exe ... --headless` 在实际 TensorRT 引擎上完成 20 帧视频推理 | `build-gpu/cpp/Release/pose_camera.exe` |
| 完整 C++ 后端流水线 | 20 帧全部处理，0 丢帧/异常/误告警；主机输出延迟 P95 14.48 ms | `runs/cpp-video-smoke/summary.json` |
| Python 依赖完整性 | `python -m pip check` 通过 | 本地包管理器 |

ONNX SHA256：`788bfd48f5bc8926e6b3a472632563341304a6d3bf266ee85cb3c313a6637ff3`。权重来源为 Ultralytics 官方 assets 的 YOLO11n Pose；bus 图片来自 Ultralytics 官方示例。三张转换图像都是同一图片的尺寸变体，不能作为泛化准确率评估集。20 帧回放也不代表真实动态视频。

回归测试还覆盖人物独立窗口、静态平躺抑制、单帧抑制、恢复去重、身份过期、故障后端重建、断连代数变化、不可用源失败摘要、训练数据隔离、漏检/未匹配惩罚，以及告警时刻 ID 切换的错误归属。

## FP16 精度结果

`runs/trt-consistency.json` 的整体 `passed=false` 是保留的真实结果：使用坐标 atol=2、rtol=0.01、分数 atol=0.03 比较所有 8400 个候选时，FP16 部分 raw 坐标偏差超过限值，最大 8.0175（模型坐标）；分数最大绝对偏差 0.031234，在相对/绝对组合阈值内通过。最终置信度筛选、NMS 和坐标还原后，所有人物数量一致，最小 IoU 0.99643，可见关键点最大偏差约 0.5022 原图像素。没有改阈值掩盖 raw 检查失败。

当前只能确认这三张同源图像上的后处理稳定；真实姿态和跌倒分类精度仍需独立标注集评估。严格要求 raw 一致性的场景可使用已通过严格检查的 FP32 引擎。基准是笔记本当前电源/温度/桌面负载下的单次测量，包含数据传输；它不代表摄像头系统 FPS。20 帧短回放包含首次推理开销，不能与预热后的 benchmark 直接相比。

## Windows 复现

```powershell
# C++ 环境设置与构建
. .\tools\gpu-env.ps1
.\tools\build-gpu.ps1
.\tools\verify-cuda.ps1

# GPU 与 C++ 回归
$env:POSEGUARD_TEST_GPU='1'
.\.venv\Scripts\python.exe -m pytest -q

# 完整 C++ 后端
.\.venv\Scripts\python.exe -m poseguard run --backend cpp --model models/pose-fp16.engine `
  --library build-gpu/cpp/Release/pose_trt.dll --source 0 --output runs/my-cpp-camera
```

CUDA 开发组件通过 NVIDIA 官方 redistrib JSON 的 SHA256 验证后合并安装；TensorRT 来自 NVIDIA 官方 SDK ZIP，OpenCV 来自官方发布的 SourceForge 包。TensorRT wheel 需在该 SDK 的 `python/` 下选择与解释器对应的文件；这里安装了 cp312 wheel。无需覆盖既有驱动或管理员级系统安装。

## 当前剩余目标验证

- Linux NVIDIA/C++ TensorRT 编译和性能；VMware Ubuntu CPU 已完成四视频回放，GPU/C++ 仍需目标设备重建。
- 外置 USB 物理拔插、独立 RTSP 摄像头与物理网络断线及上游缓存完整延迟。Windows 实摄像头到 Ubuntu 的 RTSP 停流恢复已完成，边界见当前状态。
- 独立人物/视频真实标注集的姿态及跌倒事件 Precision/Recall/F1。
- 8 小时验证仍未完成。Windows 内置摄像头 3601 秒、Ubuntu RTSP 3600 秒稳定性记录均已保存；Ubuntu 丢帧 79.46%，不代表实时性能通过，主进程 RSS 不覆盖采集子进程。

当前提供的 `models/synthetic-classifier.joblib` 仅为接口验证模型。默认真实摄像头配置使用未训练的规则基线，不能将合成训练的高分数作为真实准确率。
