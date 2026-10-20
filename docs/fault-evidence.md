# 摄像头故障证据入口

当前已完成的 Ubuntu 实摄像头 RTSP 停流恢复与设备边界见 [验收状态](acceptance-status-2026-10-09.md)。Windows 内置摄像头经 FFmpeg/MediaMTX 发流到 Ubuntu，已保存 90 秒客户端故障/恢复记录；它不代替外置 USB 拔插、独立 RTSP 硬件或物理网络断线。

`tools/device_fault_evidence.py` 在当前主机枚举摄像头、检查给定 RTSP 端口和读帧，并执行真实采集子进程阻塞/重启回归。每次使用新的输出目录：

```powershell
.\.venv\Scripts\python.exe tools/device_fault_evidence.py --output runs/device-preflight
```

```bash
python tools/device_fault_evidence.py --output runs/device-preflight --camera-index 0 --rtsp rtsp://camera-host/live
```

输出 `device-fault-report.json` 区分读帧、软件看门狗和物理故障。OpenCV 摄像头索引不能证明设备为外置 USB，设备分类需人工确认。RTSP URL 的用户密码及查询参数不会写入探测报告；包含凭据的命令行仍可能进入 shell 历史，优先在受保护的配置文件中管理连接信息。

## 本地真实 RTSP 协议故障

`tools/rtsp_fault_test.py` 用完整且经官方 SHA256 校验的 MediaMTX 与 FFmpeg 发布真实录制视频，然后实际终止/恢复发布进程和 RTSP 服务进程。它验证 TCP/RTSP 故障和客户端恢复，不能替代真实远端摄像头与物理网络断线。

```powershell
.\.venv\Scripts\python.exe tools/rtsp_fault_test.py --output runs/rtsp-fault --archive runs/mediamtx-retry.zip --model models/pose-fp32.engine
```

```bash
python tools/rtsp_fault_test.py --output runs/rtsp-fault-linux --server /path/to/mediamtx --backend onnx --model models/yolo11n-pose.onnx
```

通过判定要求：两次故障实际执行、客户端正常完成、至少两次重连、至少三个流代打开并交付帧、采集线程及子进程退出。`faults.jsonl` 是 UTC 故障时刻，`client/health.jsonl` 是看门狗/读取失败/重连日志，`client/frames.csv` 记录各代实际处理帧。

## 现场 USB / 远端 RTSP 故障

有外置 USB 摄像头或真实远端 RTSP 摄像头后，在可交互终端运行：

```powershell
.\.venv\Scripts\python.exe tools/manual_fault_test.py --config configs/usb-test.yaml --output runs/physical-usb-fault --device-kind external-usb --device-id CAMERA-SERIAL --operator REVIEWER --duration 180
```

```bash
python tools/manual_fault_test.py --config configs/rtsp-test.yaml --output runs/remote-rtsp-fault --device-kind remote-rtsp --device-id CAMERA-ASSET-ID --operator REVIEWER --duration 180
```

初始帧到达后按终端窗口提示操作设备，断开和恢复后各按 Enter，工具记录操作者、设备编号、UTC 时刻，并核对动作后读取失败以及恢复后的实际帧。结果名称 `passed_operator_attested` 明确物理动作由操作者确认，软件验证故障与恢复结果；现场照片/视频可作为补充证据。

内置摄像头不能做外置 USB 物理拔插。VMware 可以将已存在的外置 USB 设备连接给 Ubuntu，但虚拟机菜单中的断开不等同于物理拔插。`manual_fault_test.py` 的 `client.yaml` 会保留连接凭据，相关证据目录应私有保存。

## 内置摄像头供 Linux VM 长测

`tools/serve_camera_rtsp.py` 将 Windows 实际摄像头用 FFmpeg 编码为 RTSP，仅绑定指定的 VMware 主机接口，不修改系统防火墙：

```powershell
.\.venv\Scripts\python.exe tools/serve_camera_rtsp.py --server runs/rtsp-loopback-fault-completed/mediamtx.exe --output runs/camera-rtsp-linux --bind HOST_VMNET_IP --port 8554 --duration 4200
```

Ubuntu 的 `source` 使用 `rtsp://HOST_VMNET_IP:8554/soak`。该测试可以证明 Linux 在真实摄像头帧与实际 RTSP 通信条件下运行；摄像头仍位于 Windows 主机，不能写成 Linux 原生 USB 驱动或独立 RTSP 硬件摄像头验收。主机绑定地址必须来自 `Get-NetIPAddress` 的实际 VMware 接口 IP。
