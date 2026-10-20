# 故障路径实际结果

## 完成的真实 RTSP 故障

- MediaMTX 官方 Windows 资产完整取得：28,583,662 字节，SHA256 `a84224f5a490855684e62ba6fb71e80ea2a5e16aed61883bb58380290e93e467` 与 GitHub release API 的官方 digest 一致。使用完整档案，未执行此前局部下载文件。
- Windows 本地 RTSP/TCP 服务实际终止发布进程 4 秒、恢复，再终止服务 4 秒并恢复。最终 `runs/rtsp-loopback-fault-final-passed/fault-report.json` 为 **passed**：45.780 秒、839 个处理帧、1,001 个采集帧、7 次重连、0 推理故障。流代 0/5/7 均实际处理帧，两次故障前及恢复后的帧交付检查均通过，采集线程和子进程退出。
- 上述输入是 URFD 录制视频经 FFmpeg 发布，证明真实 RTSP 协议故障与客户端恢复；不是独立物理 RTSP 摄像头或交换机断线。

## Linux VMware 的真实摄像头 RTSP 恢复

- Windows 的 `HP True Vision FHD Camera` 实际捕获，经 FFmpeg 与 MediaMTX 在 VMware VMnet8 单一接口 `169.254.12.37:8554` 发布。服务未修改系统防火墙，也未监听 WLAN 或通配地址。设备是内置摄像头；USB PnP 设备路径不能证明它可进行外置 USB 物理拔插。
- 实际发布进程在北京时间 2026-10-09 20:17:53.384518 停止，20:17:57.390584 重启，间隔 **4.006066 秒**。UTC 原始记录见 `runs/windows-camera-rtsp-linux-soak-v3/faults.jsonl`。
- `/etc/os-release` 标识为 Ubuntu 22.04.5 LTS / Linux 6.8，实际 Python 3.12 / glibc 2.39 与更新软件包反映混合升级环境，不能当作干净标准 22.04 安装。ONNX Runtime CPU 客户端实际运行 **90.356 秒 / 635 帧 / 7.028 FPS**，记录 5 次重连，0 推理故障，流代 0 与 5 在故障前后均交付帧，进程和线程正常退出。恢复打开约在发布端重启后 **3.529 秒**；该跨主机时差由 `provenance.created_utc + health.elapsed_seconds` 估计，含主机/客体时钟和启动记录误差。
- 关联报告：`runs/remediation/linux-rtsp-fault-correlated.json`；Linux 原始输出：`runs/remediation/ubuntu-evidence/runs/linux-live-rtsp-fault/`。这是 Linux 真实摄像头 RTSP 软件/协议运行证据，摄像头采集驱动仍在 Windows 主机。
- FFmpeg 发布日志有重复帧，发布 30 FPS 不等于摄像头独立图像 30 FPS。主机延迟从 OpenCV 读取完成开始，未包含上游采集/编码缓冲。

## Linux VMware 一小时稳定运行

- 同一实际摄像头 RTSP 源在完成故障恢复后，Ubuntu 独立执行 **3,600.380 秒 / 22,173 个处理帧**，0 重连、0 推理故障。原始摘要 `synthetic=false`，source 为 VMnet8 RTSP，`capture_thread_alive=false`、`capture_worker_alive=false`；健康日志记录采集进程 12438 正常退出码 0。
- 帧时间跨度 **3,598.465 秒**、最大相邻处理帧间隔 **0.891 秒**。新的长测门槛允许最多 5 秒启动边缘开销并限制最大 5 秒断档；Ubuntu 原报告 `runs/remediation/ubuntu-evidence/runs/linux-live-rtsp-1h/soak.json` 的 min_seconds=3600 / max_rss_growth_mb=32 / max_frame_gap_seconds=5 显式稳定性门槛通过。
- 系统 **6.159 FPS**，全量处理延迟 P50 **237.096 ms**、P95 **321.473 ms**，RSS 前后段增长 **5.921 MB**。主机 RSS 只包含推理进程，不包含独立采集进程；结果不能证明没有内存泄漏。
- 丢帧 **79.465%**，反映该 VM CPU 无法按发布帧率完成全部推理。Ubuntu 主门槛没有设定丢帧通过阈值，稳定性通过不代表实时性能达标；不能将此结果当作 Linux GPU 速度验收。`runs/remediation/linux-soak-with-frame-gate.json` 是额外独立探索复核，其中 max_drop_ratio=0.9 也不能作为实时性能达标标准。
- 原始输出：`runs/remediation/ubuntu-evidence/runs/linux-live-rtsp-1h/`。服务会话日志也对应 20:20:39 到 21:20:38 的一次连续读取。真实摄像头采集仍位于 Windows 主机，不能替代 Linux 原生 USB 驱动实测。
- Windows 发布服务在北京时间 21:26:50 自然完成，`runs/windows-camera-rtsp-linux-soak-v3/server-report.json` 为 completed / 4,200.229 秒，最终日志哈希完整；MediaMTX 和 FFmpeg 进程已退出。

## 故障相关代码与回归

- `capture_worker` 打开设备前用启动时限，`source_opened` 后用读帧时限，防止慢打开被读帧看门狗过早终止。健康日志新增采集进程停止和重连计划事件。
- 回归验证慢打开、真实子进程阻塞后的终止/恢复、阻塞时退出、持续不可用时有限重试耗尽、RTSP 恢复后的帧交付及人工操作证据门槛。定向运行 `tests/test_pipeline.py tests/test_capture_process.py tests/test_device_fault_evidence.py`：**25 passed**。
- 新工具与现场命令见 `docs/fault-evidence.md`。`manual_fault_test.py` 要求交互终端、操作者与设备编号，并核对人工断开/恢复之后的实际故障和帧。当前未执行该工具的外置 USB 路径。

## 尚无实测的物理路径

当前没有外置 USB 摄像头，也没有独立真实 RTSP 硬件摄像头。物理 USB 拔插、物理网络拔线/交换机故障仍没有实测通过证据。不得将内置摄像头发布进程终止、VMware 菜单断开、软件回归或本地 RTSP 服务测试写成这些物理场景的通过结果。
