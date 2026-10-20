"""Sample GPU device telemetry off the inference thread."""
import csv
import threading
import subprocess
import time
from pathlib import Path


class GpuTelemetry(threading.Thread):
    def __init__(self, output):
        super().__init__(name="poseguard-gpu-telemetry", daemon=True)
        self.output, self.stop = Path(output), threading.Event()
        self.error = None

    def run(self):
        with self.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["unix_ns", "gpu_index", "memory_used_mb", "utilization_percent", "temperature_c"])
            writer.writeheader()
            while not self.stop.is_set():
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
                    if result.returncode:
                        self.error = result.stderr.strip()
                        break
                    for row in csv.reader(result.stdout.splitlines()):
                        if len(row) == 4:
                            writer.writerow(dict(unix_ns=time.time_ns(), **dict(zip(writer.fieldnames[1:], row))))
                    stream.flush()
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self.error = str(exc)
                    break
                self.stop.wait(1)
