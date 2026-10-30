"""Jetson CPU / GPU load and temperatures for the status page (plain sysfs / procfs, no jtop needed).

CPU load per core comes from /proc/stat deltas (1 s), GPU load from the GPU devfreq node's `load`
(instantaneous, 0..1000 -> sampled at 10 Hz and averaged over 1 s), temperatures from the thermal
zones (cpu-thermal, gpu-thermal, tj-thermal = hottest junction).
"""
from __future__ import annotations

import glob
import os
import threading
import time

GPU_DIR = "/sys/devices/platform/bus@0/17000000.gpu"


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_times() -> dict[str, tuple[int, int]]:
    """name -> (busy, total) jiffies for 'cpu' (all) and 'cpu0'.."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            name, *vals = line.split()
            v = [int(x) for x in vals[:8]]
            idle = v[3] + v[4]                      # idle + iowait
            out[name] = (sum(v) - idle, sum(v))
    return out


class SysStats:
    def __init__(self) -> None:
        self.zones = {}
        for z in glob.glob("/sys/class/thermal/thermal_zone*"):
            t = _read(os.path.join(z, "type"))
            if t in ("cpu-thermal", "gpu-thermal", "tj-thermal"):
                self.zones[t.split("-")[0]] = os.path.join(z, "temp")
        self.gpu_load_path = os.path.join(GPU_DIR, "load")
        self.gpu_freq_path = os.path.join(GPU_DIR, "devfreq/17000000.gpu/cur_freq")
        self.ncpu = os.cpu_count() or 1
        self.data: dict = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="sysstats").start()

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.data)

    def _run(self) -> None:
        prev = _cpu_times()
        gpu_samples: list[int] = []
        t_next = time.monotonic() + 1.0
        while True:
            v = _read(self.gpu_load_path)
            if v is not None and v.isdigit():
                gpu_samples.append(int(v))
            time.sleep(0.1)
            if time.monotonic() < t_next:
                continue
            t_next += 1.0
            cur = _cpu_times()
            pct = {}
            for name, (busy, total) in cur.items():
                pb, pt = prev.get(name, (busy, total))
                pct[name] = round(100.0 * (busy - pb) / (total - pt), 1) if total > pt else 0.0
            prev = cur
            temps = {}
            for key, path in self.zones.items():
                raw = _read(path)
                if raw and raw.lstrip("-").isdigit():
                    temps[key] = round(int(raw) / 1000.0, 1)
            freqs = []
            for i in range(self.ncpu):
                f = _read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq")
                freqs.append(int(f) // 1000 if f and f.isdigit() else None)
            gfreq = _read(self.gpu_freq_path)
            mem = {}
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        k, v = line.split(":", 1)
                        mem[k] = int(v.split()[0])
            except OSError:
                pass
            data = {
                "cpu": pct.get("cpu", 0.0),
                "cores": [pct.get(f"cpu{i}") for i in range(self.ncpu)],
                "core_mhz": freqs,
                "gpu": round(sum(gpu_samples) / len(gpu_samples) / 10.0, 1) if gpu_samples else None,
                "gpu_mhz": int(gfreq) // 1_000_000 if gfreq and gfreq.isdigit() else None,
                "temp": temps,
                "ram_used_mb": (mem["MemTotal"] - mem["MemAvailable"]) // 1024 if "MemAvailable" in mem else None,
                "ram_total_mb": mem.get("MemTotal", 0) // 1024 or None,
            }
            gpu_samples = []
            with self._lock:
                self.data = data
