"""Jetson CPU / GPU load and temperatures for the status page (plain sysfs / procfs, no jtop needed).

CPU load per core comes from /proc/stat deltas (1 s), GPU load from the GPU devfreq node's `load`
(instantaneous, 0..1000 -> sampled at 10 Hz and averaged over 1 s), temperatures from the thermal
zones (cpu-thermal, gpu-thermal, tj-thermal = hottest junction).

Power / clock config: the nvpmodel mode (id in /var/lib/nvpmodel/status, names from /etc/nvpmodel.conf),
the INA3221 rails (VDD_IN = whole module, VDD_CPU_GPU_CV, VDD_SOC), CPU / GPU frequency limits,
whether jetson_clocks pinned them (min = max) and the fan (pwm 0..255, tach rpm).
"""
from __future__ import annotations

import glob
import os
import re
import threading
import time

GPU_DIR = "/sys/devices/platform/bus@0/17000000.gpu"
GPU_DEVFREQ = os.path.join(GPU_DIR, "devfreq/17000000.gpu")
NVPMODEL_CONF = "/etc/nvpmodel.conf"
NVPMODEL_STATUS = "/var/lib/nvpmodel/status"


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _int(path: str) -> int | None:
    v = _read(path)
    return int(v) if v is not None and v.lstrip("-").isdigit() else None


def _power_modes() -> dict[int, str]:
    try:
        with open(NVPMODEL_CONF) as f:
            return {int(i): n for i, n in re.findall(r"<\s*POWER_MODEL\s+ID=(\d+)\s+NAME=(\S+)\s*>", f.read())}
    except OSError:
        return {}


def _hwmon(name: str) -> str | None:
    for d in glob.glob("/sys/class/hwmon/hwmon*"):
        if _read(os.path.join(d, "name")) == name:
            return d
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
        self.gpu_freq_path = os.path.join(GPU_DEVFREQ, "cur_freq")
        self.modes = _power_modes()
        ina = _hwmon("ina3221")
        self.rails = {}                       # label -> (mV path, mA path)
        if ina:
            for lbl in glob.glob(os.path.join(ina, "in*_label")):
                k = lbl[len(os.path.join(ina, "in")):-len("_label")]
                self.rails[_read(lbl)] = (os.path.join(ina, f"in{k}_input"), os.path.join(ina, f"curr{k}_input"))
        self.fan_pwm = (_hwmon("pwmfan") or "") + "/pwm1"
        self.fan_rpm = (_hwmon("pwm_tach") or "") + "/rpm"
        self.ncpu = os.cpu_count() or 1
        self.data: dict = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="sysstats").start()

    def _power(self) -> dict:
        st = _read(NVPMODEL_STATUS) or ""
        m = re.search(r"pmode:(\d+)", st)
        mode_id = int(m.group(1)) if m else None
        watts = {}
        for label, (mv, ma) in self.rails.items():
            v, i = _int(mv), _int(ma)
            if label and v is not None and i is not None and "sum" not in label:
                watts[label] = round(v * i / 1e6, 2)
        cpu_min = _int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq")
        cpu_max = _int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
        gpu_min, gpu_max = _int(os.path.join(GPU_DEVFREQ, "min_freq")), _int(os.path.join(GPU_DEVFREQ, "max_freq"))
        gpu_top = (_read(os.path.join(GPU_DEVFREQ, "available_frequencies")) or "0").split()[-1]
        pwm, rpm = _int(self.fan_pwm), _int(self.fan_rpm)
        return {
            "mode": self.modes.get(mode_id, f"mode {mode_id}") if mode_id is not None else None,
            "watts": watts,                                     # VDD_IN = whole module
            "cpu_mhz_min": cpu_min // 1000 if cpu_min else None, "cpu_mhz_max": cpu_max // 1000 if cpu_max else None,
            "gpu_mhz_min": gpu_min // 1_000_000 if gpu_min else None,
            "gpu_mhz_max": gpu_max // 1_000_000 if gpu_max else None,
            "gpu_mhz_top": int(gpu_top) // 1_000_000 if gpu_top.isdigit() else None,
            "clocks_pinned": bool(cpu_min and cpu_min == cpu_max and gpu_min and gpu_min == gpu_max),   # jetson_clocks
            "fan_pct": round(pwm * 100 / 255) if pwm is not None else None,
            "fan_rpm": rpm,
        }

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
            data["power"] = self._power()
            gpu_samples = []
            with self._lock:
                self.data = data
