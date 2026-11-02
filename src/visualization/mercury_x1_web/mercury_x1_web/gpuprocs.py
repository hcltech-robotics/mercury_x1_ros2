"""GPU memory per process for the head display's Status tab.

On the Jetson the GPU shares the RAM; what each process holds for the GPU is in the nvmap debugfs table
(/sys/kernel/debug/nvmap/iovmm/clients, root only). jtop's root service reads it and hands it to the
`jtop` group over /run/jtop.sock, so web_server needs no root: jetson.processes =
[pid, user, gpu, type, priority, state, cpu %, ram kB, gpu kB, name (15 chars)].
The short names ("component_conta", "python") are resolved from /proc/<pid>/cmdline (ROS __node, script).
"""
from __future__ import annotations

import os
import threading
import time

# what the GPU users on this robot are (by resolved name)
WHAT = {
    "nvblox_node": "nvblox 3D reconstruction",
    "cuvslam_rgbd": "cuVSLAM visual odometry",
    "camera_container": "Orbbec camera (JPEG decode)",
    "camera": "Orbbec camera",
    "mercury-speech": "speech: Parakeet STT + Kokoro TTS",
    "mercury-tts": "Kokoro speech",
    "Xorg": "head display (X server)",
    "surf": "head display (browser)",
    "WebKitWebProcess": "head display (browser page)",
    "rtabmap": "RTAB-Map SLAM",
    "nav2_container": "Nav2",
}
SCRIPTS = {"tts_service.py": "mercury-tts", "speech_service.py": "mercury-speech"}


def _describe(pid: int, short: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            args = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return short
    for a in args:
        if a.startswith("__node:="):
            return a.split(":=", 1)[1]
    if not args:
        return short
    exe = os.path.basename(args[0])
    if exe.startswith("python"):
        for a in args[1:]:
            if a.endswith(".py"):
                return SCRIPTS.get(os.path.basename(a), os.path.basename(a))
        if "-m" in args[:-1]:
            return args[args.index("-m") + 1]
    return exe


class GpuProcs:
    def __init__(self, period: float = 2.0) -> None:
        self.period = period
        self.data: dict = {"procs": [], "error": "starting"}
        self._lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True, name="gpuprocs").start()

    def snapshot(self) -> dict:
        with self._lock:
            return self.data

    def _publish(self, raw: list) -> None:
        procs, idle = [], 0
        for p in raw:
            pid, cpu, ram_kb, gpu_kb, short = int(p[0]), p[6], p[7], p[8], str(p[9])
            if not gpu_kb:
                idle += 1          # has a GPU context but holds no GPU memory (e.g. rtabmap, the browser)
                continue
            name = _describe(pid, short)
            procs.append({"pid": pid, "name": name, "what": WHAT.get(name, ""), "gpu_mb": round(gpu_kb / 1024),
                          "ram_mb": round((ram_kb or 0) / 1024), "cpu": round(cpu or 0, 1)})
        procs.sort(key=lambda p: -p["gpu_mb"])
        with self._lock:
            self.data = {"procs": procs, "total_mb": sum(p["gpu_mb"] for p in procs), "idle": idle,
                         "t": time.time(), "error": ""}

    def _run(self) -> None:
        try:
            from jtop import jtop
        except ImportError:
            with self._lock:
                self.data = {"procs": [], "error": "jtop is not installed (sudo pip3 install jetson-stats)"}
            return
        while True:
            try:
                with jtop() as j:
                    last = 0.0
                    while j.ok():
                        if time.monotonic() - last >= self.period:
                            last = time.monotonic()
                            self._publish(j.processes)
            except Exception as e:  # noqa: BLE001 - jtop service down / socket permission: show it, retry
                with self._lock:
                    self.data = {"procs": [], "error": f"jtop: {str(e)[:120] or type(e).__name__}"}
            time.sleep(30)
