"""Wi-Fi status / scan / connect through NetworkManager (nmcli), for the head display.

Changes go through `sudo -n nmcli` (the robot user has passwordless sudo): NetworkManager's polkit
rules only let an active local login session change connections, and the stack runs as a service.
Status is polled every 5 s in a thread, so the status websocket never waits on nmcli.
"""
from __future__ import annotations

import subprocess
import threading
import time


def _nmcli(*args: str, sudo: bool = False, timeout: float = 10.0) -> subprocess.CompletedProcess:
    cmd = (["sudo", "-n"] if sudo else []) + ["nmcli", "-t", *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _split(line: str) -> list[str]:
    """nmcli -t fields are ':'-separated with '\\:' escaping literal colons."""
    out, cur, esc = [], "", False
    for ch in line:
        if esc:
            cur += ch
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == ":":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


class Wifi:
    def __init__(self) -> None:
        self.device = self._find_device()
        self.status: dict = {"device": self.device, "connected": False}
        self.networks: list[dict] = []
        self.busy = ""                # "scanning" / "connecting to X" while an operation runs
        self.message = ""             # result of the last operation
        self._lock = threading.Lock()
        threading.Thread(target=self._poll, daemon=True, name="wifi").start()

    @staticmethod
    def _find_device() -> str:
        try:
            for line in _nmcli("-f", "DEVICE,TYPE", "device").stdout.splitlines():
                dev, typ = _split(line)[:2]
                if typ == "wifi":
                    return dev
        except (OSError, subprocess.SubprocessError):
            pass
        return ""

    def _poll(self) -> None:
        while True:
            try:
                self._update_status()
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(5.0)

    def _update_status(self) -> None:
        st: dict = {"device": self.device, "connected": False}
        if self.device:
            for line in _nmcli("-f", "ACTIVE,SSID,SIGNAL,CHAN,SECURITY", "device", "wifi", "list",
                               "ifname", self.device, "--rescan", "no").stdout.splitlines():
                active, ssid, signal, chan, sec = (_split(line) + [""] * 5)[:5]
                if active == "yes":
                    st.update(connected=True, ssid=ssid, signal=int(signal or 0), channel=chan, security=sec)
                    break
            for line in _nmcli("-f", "IP4.ADDRESS", "device", "show", self.device).stdout.splitlines():
                if line.startswith("IP4.ADDRESS"):
                    st["ip"] = _split(line)[1].split("/")[0]
                    break
        with self._lock:
            self.status = st

    def snapshot(self) -> dict:
        with self._lock:
            return {**self.status, "networks": list(self.networks), "busy": self.busy, "message": self.message}

    def saved(self) -> set[str]:
        names = set()
        for line in _nmcli("-f", "NAME,TYPE", "connection", "show").stdout.splitlines():
            name, typ = (_split(line) + ["", ""])[:2]
            if typ == "802-11-wireless":
                names.add(name)
        return names

    def scan(self) -> None:
        """blocking (~3-6 s): rescan and list networks, strongest first, one entry per SSID"""
        self.busy, self.message = "scanning", ""
        try:
            _nmcli("device", "wifi", "rescan", "ifname", self.device, sudo=True, timeout=15)
            time.sleep(2.0)
            best: dict[str, dict] = {}
            saved = self.saved()
            for line in _nmcli("-f", "IN-USE,SSID,SIGNAL,SECURITY", "device", "wifi", "list",
                               "ifname", self.device, "--rescan", "no").stdout.splitlines():
                inuse, ssid, signal, sec = (_split(line) + [""] * 4)[:4]
                if not ssid:
                    continue
                n = {"ssid": ssid, "signal": int(signal or 0), "secure": bool(sec and sec != "--"),
                     "active": inuse == "*", "saved": ssid in saved}
                if ssid not in best or n["signal"] > best[ssid]["signal"] or n["active"]:
                    best[ssid] = n
            self.networks = sorted(best.values(), key=lambda n: (not n["active"], -n["signal"]))
        except (OSError, subprocess.SubprocessError) as e:
            self.message = f"scan failed: {e}"
        finally:
            self.busy = ""

    def connect(self, ssid: str, password: str = "") -> None:
        """blocking (up to ~45 s). A saved network reconnects with its stored secret when no password is
        given. If the new network fails, NetworkManager falls back to the previous one by itself."""
        self.busy, self.message = f"connecting to {ssid}", ""
        try:
            if not password and ssid in self.saved():
                r = _nmcli("connection", "up", "id", ssid, sudo=True, timeout=45)
            else:
                args = ["device", "wifi", "connect", ssid, "ifname", self.device]
                if password:
                    args += ["password", password]
                r = _nmcli(*args, sudo=True, timeout=45)
            ok = r.returncode == 0
            self.message = f"connected to {ssid}" if ok else (r.stderr or r.stdout).strip().splitlines()[-1][:160]
        except (OSError, subprocess.SubprocessError, IndexError) as e:
            self.message = f"connect failed: {e}"
        finally:
            self.busy = ""
            self._update_status()
