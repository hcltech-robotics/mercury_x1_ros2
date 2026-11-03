"""Unitree BLE handheld remote (R3 and the older model) protocol.

From ~/Documents/dev/unitree_joy_bridge: GATT service ffe0; after connecting, write the ASCII
string "59532b32" (hex of "YS+2") to ffe2, then the remote streams 20-byte notifications on ffe1:

  0..15  float32 LE  lx, rx, ry, ly    (-1..1, right / up positive)
  16     uint8       buttons R1 L1 Start Select R2 L2 F1 F2   (bit 0..7)
  17     uint8       buttons A B X Y Up Right Down Left         (bit 0..7)
  18     uint8       battery %
  19     int8        rssi
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
WRITE_UUID = "0000ffe2-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
HANDSHAKE = "".join(f"{ord(c):x}" for c in "YS+2").encode()  # b"59532b32"
NAME_PREFIX = "Unitree"

BUTTONS = ["R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2",
           "A", "B", "X", "Y", "Up", "Right", "Down", "Left"]


@dataclass
class RemoteState:
    lx: float = 0.0   # right +
    ly: float = 0.0   # up +
    rx: float = 0.0
    ry: float = 0.0
    buttons: dict[str, bool] = field(default_factory=lambda: {b: False for b in BUTTONS})
    battery: int = -1
    rssi: int = 0


def decode(data: bytes) -> RemoteState | None:
    if len(data) < 20:
        return None
    lx, rx, ry, ly = struct.unpack_from("<4f", data, 0)
    bits = data[16] | (data[17] << 8)
    buttons = {name: bool((bits >> i) & 1) for i, name in enumerate(BUTTONS)}
    clip = lambda v: max(-1.0, min(1.0, v)) if v == v else 0.0  # NaN -> 0
    rssi = data[19] - 256 if data[19] >= 128 else data[19]
    return RemoteState(clip(lx), clip(ly), clip(rx), clip(ry), buttons, int(data[18]), rssi)


def joy_axes(s: RemoteState) -> list[float]:
    """ROS joy convention (as joy_node): stick left = +1, up = +1; d-pad as axes 4/5 (left/up = +1)."""
    hat_x = float(s.buttons["Left"]) - float(s.buttons["Right"])
    hat_y = float(s.buttons["Up"]) - float(s.buttons["Down"])
    return [-s.lx, s.ly, -s.rx, s.ry, hat_x, hat_y]
