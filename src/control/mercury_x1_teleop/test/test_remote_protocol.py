import struct

from mercury_x1_teleop.remote_protocol import BUTTONS, HANDSHAKE, decode, joy_axes


def packet(lx=0.0, rx=0.0, ry=0.0, ly=0.0, pressed=(), battery=87, rssi=-60):
    bits = 0
    for name in pressed:
        bits |= 1 << BUTTONS.index(name)
    return struct.pack("<4f", lx, rx, ry, ly) + bytes([bits & 0xFF, bits >> 8, battery, rssi & 0xFF])


def test_handshake():
    assert HANDSHAKE == b"59532b32"


def test_decode_sticks_buttons_battery():
    s = decode(packet(lx=0.5, ly=-1.0, rx=0.25, ry=0.75, pressed=("L1", "B", "Up")))
    assert (s.lx, s.ly, s.rx, s.ry) == (0.5, -1.0, 0.25, 0.75)
    assert [b for b in BUTTONS if s.buttons[b]] == ["L1", "B", "Up"]
    assert s.battery == 87 and s.rssi == -60


def test_short_and_nan():
    assert decode(b"\x00" * 19) is None
    s = decode(packet(lx=float("nan"), ly=3.0))
    assert s.lx == 0.0 and s.ly == 1.0


def test_joy_axes_ros_convention():
    s = decode(packet(lx=1.0, ly=1.0, pressed=("Left", "Up")))
    assert joy_axes(s) == [-1.0, 1.0, -0.0, 0.0, 1.0, 1.0]
