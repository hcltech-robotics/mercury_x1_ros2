#!/usr/bin/env python3
"""Live state of the Jetson's 40-pin header for the kiosk Debug tab (systemd unit mercury-pins, runs as root).

Strictly read-only: the GPIO and pinmux registers are read through pins_read (/dev/mem opened O_RDONLY, mapped
PROT_READ). No pin is requested, exported, re-muxed or driven - unlike gpiod / sysfs, which on Tegra switch a pin
to GPIO mode while they hold it. The arm-controller programming pins (EN / G0) and the left arm's UART are not
even read. Writes /run/mercury-pins/pins.json 4x a second; the web server's "pins" API action serves it.

Header pin map: Orin NX / Nano carrier standard (the factory flash scripts' G4/G5/G23/G24 labels match it).
Registers: drivers/gpio/gpio-tegra186.c and drivers/pinctrl/tegra/pinctrl-tegra234.c (Linux 6.8).
"""
from __future__ import annotations

import json
import os
import subprocess
import time

READER = "/usr/local/lib/mercury-pins/pins_read"
OUT_DIR = "/run/mercury-pins"
PERIOD_S = 0.25

MAIN_GPIO, AON_GPIO = 0x02210000, 0x0C2F1000          # "gpio" register regions (device tree reg-names)
MAIN_MUX, AON_MUX = 0x02430000, 0x0C300000
MAIN_PORTS = {"G": (4, 0), "H": (4, 1), "I": (4, 2), "N": (2, 1), "Q": (2, 3), "R": (2, 4), "Y": (1, 1),
              "Z": (1, 2), "AC": (0, 1)}                # port -> (bank, port index)
AON_PORTS = {"CC": (0, 2), "DD": (0, 3)}

# pin: (signal on the header, Tegra pin "PORT.n", pinmux offset, functions 0..3) - or a fixed / reserved role
P = {
    1: "3.3 V", 2: "5 V", 4: "5 V", 17: "3.3 V",
    6: "GND", 9: "GND", 14: "GND", 20: "GND", 25: "GND", 30: "GND", 34: "GND", 39: "GND",
    7: ("reserved", "PAC.06", "left arm controller G0 (boot mode, firmware flashing)"),
    29: ("reserved", "PQ.05", "left arm controller EN (reset)"),
    16: ("reserved", "PY.04", "right arm controller G0 (boot mode, firmware flashing)"),
    18: ("reserved", "PY.03", "right arm controller EN (reset)"),
    8: ("reserved", "PR.02", "UART1 TX: left arm (ttyTHS1)"),
    10: ("reserved", "PR.03", "UART1 RX: left arm (ttyTHS1)"),
    11: ("reserved", "PR.04", "UART1 RTS (left arm UART)"),
    36: ("reserved", "PR.05", "UART1 CTS (left arm UART)"),
    3: ("I2C1_SDA", "PDD.02", 0x2018, ("I2C8", "RSVD1", "RSVD2", "RSVD3")),
    5: ("I2C1_SCL", "PDD.01", 0x2020, ("I2C8", "RSVD1", "RSVD2", "RSVD3")),
    27: ("I2C0_SDA", "PDD.00", 0x2040, ("I2C2", "RSVD1", "RSVD2", "RSVD3")),
    28: ("I2C0_SCL", "PCC.07", 0x2030, ("I2C2", "RSVD1", "RSVD2", "RSVD3")),
    12: ("I2S0_SCLK", "PH.07", 0x4088, ("RSVD0", "I2S2", "RSVD2", "RSVD3")),
    13: ("SPI1_SCK", "PY.00", 0xD030, ("SPI3", "RSVD1", "RSVD2", "RSVD3")),
    15: ("GPIO12", "PN.01", 0x10020, ("GP", "RSVD1", "RSVD2", "RSVD3")),
    19: ("SPI0_MOSI", "PZ.05", 0xD040, ("SPI1", "RSVD1", "RSVD2", "RSVD3")),
    21: ("SPI0_MISO", "PZ.04", 0xD018, ("SPI1", "RSVD1", "RSVD2", "RSVD3")),
    22: ("SPI1_MISO", "PY.01", 0xD000, ("SPI3", "RSVD1", "RSVD2", "RSVD3")),
    23: ("SPI0_SCK", "PZ.03", 0xD028, ("SPI1", "RSVD1", "RSVD2", "RSVD3")),
    24: ("SPI0_CS0", "PZ.06", 0xD008, ("SPI1", "RSVD1", "RSVD2", "RSVD3")),
    26: ("SPI0_CS1", "PZ.07", 0xD038, ("SPI1", "RSVD1", "RSVD2", "RSVD3")),
    31: ("GPIO11", "PQ.06", 0x0070, ("RSVD0", "EXTPERIPH4", "DCB", "RSVD3")),
    32: ("GPIO07", "PG.06", 0x4080, ("GP", "RSVD1", "RSVD2", "RSVD3")),
    33: ("GPIO13", "PH.00", 0x4040, ("RSVD0", "GP", "I2S7", "RSVD3")),
    35: ("I2S0_FS", "PI.02", 0x40A0, ("RSVD0", "I2S2", "RSVD2", "RSVD3")),
    37: ("SPI1_MOSI", "PY.02", 0xD048, ("SPI3", "RSVD1", "RSVD2", "RSVD3")),
    38: ("I2S0_DIN", "PI.01", 0x4098, ("RSVD0", "I2S2", "RSVD2", "RSVD3")),
    40: ("I2S0_DOUT", "PI.00", 0x4090, ("RSVD0", "I2S2", "RSVD2", "RSVD3")),
}
PULL = ("none", "down", "up", "rsvd")


def gpio_addr(tegra: str) -> int:
    port, n = tegra[1:].split(".")
    if port in AON_PORTS:
        bank, idx = AON_PORTS[port]
        base = AON_GPIO
    else:
        bank, idx = MAIN_PORTS[port]
        base = MAIN_GPIO
    return base + bank * 0x1000 + idx * 0x200 + int(n) * 0x20


def mux_addr(tegra: str, off: int) -> int:
    return (AON_MUX if tegra[1:].split(".")[0] in AON_PORTS else MAIN_MUX) + off


def readable() -> dict[int, tuple]:
    return {pin: v for pin, v in P.items() if isinstance(v, tuple) and v[0] != "reserved"}


def read_all(pins: dict[int, tuple]) -> dict[int, dict]:
    addrs = []
    for _, (_, tegra, off, _) in sorted(pins.items()):
        g = gpio_addr(tegra)
        addrs += [g, g + 0x08, g + 0x10, mux_addr(tegra, off)]     # ENABLE_CONFIG, INPUT, OUTPUT_VALUE, PADCTL
    out = subprocess.run([READER, *[hex(a) for a in addrs]], capture_output=True, text=True, timeout=2).stdout
    vals = {}
    for line in out.splitlines():
        a, v = line.split()[:2]
        vals[int(a, 16)] = int(v, 16) if v.startswith("0x") else None
    res = {}
    for pin, (sig, tegra, off, funcs) in sorted(pins.items()):
        g = gpio_addr(tegra)
        en, inp, outv, mux = vals.get(g), vals.get(g + 8), vals.get(g + 0x10), vals.get(mux_addr(tegra, off))
        if None in (en, inp, outv, mux):
            res[pin] = {"error": "read failed"}
            continue
        gpio_mode = not (mux >> 10) & 1 and en & 1
        res[pin] = {
            # what the pad reads now; the INPUT register only samples while the pin's GPIO block is enabled
            "level": inp & 1 if en & 1 else None,
            "mode": "GPIO" if gpio_mode else funcs[mux & 3],
            "dir": ("out" if en & 2 else "in") if gpio_mode else "",
            "out": outv & 1 if gpio_mode and en & 2 else None,
            "pull": PULL[(mux >> 2) & 3],
            "tristate": bool(mux & 0x10),
            "input_en": bool(mux & 0x40),
            "raw": {"cfg": en, "mux": mux},
        }
    return res


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.chmod(OUT_DIR, 0o755)
    pins = readable()
    last: dict[int, int] = {}
    changed_at: dict[int, float] = {}
    changes: dict[int, int] = {}
    while True:
        t = time.time()
        try:
            state = read_all(pins)
            err = ""
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            state, err = {}, str(e)[:200]
        for pin, s in state.items():
            lv = s.get("level")
            if lv is not None and pin in last and lv != last[pin]:
                changed_at[pin] = t
                changes[pin] = changes.get(pin, 0) + 1
            if lv is not None:
                last[pin] = lv
        header = []
        for pin in range(1, 41):
            v = P[pin]
            if isinstance(v, str):
                header.append({"pin": pin, "kind": "power" if "V" in v else "ground", "name": v})
            elif v[0] == "reserved":
                header.append({"pin": pin, "kind": "reserved", "name": v[1], "tegra": v[1], "role": v[2]})
            else:
                header.append({"pin": pin, "kind": "io", "name": v[0], "tegra": v[1], **state.get(pin, {}),
                               "changed_at": changed_at.get(pin), "changes": changes.get(pin, 0)})
        doc = {"t": t, "error": err, "pins": header}
        tmp = os.path.join(OUT_DIR, "pins.json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, os.path.join(OUT_DIR, "pins.json"))
        time.sleep(max(0.0, PERIOD_S - (time.time() - t)))


if __name__ == "__main__":
    main()
