# The two ESP32 arm boards: what they do besides the arms

Static analysis of the factory firmware (`left_firmware.bin`, `right_firmware.bin` from the original flash kit, ESP32-PICO-D4,
Arduino-ESP32 2.0.3, app at 0x10000), decompiled with Ghidra on 2026-11-04. Nothing was sent to the boards. Both
boards talk to the Jetson over UART0 (115200 8N1; left = `ttyTHS1` on header pins 8/10, right = USB serial) and to the
joint drives over CAN (GPIO32 TX / GPIO33 RX, 1 Mbit/s; joints RPDO 0x201+n, end board = node 7).

## Left board: base IO, motor-power relay, e-stop

| ESP32 GPIO | Use | Protocol |
|---|---|---|
| 4 | output: motor-power relay / enable | `0x10` POWER_ON: high, 1 s, restart CAN, enable drives. `0x11` POWER_OFF: low. High at boot (setup calls power-on) |
| 21 | input, pull-up: e-stop / motor power lost, **HIGH (or open) = tripped** | polled every ~2 ms, no opcode; see below |
| 36, 37, 38, 39, 34, 35 | base inputs IN1..IN6 (no internal pull-ups) | `FE FE 03 A1 n FA` (n = 1..6) -> `FE FE 04 A1 n level FA`; no automatic action |
| 19, 23, 18, 13, 14, 27 | base outputs OUT1..OUT6, low at boot | `FE FE 04 A0 n level FA`, no reply |
| 25, 26, 22 | outputs set 1 / 0 / 1 at boot, never changed (purpose unknown) | - |
| 12 | output, low, never written | - |
| 0 | set as OUTPUT low by an off-by-one in the pin table (boot strap pin, also driven by the Jetson's PAC.06 for flashing) | - |

E-stop: when GPIO21 reads high twice 10 ms apart while powered, the board zeroes the joint targets over CAN, sends
CANopen "stop all nodes", uninstalls the CAN driver, sets the robot's "stopped" flag (byte 0 of the `0xA2` status reply),
clears its power flag and sends an unsolicited `FE FE 03 12 00 FA` (IS_POWER_ON = 0). It does **not** switch GPIO4 off:
the hardware cuts the motor bus and GPIO21 senses it. Recovery: `0x10` power on (reinstalls CAN). Our driver polls the
power flag, so it notices this as "arm not powered".

## Right board: no base IO

- GPIO21 / 22 = UART1 RX / TX at 1 Mbit/s: Feetech STS bus servos ID 11 and 12 (jaw / neck head joints). Opcodes 0x1C,
  0x20 (read angle), 0x21, 0x22 (send angle), 0x2B, 0x50, 0x54, 0x56, 0x57, 0x13, 0x18 and 0x10 (torque on) reach them.
- No pin set-up, no e-stop poll, no `0xA0` / `0xA1`, and its power-on does not touch GPIO4.

## Not found on either board

No charge-detect, battery or power-good input: no ADC use and no other GPIO reads. A charge signal wired to the left
board could only be one of IN1..IN6 (GPIO 34-39), readable with `0xA1`.

## Hazards / quirks

- Do not send `0x60` SET_PIN_MODE: it reconfigures any ESP32 GPIO (a legacy map turns pin 22 into GPIO21, the e-stop
  input, and pin 4 is the relay).
- `0xD6` GET_QUICK_INFO reads GPIO 1..6 literally: only byte 5 (GPIO4, relay readback) means anything on the left.
- `0xD4` move_round is dead code (servo never attached) but busy-waits and blocks the dispatcher.
- End-effector commands (`0x61`, `0x62`, `0x66`, `0x67`, `0x6A`, `0x6D`, `0x6F`, `0x09`) are forwarded over CAN to node 7.
- A leftover M5Atom button on GPIO39 (sampled once at boot): if high at boot, errors from `0x07` are auto-cleared each loop.

Raw material (decompilation, disassembly, opcode trees, scripts): kept outside the repo; the conversion and Ghidra
scripts can be reproduced from the firmware images in the backup (`~/data/backup/mercury_x1/backup/flash_b1_orin/`).
