#!/bin/bash
# Install the udev rule for the head voice board control UART (-> /dev/xf_mic, mode 0666). Run on the robot.
set -e
RULE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/drivers/mercury_x1_voice_board/udev/99-mercury-voice-board.rules"
sudo install -m 0644 "$RULE" /etc/udev/rules.d/99-mercury-voice-board.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=add
sleep 1
ls -l /dev/xf_mic
