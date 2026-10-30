#!/bin/bash
# Start the Mercury X1 at boot: mercury-x1 (the stack, ./start.sh as elephant) and mercury-x1-head (the
# head display kiosk on tty1, needs `sudo apt install surf matchbox-window-manager`).
#   scripts/install_autostart.sh            install + enable (does not start them now)
#   scripts/install_autostart.sh --remove   disable + remove
# Afterwards: sudo systemctl start|stop|restart mercury-x1 / mercury-x1-head, journalctl -u <unit>
set -e
UNITS="mercury-x1 mercury-x1-head"
if [ "$1" = "--remove" ]; then
  for u in $UNITS; do
    sudo systemctl disable "$u" 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/$u.service"
  done
  sudo systemctl daemon-reload
  echo "autostart removed (running processes are left alone; scripts/mx1_stop.sh stops them)"
  exit 0
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for u in $UNITS; do sudo install -m 644 "$HERE/config/systemd/$u.service" "/etc/systemd/system/$u.service"; done
sudo systemctl daemon-reload
sudo systemctl enable $UNITS
echo "enabled: the stack and the head display start at boot (sudo systemctl start mercury-x1 mercury-x1-head)"
