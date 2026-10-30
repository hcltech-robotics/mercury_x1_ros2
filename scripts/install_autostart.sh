#!/bin/bash
# Start the Mercury X1 stack at boot (systemd unit mercury-x1, runs ./start.sh as elephant).
#   scripts/install_autostart.sh            install + enable (does not start it now)
#   scripts/install_autostart.sh --remove   disable + remove
# Afterwards: sudo systemctl start|stop|restart mercury-x1, journalctl -u mercury-x1
set -e
UNIT=/etc/systemd/system/mercury-x1.service
if [ "$1" = "--remove" ]; then
  sudo systemctl disable mercury-x1 2>/dev/null || true
  sudo rm -f "$UNIT"; sudo systemctl daemon-reload
  echo "autostart removed (running processes are left alone; scripts/mx1_stop.sh stops them)"
  exit 0
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo install -m 644 "$HERE/config/systemd/mercury-x1.service" "$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable mercury-x1
echo "enabled: the stack starts at boot (sudo systemctl start mercury-x1 to start it now)"
