#!/bin/bash
# Install the read-only header pin monitor (kiosk Debug tab): build pins_read, copy it and pins_service.py to a
# root-owned directory (the service runs as root, so it must not run user-writable files), enable mercury-pins.
#   server/robot_pins/setup.sh            install / update + (re)start
#   server/robot_pins/setup.sh --remove   stop + remove
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
D=/usr/local/lib/mercury-pins
if [ "$1" = "--remove" ]; then
  sudo systemctl disable --now mercury-pins 2>/dev/null || true
  sudo rm -rf /etc/systemd/system/mercury-pins.service "$D"
  sudo systemctl daemon-reload
  exit 0
fi
gcc -O2 -Wall -o /tmp/pins_read "$HERE/pins_read.c"
sudo install -d -m 755 "$D"
sudo install -m 755 /tmp/pins_read "$D/pins_read"
sudo install -m 644 "$HERE/pins_service.py" "$D/pins_service.py"
sudo install -m 644 "$REPO/config/systemd/mercury-pins.service" /etc/systemd/system/mercury-pins.service
sudo systemctl daemon-reload
sudo systemctl enable mercury-pins
sudo systemctl restart mercury-pins
sleep 2
python3 -c "import json; d = json.load(open('/run/mercury-pins/pins.json')); print('pins.json ok,', sum(p['kind'] == 'io' for p in d['pins']), 'pins read', d['error'])"
