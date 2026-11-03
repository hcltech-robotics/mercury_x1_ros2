#!/bin/bash
# Restart the whole stack (used by the page's ROS network switch). Through systemd when the
# mercury-x1 unit runs it, otherwise stop + ./start.sh directly. Log: /tmp/mx1_restart.log
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if systemctl is-active --quiet mercury-x1; then
  exec sudo -n systemctl restart --no-block mercury-x1
fi
"$HERE/scripts/mx1_stop.sh"
exec "$HERE/start.sh"
