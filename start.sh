#!/bin/bash
# Bring up the whole Mercury X1 stack: drivers + sensors + control page (scripts/mx1_up.sh), then
# SLAM and Nav2 through the page's API, so the page owns them (its Stop / Start buttons keep working).
# Run at boot by mercury-x1.service (scripts/install_autostart.sh); safe to run again (does nothing if up).
#
#   ./start.sh [--map NAME] [--mode mapping|localization] [--fresh] [--no-slam] [--no-nav]
#
# Defaults come from ~/.config/mercury_x1/autostart.env (MX1_MAP, MX1_MODE, MX1_SLAM, MX1_NAV);
# without MX1_MAP the most recently saved map in ~/maps is used, in localization mode (mapping only
# when that map does not exist yet; --mode mapping continues it, --fresh starts it over). A restart from the page (ROS network switch) leaves resume.env,
# so SLAM / Nav2 come back as they were. ROS domain / network: scripts/ros_env.sh.
# Stop everything: scripts/mx1_stop.sh (or sudo systemctl stop mercury-x1).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_DIR="$HOME/.config/mercury_x1"
[ -f "$CONF_DIR/autostart.env" ] && source "$CONF_DIR/autostart.env"
if [ -f "$CONF_DIR/resume.env" ]; then source "$CONF_DIR/resume.env"; rm -f "$CONF_DIR/resume.env"; fi
# boot history: why the module last reset (SYS_RESET_N = external reset from the carrier, e.g. a supply
# dip; watchdog / software resets show up differently). Logged once per boot.
BOOT_LOG="$HOME/.ros/mercury_boot.log"; BOOT_ID=$(cat /proc/sys/kernel/random/boot_id)
if ! grep -qs "$BOOT_ID" "$BOOT_LOG"; then
  mkdir -p "$(dirname "$BOOT_LOG")"
  echo "$(uptime -s) boot $BOOT_ID reset_reason=$(cat /sys/devices/platform/bus@0/c360000.pmc/reset_reason 2>/dev/null || echo ?)" >> "$BOOT_LOG"
fi
MAP="${MX1_MAP:-}"; MODE="${MX1_MODE:-}"; SLAM="${MX1_SLAM:-true}"; NAV="${MX1_NAV:-true}"; FRESH=false
while [ $# -gt 0 ]; do
  case "$1" in
    --map) MAP="$2"; shift ;;
    --mode) MODE="$2"; shift ;;
    --fresh) FRESH=true ;;
    --no-slam) SLAM=false; NAV=false ;;
    --no-nav) NAV=false ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done
if [ -z "$MAP" ]; then
  MAP=$(ls -t "$HOME"/maps/*.db 2>/dev/null | head -1 | xargs -r basename | sed 's/\.db$//')
  MAP="${MAP:-default}"
fi
# localize on an existing map (mapping at every boot added a new, often unlinked session to the
# database each time); map only when there is no map yet. Mapping / extending: from the page.
if [ -z "$MODE" ]; then
  if [ -f "$HOME/maps/$MAP.db" ] && [ "$FRESH" != true ]; then MODE=localization; else MODE=mapping; fi
fi
PAGE="http://127.0.0.1:8080"
api() {   # api <action> [json body]
  local body="${2:-}"; [ -n "$body" ] || body='{}'
  curl -s -m 30 -X POST -H 'Content-Type: application/json' -d "$body" "$PAGE/api/$1"; echo
}
wait_for() {   # wait_for <seconds> <what> <command...>
  local t="$1" what="$2"; shift 2
  for i in $(seq 1 "$t"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
  echo "timed out waiting for $what"; return 1
}

if pgrep -f "mercury_x1_driver/lib/mercury_x1_driver/(base|upper_body)_node" >/dev/null; then
  echo "drivers already running"
else
  "$HERE/scripts/mx1_up.sh"
fi
wait_for 90 "the control page" curl -sf -m 2 "$PAGE/config"

if [ "$SLAM" = true ]; then
  if pgrep -x rtabmap >/dev/null; then
    echo "SLAM already running"
  else
    echo "SLAM: $MODE on map '$MAP'$([ "$FRESH" = true ] && echo ' (fresh)')"
    api slam "{\"mode\": \"$MODE\", \"map\": \"$MAP\", \"fresh\": $FRESH}"
  fi
fi
if [ "$NAV" = true ]; then
  if pgrep -x bt_navigator >/dev/null; then
    echo "Nav2 already running"
  else
    # Nav2's costmaps need map -> odom: start it once RTAB-Map publishes its first map. That can take
    # long (no odometry while the E-stop is pressed), so wait in the background: the unit must not
    # fail on it (systemd would stop the whole stack)
    (
      if wait_for 7200 "the first SLAM map" sh -c "curl -s -m 2 '$PAGE/map.json' | grep -q width"; then
        pgrep -x bt_navigator >/dev/null || { echo "Nav2"; api nav; }
      fi
    ) &
    echo "Nav2 starts once SLAM has a map"
  fi
fi
echo "stack up: http://$(hostname -I | awk '{print $1}'):8080  (logs /tmp/mx1_*.log, ~/.ros/mercury_x1_web/)"
