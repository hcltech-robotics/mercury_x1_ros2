#!/bin/bash
# Bring up the Mercury X1: hardware (drivers, sensors, description) + Rerun bridge + control page.
# ./start.sh also starts SLAM / Nav2 (otherwise from the page, http://<robot>:8080). Logs: /tmp/mx1_{robot,web}.log
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ros_env.sh"   # ROS_DOMAIN_ID, DDS profile (localhost / lan)
if pgrep -f "mercury_x1_driver/lib/mercury_x1_driver/(base|upper_body)_node" >/dev/null; then
  echo "drivers already running - run scripts/mx1_stop.sh first"; exit 1
fi
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true   # stale Fast-DDS SHM locks
# extra robot.launch.py arguments, e.g. MX1_ROBOT_ARGS="voice_board:=false" in ~/.config/mercury_x1/autostart.env
AUTOSTART_ENV="${XDG_CONFIG_HOME:-$HOME/.config}/mercury_x1/autostart.env"
[ -z "${MX1_ROBOT_ARGS+x}" ] && [ -f "$AUTOSTART_ENV" ] && MX1_ROBOT_ARGS="$(. "$AUTOSTART_ENV"; echo "${MX1_ROBOT_ARGS:-}")"
# shellcheck disable=SC2086
setsid nohup ros2 launch mercury_x1_bringup robot.launch.py ${MX1_ROBOT_ARGS:-} > /tmp/mx1_robot.log 2>&1 < /dev/null &
sleep 6
setsid nohup ros2 launch mercury_x1_web web.launch.py > /tmp/mx1_web.log 2>&1 < /dev/null &
echo "up (ROS domain $ROS_DOMAIN_ID, network $MX1_ROS_NETWORK): http://$(hostname -I | awk '{print $1}'):8080  (viewer only: :9090)"
