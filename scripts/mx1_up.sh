#!/bin/bash
# Bring up the Mercury X1: hardware (drivers, sensors, description) + Rerun bridge + control page.
# SLAM / Nav2 are started from the page (http://<robot>:8080). Logs: /tmp/mx1_{robot,web}.log
set -e
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-175}
source /opt/ros/humble/setup.bash
source "$HOME/mercury_x1_ros2/install/setup.bash"
# 16 MB Fast-DDS shared-memory segments: 640x480 images exceed the 512 KB default and were half dropped
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/mercury_x1_ros2/config/fastdds_shm.xml"
if pgrep -f "mercury_x1_driver/lib/mercury_x1_driver/(base|upper_body)_node" >/dev/null; then
  echo "drivers already running - run scripts/mx1_stop.sh first"; exit 1
fi
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null || true   # stale Fast-DDS SHM locks
setsid nohup ros2 launch mercury_x1_bringup robot.launch.py > /tmp/mx1_robot.log 2>&1 < /dev/null &
sleep 6
setsid nohup ros2 launch mercury_x1_web web.launch.py > /tmp/mx1_web.log 2>&1 < /dev/null &
echo "up: http://$(hostname -I | awk '{print $1}'):8080  (viewer only: :9090)"
