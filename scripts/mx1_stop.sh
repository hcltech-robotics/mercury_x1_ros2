#!/bin/bash
# Stop every Mercury X1 ROS process started by the bringup / ad-hoc tests (SIGINT, then SIGKILL).
PATTERNS='ros2 launch|ros2 run|mercury_x1_driver/|lslidar_driver_node|component_container|usb_cam_node_exe|robot_state_publisher|rtabmap|icp_odometry|rgbd_sync|point_cloud_xyz|nav2_|controller_server|planner_server|bt_navigator|behavior_server|smoother_server|waypoint_follower|velocity_smoother|lifecycle_manager|mercury_x1_web'
# never match ourselves or the shell(s) that invoked us (e.g. an ssh "bash -c" line)
skip=" $$ "; p=$$
while [ "$p" -gt 1 ]; do p=$(ps -o ppid= -p "$p" | tr -d ' '); skip="$skip$p "; done
pids=$(for x in $(pgrep -f -u "$USER" "$PATTERNS"); do case "$skip" in *" $x "*) ;; *) echo "$x";; esac; done)
[ -z "$pids" ] && { echo "nothing running"; exit 0; }
kill -INT $pids 2>/dev/null
for i in $(seq 1 20); do sleep 0.25; alive=$(ps -o pid= -p $pids 2>/dev/null); [ -z "$alive" ] && break; done
[ -n "$alive" ] && kill -KILL $alive 2>/dev/null
echo "stopped: $(echo $pids | wc -w) processes"
