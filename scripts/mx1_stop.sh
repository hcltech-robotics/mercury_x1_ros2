#!/bin/bash
# Stop every Mercury X1 ROS process started by the bringup / ad-hoc tests (SIGINT, then SIGKILL).
PATTERNS='ros2 launch|ros2 run|mercury_x1_driver/|lslidar_driver_node|component_container|usb_cam_node_exe|robot_state_publisher|rtabmap|icp_odometry|rgbd_sync|point_cloud_xyz|nav2_|controller_server|planner_server|bt_navigator|behavior_server|smoother_server|waypoint_follower|velocity_smoother|lifecycle_manager|mercury_x1_web|nvblox_node|nvblox_mesh_tiler'
# never match ourselves or the shell(s) that invoked us (e.g. an ssh "bash -c" line)
skip=" $$ "; p=$$
while [ "$p" -gt 1 ]; do p=$(ps -o ppid= -p "$p" | tr -d ' '); skip="$skip$p "; done
# ... nor any shell / ssh / editor whose command line merely mentions a pattern (run by systemd, the
# ancestors above are systemd, so an unrelated ssh "bash -c '... mercury_x1_web ...'" would be hit)
find_pids() {
  for x in $(pgrep -f -u "$USER" "$PATTERNS"); do
    case "$skip" in *" $x "*) continue ;; esac
    case "$(ps -o comm= -p "$x")" in bash|sh|dash|zsh|ssh|sshd|sudo|timeout|tail|less|grep|vim|nano) continue ;; esac
    echo "$x"
  done
}
total=0
# loop: `ros2 launch` may respawn nodes while it is shutting down
for round in 1 2 3 4 5; do
  pids=$(find_pids)
  [ -z "$pids" ] && break
  total=$((total + $(echo $pids | wc -w)))
  kill -INT $pids 2>/dev/null
  for i in $(seq 1 20); do sleep 0.25; alive=$(ps -o pid= -p $pids 2>/dev/null); [ -z "$alive" ] && break; done
  [ -n "$alive" ] && kill -KILL $alive 2>/dev/null
  sleep 0.5
done
left=$(find_pids)
[ -n "$left" ] && { echo "still running: $left"; exit 1; }
echo "stopped: $total processes"
