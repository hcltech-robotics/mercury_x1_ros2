# ROS 2 environment of the Mercury X1 - source it (start.sh, mx1_up.sh, ~/mx1_env.sh all do).
#   ROS_DOMAIN_ID 176
#   ROS network: "localhost" (default, no ROS traffic leaves the robot -> no DDS multicast on the WiFi)
#                or "lan" (other machines can join), from ~/.config/mercury_x1/ros_network.
#                Switch on the control page (Robot tab) - it restarts the stack, since every ROS process
#                picks its transports when it starts.
export MX1_WS="${MX1_WS:-$HOME/mercury_x1_ros2}"
export ROS_DOMAIN_ID=176
MX1_ROS_NETWORK=$(cat "$HOME/.config/mercury_x1/ros_network" 2>/dev/null || echo localhost)
case "$MX1_ROS_NETWORK" in lan) ;; *) MX1_ROS_NETWORK=localhost ;; esac
export MX1_ROS_NETWORK
# 16 MB shared-memory segments (640x480 images exceed the 512 KB default and were half dropped) +
# UDP on loopback only (localhost) or on every interface (lan)
export FASTRTPS_DEFAULT_PROFILES_FILE="$MX1_WS/config/fastdds_$MX1_ROS_NETWORK.xml"
if [ "$MX1_ROS_NETWORK" = localhost ]; then export ROS_LOCALHOST_ONLY=1; else unset ROS_LOCALHOST_ONLY; fi
source /opt/ros/humble/setup.bash
[ -f "$MX1_WS/install/setup.bash" ] && source "$MX1_WS/install/setup.bash"
