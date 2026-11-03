"""Head voice board driver.  ros2 launch mercury_x1_voice_board voice_board.launch.py [port:=/dev/ttyACM3]"""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(share("mercury_x1_voice_board"), "config", "voice_board.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/xf_mic"),
        Node(package="mercury_x1_voice_board", executable="voice_board_node", name="voice_board",
             parameters=[params, {"port": LC("port")}], output="screen", respawn=True, respawn_delay=2.0),
    ])
