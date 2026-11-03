"""Low-level goalkeeper services: base + upper body."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory("mercury_x1_driver"), "config", "drivers.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("base", default_value="true"),
        DeclareLaunchArgument("upper_body", default_value="true"),
        DeclareLaunchArgument("params", default_value=params),
        Node(package="mercury_x1_driver", executable="base_node", name="mercury_base",
             parameters=[LaunchConfiguration("params")], output="screen", respawn=True,
             respawn_delay=2.0, condition=IfCondition(LaunchConfiguration("base"))),
        Node(package="mercury_x1_driver", executable="upper_body_node", name="mercury_upper_body",
             parameters=[LaunchConfiguration("params")], output="screen", respawn=True,
             respawn_delay=2.0, condition=IfCondition(LaunchConfiguration("upper_body"))),
    ])
