"""Unitree BLE remote over the built-in Bluetooth. Connect / autoconnect from the web page (Robot tab).

  ros2 launch mercury_x1_teleop remote.launch.py [deadman:=L1]   (deadman:='' -> sticks drive directly)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("deadman", default_value="L1"),
        Node(package="mercury_x1_teleop", executable="unitree_remote", name="unitree_remote", output="screen",
             parameters=[{"adapter": "hci0", "deadman_button": ParameterValue(LC("deadman"), value_type=str), "max_linear": 0.35,
                          "max_linear_reverse": 0.15, "max_angular": 0.8}],
             respawn=True, respawn_delay=3.0),
    ])
