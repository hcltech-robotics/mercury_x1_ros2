"""robot_state_publisher for the Mercury X1 (joint states come from mercury_x1_driver)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    xacro_file = os.path.join(get_package_share_directory("mercury_x1_description"), "urdf", "mercury_x1.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": robot_description, "publish_frequency": 30.0}]),
    ])
