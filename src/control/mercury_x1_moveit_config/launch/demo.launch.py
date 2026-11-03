"""Off-robot MoveIt demo: robot_state_publisher + mock arm driver + move_group (+ RViz with rviz:=true).

  ros2 launch mercury_x1_moveit_config demo.launch.py [rviz:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    here = get_package_share_directory("mercury_x1_moveit_config")
    desc = get_package_share_directory("mercury_x1_description")
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="false"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(desc, "launch", "description.launch.py"))),
        Node(package="mercury_x1_moveit_config", executable="mock_arm_driver.py", output="screen"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(here, "launch", "move_group.launch.py"))),
        Node(package="rviz2", executable="rviz2", output="log", condition=IfCondition(LaunchConfiguration("rviz")),
             arguments=["-d", os.path.join(here, "config", "moveit.rviz")] if os.path.exists(
                 os.path.join(here, "config", "moveit.rviz")) else []),
    ])
