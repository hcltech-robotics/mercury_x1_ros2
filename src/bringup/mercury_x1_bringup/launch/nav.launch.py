"""Nav2 on top of RTAB-Map (/map + map->odom). No AMCL / map_server: RTAB-Map localizes.

  ros2 launch mercury_x1_bringup nav.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC


def generate_launch_description():
    params = os.path.join(share("mercury_x1_bringup"), "config", "nav2.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share("nav2_bringup"), "launch", "navigation_launch.py")),
            launch_arguments={"params_file": LC("params_file"), "autostart": "true",
                              "use_sim_time": "false", "use_composition": "False"}.items()),
    ])
