"""GPU 3D reconstruction (Isaac ROS nvblox) from the Orbbec DaBai + mesh tiles for Rerun.

Needs the odom TF (RTAB-Map icp_odometry), so it is started from slam.launch.py (nvblox:=true).

  ros2 launch mercury_x1_perception nvblox.launch.py

Outputs: /nvblox_node/static_map_slice (Nav2 NvbloxCostmapLayer), /nvblox_node/mesh,
/nvblox/mesh_tiles (mercury_x1_msgs/MeshTile, for the Rerun bridge).
"""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(share("mercury_x1_perception"), "config", "nvblox.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        Node(package="nvblox_ros", executable="nvblox_node", name="nvblox_node", output="screen",
             parameters=[LC("params_file")],
             remappings=[("camera_0/depth/image", "/camera/depth/image_raw"),
                         ("camera_0/depth/camera_info", "/camera/depth/camera_info"),
                         ("camera_0/color/image", "/camera/color/image_raw"),
                         ("camera_0/color/camera_info", "/camera/color/camera_info")],
             respawn=True, respawn_delay=3.0),
        Node(package="mercury_x1_perception", executable="nvblox_mesh_tiler", name="nvblox_mesh_tiler",
             parameters=[{"tile_size": 2.0, "publish_period": 1.0}],
             remappings=[("mesh", "/nvblox_node/mesh"), ("mesh_tiles", "/nvblox/mesh_tiles")],
             output="log", respawn=True, respawn_delay=3.0),
    ])
