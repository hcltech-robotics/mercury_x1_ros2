"""cuVSLAM RGB-D visual odometry on the GPU from the Orbbec DaBai (colour + depth registered to colour).

  ros2 launch mercury_x1_perception cuvslam.launch.py [depth_time_offset_ms:=0.0]

Publishes /vslam/odom and /vslam/path in vslam_odom (statically aligned to odom at the first frame) for
comparison with the lidar odometry; it does not drive the TF tree.
Needs the cuVSLAM C++ SDK in /opt/cuvslam (see CMakeLists.txt).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("depth_time_offset_ms", default_value="0.0"),
        Node(package="mercury_x1_perception", executable="cuvslam_rgbd_node", name="cuvslam_rgbd", output="screen",
             parameters=[{"odom_frame": "vslam_odom", "base_frame": "base_footprint", "align_frame": "odom",
                          "depth_time_offset_ms": ParameterValue(LC("depth_time_offset_ms"), value_type=float),
                          "max_pair_dt_ms": 25.0}],
             remappings=[("color", "/camera/color/image_raw"), ("depth", "/camera/depth/image_raw"),
                         ("camera_info", "/camera/color/camera_info")]),
    ])
