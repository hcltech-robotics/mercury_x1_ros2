"""Mercury X1 hardware bringup: drivers + description + lidar + cameras.

  ros2 launch mercury_x1_bringup robot.launch.py [cameras:=true] [upper_body:=true] [depth_cloud:=false]
                                                 [voice_board:=true]

depth_cloud:=true adds the CPU depth cloud (/camera/cloud) needed by nav.launch.py perception:=cpu;
by default depth obstacles go through nvblox on the GPU (slam.launch.py nvblox:=true).
"""
import os

from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def generate_launch_description():
    drv_params = os.path.join(share("mercury_x1_driver"), "config", "drivers.yaml")
    robot_params = os.path.join(share("mercury_x1_bringup"), "config", "robot.yaml")
    lidar_params = os.path.join(share("lslidar_driver"), "params", "lsm10.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("cameras", default_value="true"),
        DeclareLaunchArgument("upper_body", default_value="true"),
        DeclareLaunchArgument("depth_cloud", default_value="false", description="CPU depth cloud /camera/cloud"),
        DeclareLaunchArgument("voice_board", default_value="true",
                              description="head voice board driver (wake word + direction, /voice_board/*)"),
        DeclareLaunchArgument("remote", default_value="true",
                              description="Unitree BLE remote over the built-in Bluetooth (connect from the web page)"),

        Node(package="mercury_x1_driver", executable="base_node", name="mercury_base",
             parameters=[drv_params, robot_params], remappings=[("odom", "wheel/odom")],
             output="screen", respawn=True, respawn_delay=2.0),
        Node(package="mercury_x1_driver", executable="upper_body_node", name="mercury_upper_body",
             parameters=[drv_params, robot_params], output="screen", respawn=True, respawn_delay=2.0,
             condition=IfCondition(LC("upper_body"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share("mercury_x1_description"), "launch", "description.launch.py"))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share("mercury_x1_voice_board"), "launch", "voice_board.launch.py")),
            condition=IfCondition(LC("voice_board"))),

        # LSLiDAR M10: native 100..260 deg is masked (it sees the robot body there)
        Node(package="lslidar_driver", executable="lslidar_driver_node", name="lslidar_driver_node",
             parameters=[lidar_params], output="screen", respawn=True, respawn_delay=2.0),

        # Orbbec DaBai on the jaw: HW depth->colour registration, depth 15 Hz (USB 2.0 link)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share("orbbec_camera"), "launch", "dabai.launch.py")),
            launch_arguments={
                "camera_name": "camera", "depth_registration": "true", "color_fps": "30",
                "depth_fps": "15", "enable_ir": "false", "enable_point_cloud": "false",
                # the DaBai streams colour only at 30 Hz (and has no frame sync): publish one colour frame per
                # depth frame, the closest in time -> colour 15 Hz; halves the decode + copies to every subscriber
                "color_only_with_depth": "true",
                "publish_tf": "true",
            }.items(),
            condition=IfCondition(LC("cameras"))),
        # light CPU obstacle cloud for costmaps / Rerun (only without nvblox)
        Node(package="rtabmap_util", executable="point_cloud_xyz", name="depth_cloud",
             parameters=[{"decimation": 4, "voxel_size": 0.04, "max_depth": 3.5, "approx_sync": True}],
             remappings=[("depth/image", "/camera/depth/image_raw"),
                         ("depth/camera_info", "/camera/depth/camera_info"),
                         ("cloud", "/camera/cloud")],
             condition=IfCondition(PythonExpression(
                 ["'", LC("cameras"), "' == 'true' and '", LC("depth_cloud"), "' == 'true'"]))),

        # Unitree BLE handheld remote -> /joy, /cmd_vel (hold L1); scan / connect / autoconnect from the web page
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share("mercury_x1_teleop"), "launch", "remote.launch.py")),
            condition=IfCondition(LC("remote"))),

        # bellybutton RGB camera
        Node(package="usb_cam", executable="usb_cam_node_exe", name="bellybutton_camera",
             parameters=[{"video_device": "/dev/RgbCam", "image_width": 640, "image_height": 480,
                          "pixel_format": "mjpeg2rgb", "framerate": 15.0,
                          "frame_id": "bellybutton_camera_optical_frame",
                          "camera_name": "bellybutton_camera"}],
             remappings=[("image_raw", "/bellybutton/image_raw"),
                         ("image_raw/compressed", "/bellybutton/image_raw/compressed"),
                         ("camera_info", "/bellybutton/camera_info")],
             output="log", respawn=True, respawn_delay=2.0,
             condition=IfCondition(LC("cameras"))),
    ])
