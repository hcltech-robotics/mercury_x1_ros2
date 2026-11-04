"""cuVSLAM RGB-D visual odometry on the GPU (Isaac ROS visual_slam, tracking_mode 2) from the Orbbec DaBai
(colour + depth registered to colour).

  ros2 launch mercury_x1_perception cuvslam.launch.py [slam:=false]

Publishes /vslam/odom and /vslam/path in vslam_odom (= odom at start-up, static TF) for comparison with the
lidar odometry; it does not drive the TF tree. slam:=true also runs cuVSLAM's loop closure / localization
(more GPU + RAM, /vslam/slam_path).
Isaac ROS 4.6 (JetPack 7.2, Jazzy): `sudo apt install ros-jazzy-isaac-ros-visual-slam`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vslam = ComposableNode(
        name="cuvslam_rgbd", package="isaac_ros_visual_slam",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        parameters=[{
            "tracking_mode": 2,                      # RGB-D
            "num_cameras": 1, "min_num_images": 1, "depth_camera_id": 0,
            "depth_scale_factor": 1000.0,            # uint16 mm
            "camera_optical_frames": ["camera_color_optical_frame"],
            "base_frame": "base_footprint", "odom_frame": "vslam_odom", "map_frame": "vslam_map",
            "publish_odom_to_base_tf": False, "publish_map_to_odom_tf": False,
            "rectified_images": False,               # the DaBai colour stream is raw; camera_info carries D
            "enable_image_denoising": False,
            # the DaBai has no colour/depth hardware sync: colour_only_with_depth (orbbec_camera) pairs the
            # closest colour frame with every depth frame (15 Hz), their stamps differ by up to ~30 ms
            "sync_matching_threshold_ms": 40.0,
            "image_jitter_threshold_ms": 40.0,
            "enable_ground_constraint_in_odometry": True,   # planar robot
            "enable_ground_constraint_in_slam": True,
            "enable_localization_n_mapping": ParameterValue(LC("slam"), value_type=bool),
            "enable_slam_visualization": False, "enable_landmarks_view": False, "enable_observations_view": False,
        }],
        remappings=[
            ("visual_slam/image_0", "/camera/color/image_raw"),
            ("visual_slam/camera_info_0", "/camera/color/camera_info"),
            ("visual_slam/depth_0", "/camera/depth/image_raw"),
            ("visual_slam/tracking/odometry", "/vslam/odom"),
            ("visual_slam/tracking/vo_path", "/vslam/path"),
            ("visual_slam/tracking/slam_path", "/vslam/slam_path"),
            ("visual_slam/tracking/vo_pose", "/vslam/pose"),
            ("visual_slam/status", "/vslam/status"),
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument("slam", default_value="false",
                              description="also loop closure / localization (cuVSLAM SLAM), not just odometry"),
        ComposableNodeContainer(name="vslam_container", namespace="", package="rclcpp_components",
                                executable="component_container", composable_node_descriptions=[vslam],
                                output="screen", respawn=True, respawn_delay=3.0),
        # vslam_odom starts where odom is at launch: the stack boots with the base at the odom origin
        Node(package="tf2_ros", executable="static_transform_publisher", name="vslam_odom_align",
             arguments=["--frame-id", "odom", "--child-frame-id", "vslam_odom"], output="log"),
    ])
