"""RTAB-Map SLAM / localization fusing the 2D lidar and the Orbbec RGB-D camera.

  ros2 launch mercury_x1_bringup slam.launch.py mode:=mapping      [map:=office] [fresh:=true]
  ros2 launch mercury_x1_bringup slam.launch.py mode:=localization  map:=office

Fusion:
  * odometry: icp_odometry scan-matches the lidar, seeded by the base driver's wheel+gyro
    odometry (guess_frame_id) -> TF odom -> odom_wheel.
  * graph: lidar ICP for registration and neighbour refinement, RGB visual words for
    loop closure and relocalization.
  * 2D grid (/map): lidar + depth camera (Grid/Sensor=2) whenever the camera is fused, so the saved
    map also holds what the lidar plane misses (tables, overhangs; 0.06..1.4 m). Built per map node from
    a 4x decimated depth image, i.e. cheap. Mapping levels the head first (web_server: neck 0, camera
    horizontal), so the depth sees the room rather than the floor. nvblox additionally feeds live 3D
    obstacles to Nav2.
  * nvblox (mercury_x1_perception): CUDA TSDF/ESDF from the Orbbec depth+colour, posed by the
    odom TF published here; gives Nav2 its 3D-aware costmap slice and Rerun a mesh.
  * cuVSLAM (cuvslam:=true): GPU RGB-D visual odometry on /vslam/odom, aligned to odom for comparison
    with the lidar odometry (not fused yet).
Maps live in ~/maps/<map>.db.
"""
import os

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory as share
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _nodes(context):
    mode = LC("mode").perform(context)
    name = LC("map").perform(context)
    fresh = LC("fresh").perform(context).lower() == "true"
    use_camera = LC("camera").perform(context).lower() == "true"
    use_nvblox = use_camera and LC("nvblox").perform(context).lower() == "true"
    use_cuvslam = use_camera and LC("cuvslam").perform(context).lower() == "true"
    maps_dir = os.path.expanduser("~/maps")
    os.makedirs(maps_dir, exist_ok=True)
    db = os.path.join(maps_dir, f"{name}.db")
    localization = mode == "localization"

    icp_common = {
        "Reg/Force3DoF": "true",
        "Icp/VoxelSize": "0.05",
        "Icp/RangeMin": "0.15",
        "Icp/MaxCorrespondenceDistance": "0.15",
        "Icp/PointToPlane": "false",
        "Icp/Iterations": "30",
        "Icp/Epsilon": "0.001",
        "Icp/CorrespondenceRatio": "0.2",
    }

    odom = Node(
        package="rtabmap_odom", executable="icp_odometry", name="icp_odometry", output="screen",
        parameters=[{
            "frame_id": "base_footprint", "odom_frame_id": "odom", "guess_frame_id": "odom_wheel",
            "publish_tf": True, "wait_for_transform": 0.2, "topic_queue_size": 10,
            "Odom/ScanKeyFrameThr": "0.6", "OdomF2M/ScanSubtractRadius": "0.05",
            "OdomF2M/ScanMaxSize": "3000", "Odom/ResetCountdown": "1",
            **icp_common,
        }],
        remappings=[("scan", "/scan"), ("odom", "/odom")],
    )

    sync = Node(
        package="rtabmap_sync", executable="rgbd_sync", name="rgbd_sync", output="screen",
        parameters=[{"approx_sync": True, "approx_sync_max_interval": 0.04, "topic_queue_size": 10}],
        remappings=[("rgb/image", "/camera/color/image_raw"),
                    ("depth/image", "/camera/depth/image_raw"),
                    ("rgb/camera_info", "/camera/color/camera_info"),
                    ("rgbd_image", "/rgbd_image")],
    )

    rtab_params = {
        "frame_id": "base_footprint", "odom_frame_id": "odom", "map_frame_id": "map",
        "subscribe_depth": False, "subscribe_rgb": False, "subscribe_rgbd": use_camera,
        "subscribe_scan": True, "approx_sync": True, "sync_queue_size": 20, "topic_queue_size": 10,
        "wait_for_transform": 0.3, "database_path": db, "publish_tf": True, "map_always_update": False,
        "Rtabmap/DetectionRate": "1.5",
        "RGBD/LinearUpdate": "0.08", "RGBD/AngularUpdate": "0.08",
        "RGBD/NeighborLinkRefining": "true", "RGBD/ProximityBySpace": "true",
        "RGBD/ProximityMaxGraphDepth": "0", "RGBD/ProximityPathMaxNeighbors": "10",
        "RGBD/OptimizeMaxError": "3.0",
        "Reg/Strategy": "1",              # ICP (lidar) for registration; visual words for loop detection
        "Mem/NotLinkedNodesKept": "false", "Mem/STMSize": "30",
        "Kp/MaxFeatures": "400", "Vis/MinInliers": "15", "Vis/MaxDepth": "4.0",
        "Optimizer/Strategy": "2",        # g2o
        # occupancy grid: lidar + depth camera
        "Grid/Sensor": "2" if use_camera else "0",
        "Grid/CellSize": "0.05", "Grid/RangeMax": "5.0", "Grid/RayTracing": "true",
        # 2D local grids: a database reprocessed with Grid/3D=true gave a 2D /map of a single scan fan
        # (2026-10-30), so the 3D map comes from the stored RGB-D frames instead (web_server / slam3d)
        "Grid/3D": "false", "Grid/NormalsSegmentation": "false",
        "Grid/MaxGroundHeight": "0.06", "Grid/MaxObstacleHeight": "1.4",
        "Grid/DepthDecimation": "4", "Grid/ClusterRadius": "0.1", "Grid/MinClusterSize": "8",
        "GridGlobal/MinSize": "20",
        **icp_common,
    }
    if localization:
        rtab_params.update({"Mem/IncrementalMemory": "false", "Mem/InitWMWithAllNodes": "true",
                            "RGBD/StartAtOrigin": "false"})
    args = ["-d"] if (fresh and not localization) else []
    slam = Node(
        package="rtabmap_slam", executable="rtabmap", name="rtabmap", output="screen",
        parameters=[rtab_params], arguments=args,
        remappings=[("scan", "/scan"), ("odom", "/odom"), ("rgbd_image", "/rgbd_image")],
    )
    nodes = [odom, slam]
    if use_camera:
        nodes.insert(1, sync)
    if use_nvblox:
        nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share("mercury_x1_perception"), "launch", "nvblox.launch.py"))))
    if use_cuvslam:
        nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share("mercury_x1_perception"), "launch", "cuvslam.launch.py"))))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="mapping", choices=["mapping", "localization"]),
        DeclareLaunchArgument("map", default_value="default", description="map name -> ~/maps/<map>.db"),
        DeclareLaunchArgument("fresh", default_value="false", description="mapping: delete the db first"),
        DeclareLaunchArgument("camera", default_value="true", description="fuse the RGB-D camera"),
        DeclareLaunchArgument("nvblox", default_value="true", description="GPU 3D mapping (Isaac ROS nvblox)"),
        DeclareLaunchArgument("cuvslam", default_value="true",
                              description="cuVSLAM RGB-D visual odometry on the GPU (/vslam/odom, comparison only)"),
        OpaqueFunction(function=_nodes),
    ])
