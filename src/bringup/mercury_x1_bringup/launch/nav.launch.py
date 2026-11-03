"""Nav2 on top of RTAB-Map (/map + map->odom). No AMCL / map_server: RTAB-Map localizes.

  ros2 launch mercury_x1_bringup nav.launch.py [perception:=nvblox|cpu] [arm_footprint:=true|false] [composition:=true|false]

perception:=nvblox (default): Orbbec depth obstacles come from the nvblox ESDF slice (GPU, started by
slam.launch.py nvblox:=true). perception:=cpu: no nvblox layers; the decimated depth cloud
(/camera/cloud, point_cloud_xyz in robot.launch.py) goes into the local voxel layer instead.

arm_footprint:=true (default): the footprint follows the arms - when a link or the hand reaches
outside the base outline, the costmaps get a grown footprint (src/arm_footprint.cpp).
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node


def _params_for(path: str, perception: str) -> str:
    if perception == "nvblox":
        return path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for name in ("local_costmap", "global_costmap"):
        p = cfg[name][name]["ros__parameters"]
        p["plugins"] = [x for x in p["plugins"] if x != "nvblox_layer"]
        p.pop("nvblox_layer", None)
    cfg["local_costmap"]["local_costmap"]["ros__parameters"]["voxel_layer"]["observation_sources"] = "scan depth"
    fd, out = tempfile.mkstemp(prefix="mx1_nav2_cpu_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def _base_footprint(path: str) -> str:
    with open(path) as f:
        return yaml.safe_load(f)["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]


def _nav(context):
    params = _params_for(LC("params_file").perform(context), LC("perception").perform(context))
    composition = LC("composition").perform(context) == "true"
    nodes = []
    if composition:
        # all Nav2 servers in one process (nav2_bringup loads them into /nav2_container but does not start
        # it); isolated = one executor thread per server, as nav2_bringup's own container
        nodes.append(Node(package="rclcpp_components", executable="component_container_isolated",
                          name="nav2_container", output="screen",
                          parameters=[params, {"autostart": True}],
                          remappings=[("/tf", "tf"), ("/tf_static", "tf_static")]))
    if LC("arm_footprint").perform(context) == "true":
        # grows the costmaps' footprint while an arm reaches out (src/arm_footprint.cpp)
        nodes.append(Node(package="mercury_x1_bringup", executable="arm_footprint", name="arm_footprint",
                          output="screen", parameters=[{"base_footprint": _base_footprint(params)}]))
    return nodes + [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share("nav2_bringup"), "launch", "navigation_launch.py")),
        launch_arguments={"params_file": params, "autostart": "true",
                          "use_sim_time": "false", "use_composition": "True" if composition else "False",
                          "container_name": "nav2_container"}.items())]


def generate_launch_description():
    params = os.path.join(share("mercury_x1_bringup"), "config", "nav2.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=params),
        DeclareLaunchArgument("perception", default_value="nvblox", choices=["nvblox", "cpu"]),
        DeclareLaunchArgument("composition", default_value="true", choices=["true", "false"],
                              description="all Nav2 servers in one process (less CPU than ~10 processes)"),
        DeclareLaunchArgument("arm_footprint", default_value="true", choices=["true", "false"],
                              description="grow the footprint for arms reaching out"),
        OpaqueFunction(function=_nav),
    ])
