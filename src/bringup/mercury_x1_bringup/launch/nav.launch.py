"""Nav2 on top of RTAB-Map (/map + map->odom). No AMCL / map_server: RTAB-Map localizes.

  ros2 launch mercury_x1_bringup nav.launch.py [perception:=nvblox|cpu] [arm_footprint:=true|false] [composition:=true|false]

perception:=nvblox (default): Orbbec depth obstacles come from the nvblox ESDF slice (GPU, started by
slam.launch.py nvblox:=true). perception:=cpu: no nvblox layers; the decimated depth cloud
(/camera/cloud, point_cloud_xyz in robot.launch.py) goes into the local voxel layer instead.

arm_footprint:=true (default): the footprint follows the arms - when a link or the hand reaches
outside the base outline, the costmaps get a grown footprint (src/arm_footprint.cpp).

The Nav2 servers are listed here rather than through nav2_bringup/navigation_launch.py: Jazzy's version also
starts the route, collision monitor and docking servers, which need their own configuration (route graph,
dock plugins) and are not used on this robot.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory as share
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import LoadComposableNodes, Node
from launch_ros.descriptions import ComposableNode

# (lifecycle node name, package, executable, component plugin)
SERVERS = [
    ("controller_server", "nav2_controller", "controller_server", "nav2_controller::ControllerServer"),
    ("smoother_server", "nav2_smoother", "smoother_server", "nav2_smoother::SmootherServer"),
    ("planner_server", "nav2_planner", "planner_server", "nav2_planner::PlannerServer"),
    ("behavior_server", "nav2_behaviors", "behavior_server", "behavior_server::BehaviorServer"),
    ("velocity_smoother", "nav2_velocity_smoother", "velocity_smoother", "nav2_velocity_smoother::VelocitySmoother"),
    ("bt_navigator", "nav2_bt_navigator", "bt_navigator", "nav2_bt_navigator::BtNavigator"),
    ("waypoint_follower", "nav2_waypoint_follower", "waypoint_follower", "nav2_waypoint_follower::WaypointFollower"),
]
TF_REMAPS = [("/tf", "tf"), ("/tf_static", "tf_static")]
# controller + behaviors -> cmd_vel_nav -> velocity smoother -> /cmd_vel (nav2_bringup has the collision
# monitor between the smoother and /cmd_vel; not used here)
REMAPS = {
    "controller_server": TF_REMAPS + [("cmd_vel", "cmd_vel_nav")],
    "behavior_server": TF_REMAPS + [("cmd_vel", "cmd_vel_nav")],
    "velocity_smoother": TF_REMAPS + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
}


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
    names = [s[0] for s in SERVERS]
    nodes = []
    if LC("arm_footprint").perform(context) == "true":
        # grows the costmaps' footprint while an arm reaches out (src/arm_footprint.cpp)
        nodes.append(Node(package="mercury_x1_bringup", executable="arm_footprint", name="arm_footprint",
                          output="screen", parameters=[{"base_footprint": _base_footprint(params)}]))
    lifecycle = {"autostart": True, "node_names": names, "use_sim_time": False}
    if composition:
        # all Nav2 servers in one process; isolated = one executor thread per server (as nav2_bringup's container)
        nodes.append(Node(package="rclcpp_components", executable="component_container_isolated",
                          name="nav2_container", output="screen", parameters=[params],
                          remappings=TF_REMAPS))
        nodes.append(LoadComposableNodes(target_container="nav2_container", composable_node_descriptions=[
            ComposableNode(package=pkg, plugin=plugin, name=name, parameters=[params],
                           remappings=REMAPS.get(name, TF_REMAPS))
            for name, pkg, _exe, plugin in SERVERS
        ] + [ComposableNode(package="nav2_lifecycle_manager", plugin="nav2_lifecycle_manager::LifecycleManager",
                            name="lifecycle_manager_navigation", parameters=[lifecycle])]))
    else:
        for name, pkg, exe, _plugin in SERVERS:
            nodes.append(Node(package=pkg, executable=exe, name=name, output="screen", parameters=[params],
                              remappings=REMAPS.get(name, TF_REMAPS),
                              respawn=True, respawn_delay=2.0))
        nodes.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                          name="lifecycle_manager_navigation", output="screen", parameters=[lifecycle]))
    return nodes


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
