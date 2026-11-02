"""MoveIt move_group for the Mercury X1 arms (and waist): planning, IK, self-collision (arm vs arm, arm vs body).

Joint states and TF come from the running stack (upper-body driver + robot_state_publisher); trajectories go to
the driver's FollowJointTrajectory servers (config/moveit_controllers.yaml).
  ros2 launch mercury_x1_moveit_config move_group.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def moveit_config():
    xacro = os.path.join(get_package_share_directory("mercury_x1_description"), "urdf", "mercury_x1.urdf.xacro")
    return (MoveItConfigsBuilder("mercury_x1", package_name="mercury_x1_moveit_config")
            .robot_description(file_path=xacro)
            .robot_description_semantic(file_path="config/mercury_x1.srdf")
            .robot_description_kinematics(file_path="config/kinematics.yaml")
            .joint_limits(file_path="config/joint_limits.yaml")
            .trajectory_execution(file_path="config/moveit_controllers.yaml")
            .planning_pipelines(pipelines=["ompl"])
            .planning_scene_monitor(publish_robot_description=True, publish_robot_description_semantic=True)
            .to_moveit_configs())


def generate_launch_description():
    return generate_move_group_launch(moveit_config())
