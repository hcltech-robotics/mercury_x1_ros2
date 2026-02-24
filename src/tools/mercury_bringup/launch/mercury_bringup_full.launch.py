import os
from pathlib import Path
import launch
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import PushRosNamespace
import launch_ros.actions
from launch.conditions import UnlessCondition
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('mercury_bringup')
    launch_dir = os.path.join(bringup_dir, 'launch')
    ekf_config = Path(bringup_dir, 'config', 'ekf.yaml')

    
    carto_slam = LaunchConfiguration('carto_slam', default='false')
    carto_slam_dec = DeclareLaunchArgument('carto_slam',default_value='false')
            
    mercury_robot = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'base_serial.launch.py')),
            launch_arguments={'akmcar': 'false'}.items(),
    )
    #choose your car,the default car is mini_mec 
    choose_car = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'mercury_robot_mode.launch.py')),
    )

    base_to_imu = launch_ros.actions.Node(
            package='tf2_ros', 
            executable='static_transform_publisher', 
            name='base_to_imu',
            arguments=['0','0','-0.15','0','0','0','base_link','imu_link'],
    )

    robot_ekf = launch_ros.actions.Node(
            condition=UnlessCondition(carto_slam),
            package='robot_localization', 
            executable='ekf_node', 
            parameters=[ekf_config],
            remappings=[("odometry/filtered", "odom_combined")]
            )

    # GUI for joint state publisher         
    gui_launch_arg = DeclareLaunchArgument(
        name="gui",
        default_value="false"
    )
                              
    joint_state_publisher_node = launch_ros.actions.Node(
            package='joint_state_publisher_gui', 
            executable='joint_state_publisher_gui', 
            remappings=[('joint_states', 'joint_commands')],
            condition=IfCondition(LaunchConfiguration('gui'))
    )

    # joystick node
    joystick_dir = get_package_share_directory('mercury_joy')
    joystick_launch_dir = os.path.join(joystick_dir, 'launch')

    joystick_mercury_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(joystick_launch_dir, 'only_joy.launch.py')),
        launch_arguments={'axis_linear': '1', 'axis_angular': '3'}.items()
    ) 
    joy_launch_arg = DeclareLaunchArgument(
        name="use_joy",
        default_value="false"
    )
    joy_node = launch_ros.actions.Node(
        package='joy', 
        executable='joy_node', 
        condition=IfCondition(LaunchConfiguration('use_joy'))
    )

    # laser node
    laser_scan = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'mercury_lidar.launch.py')),
    )

    # upper body


    ld = LaunchDescription()
    ld.add_action(gui_launch_arg)
    ld.add_action(carto_slam_dec)
    ld.add_action(mercury_robot)
    ld.add_action(base_to_imu)
    ld.add_action(joint_state_publisher_node)
    ld.add_action(choose_car)
    ld.add_action(robot_ekf)
    ld.add_action(joystick_mercury_robot)
    ld.add_action(joy_launch_arg)
    ld.add_action(joy_node)
    ld.add_action(laser_scan)



    return ld

