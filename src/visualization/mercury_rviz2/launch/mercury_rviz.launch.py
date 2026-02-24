import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    package_dir = get_package_share_directory('mercury_rviz2')
    
    # Declare the launch argument
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(package_dir, 'rviz', 'mercury_x1.rviz'),
        description='Full path to the RVIZ config file to use'
    )
    
    # Use LaunchConfiguration to reference the argument value
    rviz_config = LaunchConfiguration('rviz_config')
    
    joint_state_publisher_node = Node(
        package='joint_state_publisher_gui', 
        executable='joint_state_publisher_gui',
        remappings=[('joint_states', 'joint_commands')]
    )

    rviz_start = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config] 
    )
    
    ld = LaunchDescription()
    ld.add_action(rviz_config_arg)    
    ld.add_action(joint_state_publisher_node)
    ld.add_action(rviz_start)
    
    return ld