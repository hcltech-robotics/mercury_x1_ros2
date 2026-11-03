from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    host_arg = DeclareLaunchArgument('host', default_value='0.0.0.0')
    port_arg = DeclareLaunchArgument('port', default_value='9000')
    debug_arg = DeclareLaunchArgument('debug', default_value='false')
    
    return LaunchDescription([
        host_arg,
        port_arg,
        debug_arg,
        Node(
            package='mercury_x1_control',
            executable='arm_control_server',
            output='screen',
            parameters=[{
                'host': LaunchConfiguration('host'),
                'port': LaunchConfiguration('port'),
                'debug': LaunchConfiguration('debug')
            }]
        )
    ])