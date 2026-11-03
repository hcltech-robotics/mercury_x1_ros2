import os
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Orbbec depth Camera
    orbbec_dir = get_package_share_directory('orbbec_camera')
    orbbec_launch_dir = os.path.join(orbbec_dir, 'launch')

    Dabai = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(orbbec_launch_dir, 'dabai.launch.py')),
        launch_arguments={'camera_name': 'orbbec_camera'}.items()
    )

    # Bellybutton USB Camera
    video_device = LaunchConfiguration('video_device')

    declare_video_device = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/RgbCam',
        description='Path to video device'
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device': video_device,
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'raw_mjpeg',
            'frame_id': 'bellybutton_cam',
            'camera_name': 'bellybutton_cam',
            'io_method': 'mmap'
        }],
        remappings=[
            ('camera_info', '/bellybutton_cam/camera_info'),
			('image_raw', '/bellybutton_cam/image_raw')
    	]
    )

    # Create the launch description and populate
    ld = LaunchDescription()
    
    # Add Orbbec camera
    ld.add_action(Dabai)
    
    # Add USB camera arguments and node
    ld.add_action(declare_video_device)
    ld.add_action(usb_cam_node)

    return ld