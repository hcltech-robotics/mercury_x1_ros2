"""Rerun bridge + control page.  Open http://<robot>:8080  (plain viewer: :9090)."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    previews = [
        Node(package="mercury_x1_driver", executable="image_preview_node", name=f"preview_{name}",
             parameters=[{"rate": 4.0, "jpeg_quality": 70, "max_width": 640}],
             remappings=[("image", topic), ("image/preview", f"/preview/{name}")], respawn=True)
        for name, topic in (("orbbec", "/camera/color/image_raw"), ("bellybutton", "/bellybutton/image_raw"))
    ]
    return LaunchDescription(previews + [
        Node(package="mercury_x1_web", executable="rerun_bridge", name="rerun_bridge", output="screen",
             respawn=True, respawn_delay=3.0),
        Node(package="mercury_x1_web", executable="web_server", name="mercury_web", output="screen",
             respawn=True, respawn_delay=3.0),
    ])
