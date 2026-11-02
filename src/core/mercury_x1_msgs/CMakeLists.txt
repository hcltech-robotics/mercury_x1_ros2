cmake_minimum_required(VERSION 3.8)
project(mercury_x1_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArmState.msg"
  "msg/BaseState.msg"
  "msg/JointCommand.msg"
  "msg/CartesianCommand.msg"
  "msg/MeshTile.msg"
  "msg/VoiceWake.msg"
  "msg/VoiceBoardState.msg"
  "srv/VoiceBoardCommand.srv"
  "srv/SetWakeWord.srv"
  "srv/SetBeam.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
