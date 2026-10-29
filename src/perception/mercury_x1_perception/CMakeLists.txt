cmake_minimum_required(VERSION 3.8)
project(mercury_x1_perception)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mercury_x1_msgs REQUIRED)
find_package(nvblox_msgs REQUIRED)

add_executable(nvblox_mesh_tiler src/nvblox_mesh_tiler.cpp)
ament_target_dependencies(nvblox_mesh_tiler rclcpp mercury_x1_msgs)
# not ament_target_dependencies(nvblox_msgs): the apt package's exported python target carries a
# numpy include path from NVIDIA's build machine (/usr/local/lib/python3.10/...) -> CMake error
target_link_libraries(nvblox_mesh_tiler
  nvblox_msgs::nvblox_msgs__rosidl_generator_cpp
  nvblox_msgs::nvblox_msgs__rosidl_typesupport_cpp)

install(TARGETS nvblox_mesh_tiler DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
