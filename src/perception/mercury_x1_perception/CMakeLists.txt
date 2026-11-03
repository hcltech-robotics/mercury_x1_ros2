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

# cuVSLAM RGB-D visual odometry: NVIDIA's C++ release for Orin (cuvslam-cpp-<ver>-orin-cuda12.6.3-ubuntu22.04),
# unpacked to /opt/cuvslam/{include,lib}. Skipped where the library is not installed.
set(CUVSLAM_ROOT "/opt/cuvslam" CACHE PATH "cuVSLAM C++ SDK root")
find_library(CUVSLAM_LIB cuvslam PATHS ${CUVSLAM_ROOT}/lib NO_DEFAULT_PATH)
if(CUVSLAM_LIB)
  find_package(Eigen3 REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(tf2_ros REQUIRED)
  find_package(tf2_eigen REQUIRED)
  add_executable(cuvslam_rgbd_node src/cuvslam_rgbd_node.cpp)
  target_include_directories(cuvslam_rgbd_node PRIVATE ${CUVSLAM_ROOT}/include ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(cuvslam_rgbd_node ${CUVSLAM_LIB})
  set_target_properties(cuvslam_rgbd_node PROPERTIES INSTALL_RPATH "${CUVSLAM_ROOT}/lib")
  ament_target_dependencies(cuvslam_rgbd_node rclcpp geometry_msgs nav_msgs sensor_msgs tf2_ros tf2_eigen)
  install(TARGETS cuvslam_rgbd_node DESTINATION lib/${PROJECT_NAME})
else()
  message(STATUS "cuVSLAM not found in ${CUVSLAM_ROOT}: skipping cuvslam_rgbd_node")
endif()
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
