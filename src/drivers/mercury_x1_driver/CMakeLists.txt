cmake_minimum_required(VERSION 3.8)
project(mercury_x1_driver)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(mercury_x1_msgs REQUIRED)

add_library(mercury_x1_protocol STATIC src/serial_port.cpp src/arm_protocol.cpp src/base_protocol.cpp)
target_include_directories(mercury_x1_protocol PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
set_target_properties(mercury_x1_protocol PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(base_node src/base_node.cpp)
target_link_libraries(base_node mercury_x1_protocol)
ament_target_dependencies(base_node rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs tf2_ros mercury_x1_msgs)

add_executable(upper_body_node src/upper_body_node.cpp)
target_link_libraries(upper_body_node mercury_x1_protocol)
ament_target_dependencies(upper_body_node rclcpp sensor_msgs std_srvs mercury_x1_msgs)

install(TARGETS base_node upper_body_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_protocol test/test_protocol.cpp)
  target_link_libraries(test_protocol mercury_x1_protocol)
endif()

ament_package()
