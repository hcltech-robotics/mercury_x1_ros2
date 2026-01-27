# mercury_x1_ros2
Use of Mercury X1 mobile chassis ROS2

## Support  
- ROS 2 Humble ✅

## To-Do  
- [x] URDF model  
- [x] Real-time joint state synchronization  
- [x] Real-time IMU data synchronization  
- [x] Laser scan integration (2D Lidar)
- [x] Real-time joystick control  
- [ ] RGB Camera stream integration 
- [ ] RGBD (Deopth) Camera stream integration  
- [ ] Foxglove WebSocket bridge  
- [ ] SLAM (using `slam_toolbox`)  
- [ ] Navigation (using `nav2`)  


## Package Download and Install

```bash
source /opt/ros/humble/setup.bash 

git clone https://github.com/elephantrobotics/mercury_x1_ros2.git
cd ~/mercury_x1_ros2
rosdep install --from-paths src --ignore-src -y --rosdistro humble
colcon build    # Compile all function packages
ource install/setup.bash

# Compile the function package separately:
If you only compile “turn_on_mercury_robot“, you need to execute the command:
colcon build --packages-select turn_on_mercury_robot
```
# 一、Mercury Seven Axis

>> **Note**:  Make sure the machine is powered on and enabled before use

## Bottom car keyboard control

**1. First start the car's underlying communication program and map building:**

```bash
$ ros2 launch slam_gmapping slam_gmapping.launch.py
```

**2. Then start the keyboard control program:**

```bash
$ ros2 run mercury_x1_control mercury_keyboard
```

## Chassis car + upper body robotic arm control

**1. First start the car's underlying communication program and map building:**

```bash
$ ros2 launch slam_gmapping slam_gmapping.launch.py
```

**2. Then Turn on upper body robotic arm control:**

```bash
$ ros2 run mercury_x1_control slider_control
```

**3. Finally start the keyboard control program:**

```bash
$ ros2 run mercury_x1_control mercury_keyboard
```

# 二、Mercury Turing Six Axis

>> **Note**:  Make sure the machine is powered on and enabled before use

## Bottom car keyboard control

**1. First start the car's underlying communication program and map building:**

```bash
$ ros2 launch slam_gmapping slam_gmapping_turing.launch.py
```

**2. Then start the keyboard control program:**

```bash
$ ros2 run mercury_x1_control mercury_keyboard
```

## Chassis car + upper body robotic arm control

**1. First start the car's underlying communication program and map building:**

```bash
$ ros2 launch slam_gmapping slam_gmapping_turing.launch.py
```

**2. Then Turn on upper body robotic arm control:**

```bash
$ ros2 run mercury_x1_control slider_control_turing
```

**3. Finally start the keyboard control program:**

```bash
$ ros2 run mercury_x1_control mercury_keyboard
```

# Common ROS 2 Commands Summary:

```bash
# Build a single package:
# Example: build only turn_on_mercury_robot
colcon build --packages-select turn_on_mercury_robot

# Build all packages:
colcon build

# Note: After modifying launch files, you must recompile for changes to take effect.

# NFS Mount
sudo mount -t nfs 192.168.0.100:/home/er/mercury_x1_ros2/ /mnt


# 1. Launch robot base
# Mercury Seven Axis
ros2 launch turn_on_mercury_robot turn_on_mercury_robot.launch.py
# Mercury Six Axis
ros2 launch turn_on_mercury_robot turn_on_mercury_robot_turing.launch.py


# 2. Launch base controller
# Mercury Seven Axis
ros2 launch turn_on_mercury_robot turn_on_mercury_robot.launch.py
# Mercury Six Axis
ros2 launch turn_on_mercury_robot turn_on_mercury_robot_turing.launch.py


# 3. Launch camera
ros2 launch turn_on_mercury_robot mercury_camera.launch.py


# 4. Launch LiDAR
ros2 launch turn_on_mercury_robot mercury_lidar.launch.py


# 5. Keyboard control for mobile base
ros2 run mercury_x1_control mercury_keyboard


# 6. Simple following capabilities
# ① LiDAR-based following
ros2 launch simple_follower_ros2 laser_follower.launch.py

# ② Vision-based line following
ros2 launch simple_follower_ros2 line_follower.launch.py

# ③ Vision-based object tracking
ros2 launch simple_follower_ros2 visual_follower.launch.py


# 7. 2D SLAM (Mapping)
# ① Mapping with gmapping
# Mercury Seven Axis
ros2 launch slam_gmapping slam_gmapping.launch.py
# Mercury Six Axis
ros2 launch slam_gmapping slam_gmapping_turing.launch.py

# ② Mapping with slam_toolbox
ros2 launch mercury_slam_toolbox online_async.launch.py

# ③ Mapping with Cartographer
ros2 launch mercury_cartographer cartographer.launch.py

# Save map
ros2 launch mercury_nav2 save_map.launch.py


# 8. 2D Navigation
ros2 launch mercury_nav2 mercury_nav2.launch.py


# 9. RRT-based autonomous exploration & mapping  
# [Publish four waypoints clockwise/counterclockwise; the last point must be in the known map area]
# Step 1:
ros2 launch mercury_slam_toolbox online_sync.launch.py
# Step 2:
ros2 launch mercury_robot_rrt mercury_rrt_slam.launch.py


# 10. RTAB-Map SLAM
ros2 launch mercury_robot_rtab mercury_slam_rtab.launch.py


# 11. RTAB-Map Navigation
ros2 launch mercury_robot_rtab mercury_nav2_rtab.launch.py


# 12. USB Joystick Control
ros2 launch mercury_joy mercury_joy.launch.py


# 13. Launch RViz2
# Note: Use a virtual machine to run RViz2


# 14. Full robot control (mobile base + upper-body manipulator)
# ① Mapping with gmapping
# Mercury Seven Axis
ros2 launch slam_gmapping slam_gmapping.launch.py
# Mercury Six Axis
ros2 launch slam_gmapping slam_gmapping_turing.launch.py

# ② Launch upper-body manipulator control
# Mercury Seven Axis
ros2 run mercury_x1_control slider_control
# Mercury Six Axis
ros2 run mercury_x1_control slider_control_turing

# ③ Keyboard control for mobile base
ros2 run mercury_x1_control mercury_keyboard
```