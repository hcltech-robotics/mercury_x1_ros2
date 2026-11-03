# State of the wheeled base, published by mercury_x1_driver/base.
std_msgs/Header header
bool connected                   # frames received from the base MCU within the last 0.5 s
string firmware_version          # e.g. "v1.2"
uint8 stop_flag                  # raw stop/e-stop flag byte from the MCU
float32 voltage                  # battery/supply voltage [V]
uint8 charge_status              # sensor_msgs/BatteryState POWER_SUPPLY_STATUS_*, ESTIMATED from the voltage
                                 # trend (the MCU reports no charging state): 0 unknown, 1 charging,
                                 # 2 discharging, 4 full
uint16[] mcu_aux                 # undecoded words u6, u7 of the ultrasonic frame (4, 0 so far)
float32[] ultrasonic_range       # every ultrasonic channel [m]; >= max range means no echo / not fitted
string[] ultrasonic_names        # mounting position of each channel above ("" = not fitted)
geometry_msgs/Twist measured     # wheel-odometry twist reported by the MCU (scaled)
geometry_msgs/Twist commanded    # what the driver is currently sending (after limits/watchdog)
bool cmd_timeout                 # true when no fresh cmd_vel -> base held at zero
bool obstacle_hold               # true when forward motion is being blocked by the ultrasonic guard
