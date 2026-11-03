# State of one 7-DoF arm, published by mercury_x1_driver/upper_body.
std_msgs/Header header
string arm                       # "left" | "right"
bool connected                   # serial link to the arm MCU answers
bool powered                     # servos enabled (power_on)
bool moving                      # firmware reports a motion in progress
bool freedrive                   # arm released for moving by hand (motion commands are ignored)
float64[7] position              # joint 1..7 [rad] (firmware joint order)
float64[7] velocity              # finite-difference estimate [rad/s]
float64[6] coords                # flange in the arm base frame: x y z [mm], rx ry rz [deg] (firmware native)
float64[6] body_coords           # flange in the torso/body frame (firmware "base coords"), same units; NaN if not polled
uint8[] status_raw               # raw MERCURY_ROBOT_STATUS (0xA2) payload; all zeros = healthy
float32 state_rate_hz            # how fast joint/coord state is currently refreshed
string firmware_version          # arm MCU firmware (0x02 / 10, e.g. "1.1" - what myStudio checks); "" until read
string atom_version              # end-of-arm (atom) board firmware (0x09 / 10, e.g. "1.2"); "" until read
int16 board_version              # hardware / board version (0x01, e.g. 7); -1 until read
