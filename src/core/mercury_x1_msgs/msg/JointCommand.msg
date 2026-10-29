# Joint-space target for any subset of the upper-body joints.
# Names follow the URDF: joint1_L..joint7_L, joint1_R..joint7_R, eye (jaw tilt), head (neck pan), body (waist).
# Joints of one arm that are not listed keep their current position.
string[] name
float64[] position               # [rad], same length as name
uint8 speed                      # 1..100 (firmware speed), 0 = driver default
bool queue                       # false (default): preempt - stop the current motion, then go.
                                 # true: append to the firmware motion queue.
