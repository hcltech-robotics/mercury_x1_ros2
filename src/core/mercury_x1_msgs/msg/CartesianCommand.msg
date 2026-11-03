# Cartesian target for one arm's flange, in firmware-native units.
uint8 FRAME_ARM = 0              # the arm's own base frame (same frame as ArmState.coords)
uint8 FRAME_BODY = 1             # torso/body frame (same frame as ArmState.body_coords)

string arm                       # "left" | "right"
uint8 frame
float64[6] coords                # x y z [mm], rx ry rz [deg]
uint8 speed                      # 1..100, 0 = driver default
bool joint_interpolated          # false (default): linear (moveL) path; true: moveJ to the pose
bool queue                       # false (default): preempt current motion; true: append to firmware queue
