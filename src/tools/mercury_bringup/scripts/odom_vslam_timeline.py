"""when does cuVSLAM go wrong: vo_state, its error vs wheel odometry, and the robot's turn rate, per 2 s"""
import math, sys, numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
W, V, ST = [], [], []
while r.has_next():
    topic, data, t = r.read_next()
    if topic == "/vslam/status":
        m = deserialize_message(data, get_message(types[topic])); ST.append((t * 1e-9, m.vo_state)); continue
    if topic not in ("/wheel/odom", "/vslam/odom"): continue
    m = deserialize_message(data, get_message(types[topic])); p = m.pose.pose; q = p.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    (W if topic == "/wheel/odom" else V).append((t * 1e-9, p.position.x, p.position.y, yaw, m.twist.twist.angular.z, m.twist.twist.linear.x))
W, V, ST = np.array(W), np.array(V), np.array(ST)
t0 = W[0, 0]; T = np.arange(0, W[-1, 0] - t0, 2.0)
def al(a):
    c, s = math.cos(-a[0, 3]), math.sin(-a[0, 3]); X, Y = a[:, 1] - a[0, 1], a[:, 2] - a[0, 2]; return c * X - s * Y, s * X + c * Y
wx, wy = al(W); vx, vy = al(V)
print(" t[s]  vo  wheel v[m/s] w[deg/s]   err[m]   (only rows where the robot moves or the error changes)")
pe = 0
for tt in T:
    i = np.searchsorted(W[:, 0], t0 + tt); j = np.searchsorted(V[:, 0], t0 + tt); k = np.searchsorted(ST[:, 0], t0 + tt)
    if i >= len(W) or j >= len(V): break
    e = math.hypot(wx[i] - vx[j], wy[i] - vy[j]); seg = W[max(0, i - 40):i + 1]
    v, w = np.abs(seg[:, 5]).max(), math.degrees(np.abs(seg[:, 4]).max())
    if v > 0.02 or w > 2 or abs(e - pe) > 0.05:
        print(f"{tt:5.0f}  {int(ST[min(k, len(ST) - 1), 1]):2d}  {v:6.2f}  {w:7.1f}  {e:7.2f}")
    pe = e
print("vo_state values seen:", sorted(set(ST[:, 1].astype(int))), " wheel-odom max |w| deg/s:", math.degrees(np.abs(W[:, 4]).max()).__round__(1), " max v:", np.abs(W[:, 5]).max().__round__(2))
