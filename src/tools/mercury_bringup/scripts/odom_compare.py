"""odo_cmp.py BAG: compare odometry sources in a rosbag2 (wheel, ICP, cuVSLAM, RTAB-Map localization)."""
import math, sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag = sys.argv[1]
r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = {"/wheel/odom": "wheel", "/odom": "icp", "/vslam/odom": "cuvslam", "/localization_pose": "rtabmap"}
tr = {v: [] for v in want.values()}
while r.has_next():
    topic, data, t = r.read_next()
    if topic not in want: continue
    m = deserialize_message(data, get_message(types[topic]))
    p = m.pose.pose if hasattr(m.pose, "pose") else m.pose
    q = p.orientation; yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    tr[want[topic]].append((t * 1e-9, p.position.x, p.position.y, p.position.z, yaw))
def stats(a):
    a = np.array(a); d = np.diff(a[:, 1:3], axis=0); L = np.linalg.norm(d, axis=1)
    L = L[L < 0.5]  # drop jumps (relocalisation)
    dz = a[-1, 3] - a[0, 3]; ret = np.hypot(a[-1, 1] - a[0, 1], a[-1, 2] - a[0, 2])
    dyaw = np.unwrap(a[:, 4]); turn = np.abs(np.diff(dyaw)).sum(); net = math.degrees(dyaw[-1] - dyaw[0])
    return len(a), a[-1, 0] - a[0, 0], L.sum(), ret, dz, math.degrees(turn), net
print(f"{'source':9s} {'msgs':>6s} {'dur s':>6s} {'path m':>7s} {'return m':>9s} {'dz m':>6s} {'|turn| deg':>10s} {'net yaw deg':>11s}")
for k, a in tr.items():
    if len(a) < 2: print(f"{k:9s}  (no data)"); continue
    n, dur, L, ret, dz, turn, net = stats(a)
    print(f"{k:9s} {n:6d} {dur:6.1f} {L:7.2f} {ret:9.3f} {dz:6.3f} {turn:10.0f} {net:11.1f}")
# trajectories aligned at the start (translate + rotate so the first pose is the origin facing +x), sampled every 2 s
def aligned(a):
    a = np.array(a); x0, y0, yaw0 = a[0, 1], a[0, 2], a[0, 4]; c, s = math.cos(-yaw0), math.sin(-yaw0)
    X = a[:, 1] - x0; Y = a[:, 2] - y0; return a[:, 0], c * X - s * Y, s * X + c * Y, np.degrees(np.unwrap(a[:, 4]) - yaw0)
ref = aligned(tr["wheel"]); t0 = ref[0][0]; T = np.arange(0, ref[0][-1] - t0, 2.0)
print("\nposition vs wheel odometry (start-aligned), every 2 s: dist from wheel estimate [m], yaw diff [deg]")
for k in ("icp", "cuvslam", "rtabmap"):
    if len(tr[k]) < 2: continue
    o = aligned(tr[k]); errs = []; yerrs = []
    for tt in T:
        i = np.searchsorted(ref[0], t0 + tt); j = np.searchsorted(o[0], t0 + tt)
        if i >= len(ref[0]) or j >= len(o[0]): break
        errs.append(math.hypot(ref[1][i] - o[1][j], ref[2][i] - o[2][j])); yerrs.append(((ref[3][i] - o[3][j]) + 180) % 360 - 180)
    e = np.array(errs); print(f"{k:9s} max {e.max():.2f} m, final {e[-1]:.2f} m, yaw diff final {yerrs[-1]:.1f} deg, max |yaw| {np.abs(yerrs).max():.1f}")
    print("   ", " ".join(f"{v:.2f}" for v in e[::3]))
np.save(bag.rstrip("/") + "_tracks.npy", {k: np.array(v) for k, v in tr.items()}, allow_pickle=True)
