#!/usr/bin/env python3
"""Compare the Mercury X1 odometry sources live while you drive (remote / web teleop).

  source ~/mx1_env.sh && python3 ~/mercury_x1_ros2/scripts/odom_compare.py

Each source is zeroed at the robot's pose when the script starts; it then shows, per source,
the path length driven, the pose relative to the start (dx forward / dy left in the start
heading, dyaw) and the difference to `slam` (RTAB-Map map -> base_footprint, loop-closure
corrected), which is the best estimate. Ctrl+C prints a summary.

  wheel  /wheel/odom   wheel encoders + gyro yaw (base_node)
  icp    /odom         lidar scan matching (rtabmap icp_odometry), seeded by wheel
  vslam  /vslam/odom   cuVSLAM RGB-D visual odometry on the GPU (comparison only)
  slam   TF map -> base_footprint

Tests: drive a measured straight line (compare `path` with the tape), spin 360 deg in place
(dyaw should come back to 0), drive a loop and park on the start mark (everything should read
~0: what remains is that source's drift).
"""
from __future__ import annotations

import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from tf2_ros import Buffer, TransformListener

SOURCES = (("wheel", "/wheel/odom"), ("icp", "/odom"), ("vslam", "/vslam/odom"))


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class Track:
    def __init__(self) -> None:
        self.start = None      # (x, y, yaw)
        self.last = None
        self.path = 0.0
        self.turn = 0.0        # accumulated |dyaw|
        self.stamp = 0.0

    def update(self, x: float, y: float, yaw: float) -> None:
        if self.start is None:
            self.start = self.last = (x, y, yaw)
        lx, ly, lyaw = self.last
        step, dyaw = math.hypot(x - lx, y - ly), abs(wrap(yaw - lyaw))
        # path / turn totals only count real motion: standing still, scan matching jitters by ~1 mm
        if step < 0.003 and dyaw < math.radians(0.3):
            self.stamp = time.monotonic()
            return
        self.path += step
        self.turn += dyaw
        self.last = (x, y, yaw)
        self.stamp = time.monotonic()

    def rel(self):
        """pose relative to the start, in the start heading's frame"""
        if self.start is None:
            return None
        sx, sy, syaw = self.start
        x, y, yaw = self.last
        dx, dy = x - sx, y - sy
        c, s = math.cos(-syaw), math.sin(-syaw)
        return c * dx - s * dy, s * dx + c * dy, wrap(yaw - syaw)


class OdomCompare(Node):
    def __init__(self) -> None:
        super().__init__("odom_compare")
        self.tracks = {name: Track() for name, _ in SOURCES}
        self.tracks["slam"] = Track()
        for name, topic in SOURCES:
            self.create_subscription(Odometry, topic, lambda m, n=name: self._on_odom(n, m), 20)
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.create_timer(0.1, self._poll_tf)
        self.create_timer(0.5, self._print)
        self.t0 = time.monotonic()

    def _on_odom(self, name: str, m: Odometry) -> None:
        p = m.pose.pose
        self.tracks[name].update(p.position.x, p.position.y, yaw_of(p.orientation))

    def _poll_tf(self) -> None:
        try:
            t = self.tf.lookup_transform("map", "base_footprint", rclpy.time.Time())
        except Exception:
            return
        tr = t.transform
        self.tracks["slam"].update(tr.translation.x, tr.translation.y, yaw_of(tr.rotation))

    def rows(self):
        ref = self.tracks["slam"].rel()
        for name, tr in self.tracks.items():
            r = tr.rel()
            if r is None:
                yield f"{name:6s}  (no data)"
                continue
            stale = time.monotonic() - tr.stamp > 2.0
            line = (f"{name:6s}  path {tr.path:6.2f} m  turned {math.degrees(tr.turn):7.1f}°  "
                    f"dx {r[0]:+6.2f}  dy {r[1]:+6.2f}  dyaw {math.degrees(r[2]):+7.1f}°")
            if ref is not None and name != "slam":
                err = math.hypot(r[0] - ref[0], r[1] - ref[1])
                line += f"  | vs slam {err:5.2f} m {math.degrees(wrap(r[2] - ref[2])):+6.1f}°"
            yield line + ("  STALE" if stale else "")

    def _print(self) -> None:
        print("\033[H\033[J" + f"odometry comparison  ({time.monotonic() - self.t0:5.0f} s, Ctrl+C = summary)\n")
        for line in self.rows():
            print(line)


def main() -> None:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)   # Ctrl+C -> KeyboardInterrupt -> summary
    node = OdomCompare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    print("\nsummary (relative to the start pose):")
    for line in node.rows():
        print("  " + line)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
