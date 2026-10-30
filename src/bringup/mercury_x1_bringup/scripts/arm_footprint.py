#!/usr/bin/env python3
"""Arm-aware Nav2 footprint: the base outline grown to cover whatever part of the arms sticks out.

Every tick the arm links' hull points (config/arm_hulls.yaml, from the URDF meshes) plus a tool
point in front of each flange are put through TF into base_footprint and flattened to the floor.
Points that stay inside the base outline are ignored (nav2.yaml's footprint already covers the arms
in the carry pose); if any point is further out than `tolerance`, the footprint becomes the convex
hull of the base outline and those points, padded by `padding`.

The result goes to <costmap>/footprint (geometry_msgs/Polygon), which Nav2's costmaps take over
their `footprint` parameter: MPPI (consider_footprint) and the behaviours check collisions with it,
the costmaps' inflation follows it. It is published only when it changes (a footprint change makes
the costmaps re-inflate) and again whenever a costmap (re)subscribes.
"""
from __future__ import annotations

import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32, Polygon
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain), counter-clockwise"""
    p = sorted(map(tuple, pts))
    if len(p) < 3:
        return np.array(p)

    def half(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(reversed(p))
    return np.array(lower[:-1] + upper[:-1])


def outside_distance(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """distance of each point outside a convex counter-clockwise polygon (<= 0 inside)"""
    a, b = poly, np.roll(poly, -1, axis=0)
    e = b - a
    n = np.stack([e[:, 1], -e[:, 0]], 1) / np.linalg.norm(e, axis=1)[:, None]   # outward normals
    return ((pts[:, None, :] - a[None]) * n[None]).sum(-1).max(1)


def pad_points(pts: np.ndarray, r: float, segments: int = 8) -> np.ndarray:
    """points grown by a disc of radius r (each hull vertex -> a ring of points)"""
    if r <= 0:
        return pts
    hull = convex_hull(pts) if len(pts) >= 3 else pts
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = r * np.stack([np.cos(a), np.sin(a)], 1)
    return (hull[:, None, :] + ring[None]).reshape(-1, 2)


class ArmFootprint(Node):
    def __init__(self) -> None:
        super().__init__("arm_footprint")
        p = self.declare_parameter
        base = yaml.safe_load(p("base_footprint", "[[0.22, 0.24], [0.22, -0.24], [-0.39, -0.24], [-0.39, 0.24]]").value)
        self.base = convex_hull(np.array(base, float))
        self.frame = p("robot_base_frame", "base_footprint").value
        self.tolerance = float(p("tolerance", 0.05).value)        # ignore arm parts this little outside the base
        self.padding = float(p("padding", 0.04).value)            # clearance added around protruding arm parts
        self.tool_length = float(p("tool_length", 0.08).value)    # hand / gripper beyond the flange (+z of link7)
        self.min_change = float(p("min_change", 0.02).value)
        self.max_height = float(p("max_height", 2.0).value)       # arm parts above this cannot hit anything
        costmaps = list(p("costmaps", ["local_costmap", "global_costmap"]).value)
        hull_file = p("hulls_file", os.path.join(get_package_share_directory("mercury_x1_bringup"), "config",
                                                 "arm_hulls.yaml")).value
        with open(hull_file) as f:
            self.hulls = {k: np.array(v, float) for k, v in yaml.safe_load(f).items()}
        for side in ("L", "R"):
            flange = f"link7_{side}"
            if flange in self.hulls and self.tool_length > 0:
                self.hulls[flange] = np.vstack([self.hulls[flange], [[0.0, 0.0, self.tool_length]]])

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pubs = {c: self.create_publisher(Polygon, f"/{c}/footprint", 1) for c in costmaps}
        self.subs_seen = {c: 0 for c in costmaps}
        self.current: np.ndarray | None = None
        self.expanded = False
        self.create_timer(1.0 / float(p("rate", 5.0).value), self._tick)
        self.get_logger().info(f"arm-aware footprint for {costmaps} ({len(self.hulls)} links, tool {self.tool_length} m)")

    def _arm_points(self) -> np.ndarray | None:
        pts = []
        for link, hull in self.hulls.items():
            try:
                t = self.tf.lookup_transform(self.frame, link, Time()).transform
            except TransformException:
                return None
            tr = t.translation
            w = hull @ quat_matrix(t.rotation).T + np.array([tr.x, tr.y, tr.z])
            pts.append(w[w[:, 2] <= self.max_height, :2])
        return np.vstack(pts) if pts else None

    def _footprint(self) -> np.ndarray:
        arm = self._arm_points()
        if arm is None:                          # no arm TF (arms off / driver down): the base outline
            return self.base
        out = arm[outside_distance(self.base, arm) > self.tolerance]
        if len(out) == 0:
            return self.base
        grown = pad_points(out, self.padding)
        return convex_hull(np.vstack([self.base, grown]))

    def _tick(self) -> None:
        fp = self._footprint()
        changed = (self.current is None or len(fp) != len(self.current)
                   or np.abs(fp - self.current).max() > self.min_change)
        expanded = fp is not self.base
        if changed:
            self.current = fp
            if expanded != self.expanded or expanded:
                ext = fp.min(0), fp.max(0)
                self.get_logger().info(
                    "footprint: " + ("base outline (arms inside)" if not expanded else
                                     f"grown for the arms, x {ext[0][0]:+.2f}..{ext[1][0]:+.2f} "
                                     f"y {ext[0][1]:+.2f}..{ext[1][1]:+.2f} m"))
            self.expanded = expanded
        for c, pub in self.pubs.items():
            n = pub.get_subscription_count()
            if changed or n > self.subs_seen[c]:     # a costmap (re)started: give it the current footprint
                pub.publish(Polygon(points=[Point32(x=float(x), y=float(y)) for x, y in self.current]))
            self.subs_seen[c] = n


def main() -> None:
    rclpy.init()
    node = ArmFootprint()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
