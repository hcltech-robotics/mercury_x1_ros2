"""Robot pose from /tf without a full tf2 buffer: caches only the 2D chain
map -> odom -> odom_wheel -> base_footprint and composes it on demand."""

from __future__ import annotations

import math

CHAIN = {"odom": "map", "odom_wheel": "odom", "base_footprint": "odom_wheel"}  # child -> parent


class PoseChain:
    def __init__(self) -> None:
        self.edges: dict[str, tuple[str, float, float, float]] = {}

    def update(self, msg) -> None:
        for t in msg.transforms:
            child = t.child_frame_id.lstrip("/")
            parent = t.header.frame_id.lstrip("/")
            if CHAIN.get(child) == parent:
                q = t.transform.rotation
                self.edges[child] = (parent, t.transform.translation.x, t.transform.translation.y,
                                     math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))

    def pose(self, frame: str = "base_footprint"):
        """(root_frame, x, y, yaw) of `frame` in the highest frame reachable through the chain."""
        x = y = yaw = 0.0
        while frame in self.edges:
            parent, tx, ty, tyaw = self.edges[frame]
            c, s = math.cos(tyaw), math.sin(tyaw)
            x, y, yaw = tx + c * x - s * y, ty + s * x + c * y, tyaw + yaw
            frame = parent
        if frame == "base_footprint":
            return None
        return frame, x, y, math.atan2(math.sin(yaw), math.cos(yaw))
