"""Keep-out zones drawn on the page's map (a glass wall, a doorway, a desk area): the robot must not drive there.

Stored per map in ~/maps/<map>.keepout.json as [{"name", "points": [[x, y], ...]}] in the map frame. Two points
are a wall line (WALL_HALF_WIDTH_M either side), three or more a polygon. web_server publishes them for Nav2's
KeepoutFilter (both costmaps, config/nav2.yaml): a mask OccupancyGrid (100 = keep out) covering only the zones,
and the CostmapFilterInfo that points the filter at it; both latched, so Nav2 picks them up whenever it starts.
Costmap filters run after the inflation layer, so the mask itself is grown by MARGIN_M: the footprint check of
the controller and the planner then keep the robot that far off the line the user drew (the page shows that
margin as a halo around each zone)."""
from __future__ import annotations

import json
import math
import os

import numpy as np

RES = 0.05                 # m per mask cell (the map's resolution)
WALL_HALF_WIDTH_M = 0.05   # a two-point zone = a 10 cm thick wall
MARGIN_M = 0.25            # clearance around every zone: ~ the base's half width (0.24 m). The global planner (NavFn)
                           # plans for a point and filters skip the inflation layer, so the mask carries the clearance
MASK_TOPIC = "/keepout_filter_mask"
INFO_TOPIC = "/costmap_filter_info"


def load(path: str) -> list[dict]:
    try:
        with open(path) as f:
            zones = json.load(f)
    except (OSError, ValueError):
        return []
    return clean(zones)


def save(path: str, zones: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(zones, f, indent=1)
    os.replace(tmp, path)


def clean(zones) -> list[dict]:
    """validated copy: names, 2+ finite points each"""
    out = []
    for k, z in enumerate(zones if isinstance(zones, list) else []):
        try:
            pts = [[round(float(x), 3), round(float(y), 3)] for x, y in z.get("points", [])]
        except (TypeError, ValueError):
            continue
        pts = [p for p in pts if all(math.isfinite(v) for v in p)]
        if len(pts) >= 2:
            name = str(z.get("name") or f"zone {k + 1}").strip()[:40] or f"zone {k + 1}"
            out.append({"name": name, "points": pts})
    return out


def _seg_dist(px: np.ndarray, py: np.ndarray, a, b) -> np.ndarray:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    t = np.zeros_like(px) if L2 == 0 else np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _inside(px: np.ndarray, py: np.ndarray, pts) -> np.ndarray:
    """even-odd point-in-polygon"""
    inside = np.zeros(px.shape, bool)
    n = len(pts)
    for i in range(n):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % n]
        cross = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cross & (px < xint)
    return inside


def mask(zones: list[dict]) -> tuple[np.ndarray, float, float]:
    """(grid int8 [rows, cols] with 100 inside, origin x, origin y); row 0 = lowest y (OccupancyGrid order)"""
    if not zones:
        return np.zeros((1, 1), np.int8), 0.0, 0.0
    pad = MARGIN_M + WALL_HALF_WIDTH_M + 2 * RES
    allp = np.array([p for z in zones for p in z["points"]], float)
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad
    cols, rows = int(math.ceil((x1 - x0) / RES)), int(math.ceil((y1 - y0) / RES))
    xs = x0 + (np.arange(cols) + 0.5) * RES
    ys = y0 + (np.arange(rows) + 0.5) * RES
    px, py = np.meshgrid(xs, ys)
    hit = np.zeros(px.shape, bool)
    for z in zones:
        pts = z["points"]
        edges = list(zip(pts, pts[1:])) if len(pts) == 2 else list(zip(pts, pts[1:] + pts[:1]))
        reach = WALL_HALF_WIDTH_M + MARGIN_M if len(pts) == 2 else MARGIN_M
        d = np.min([_seg_dist(px, py, a, b) for a, b in edges], axis=0)
        hit |= d <= reach
        if len(pts) >= 3:
            hit |= _inside(px, py, pts)
    return np.where(hit, 100, 0).astype(np.int8), float(x0), float(y0)


def messages(zones: list[dict], stamp):
    """(OccupancyGrid mask, CostmapFilterInfo) for the KeepoutFilter"""
    from nav_msgs.msg import OccupancyGrid
    from nav2_msgs.msg import CostmapFilterInfo
    grid, x0, y0 = mask(zones)
    m = OccupancyGrid()
    m.header.frame_id, m.header.stamp = "map", stamp
    m.info.map_load_time = stamp
    m.info.resolution = RES
    m.info.height, m.info.width = grid.shape
    m.info.origin.position.x, m.info.origin.position.y = x0, y0
    m.info.origin.orientation.w = 1.0
    m.data = grid.ravel().tolist()
    info = CostmapFilterInfo()
    info.header.frame_id, info.header.stamp = "map", stamp
    info.type = 0                    # keepout
    info.filter_mask_topic = MASK_TOPIC
    info.base, info.multiplier = 0.0, 1.0
    return m, info
