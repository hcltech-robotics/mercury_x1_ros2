"""ROS 2 -> Rerun bridge for the Mercury X1.

Mirrors /tf into Rerun named frames (Transform3D parent_frame/child_frame) and logs every
data entity with CoordinateFrame(<ros frame_id>), so the viewer resolves poses the way tf2
does. Serves the data over gRPC (default :9876) and a stand-alone web viewer (:9090); the
control page served by web_server embeds the same stream.

All rr.log calls go through one worker thread and a bounded queue: the gRPC server feeds every
viewer in lockstep, so one slow or stale viewer (e.g. a tab still running an older viewer) blocks
rr.log. Without the queue that froze this node's ROS callbacks (TF, parameters); now frames are dropped.

nvblox outputs are optional layers (parameter `nvblox_layers`, switched from the control page):
a layer that is off is not subscribed, so it costs neither this bridge nor nvblox anything.
"""

from __future__ import annotations

import json
import math
import os
import queue
import socket
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
import rerun as rr
import rerun.blueprint as rrb
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PolygonStamped, PoseArray
from mercury_x1_msgs.msg import MeshTile
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import CameraInfo, CompressedImage, LaserScan, PointCloud2, Range
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

from .pose_chain import PoseChain
from .slam3d import voxel_down

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
# /tf_static has several latched publishers (robot_state_publisher, orbbec driver, ...): with
# depth 1 one sample would overwrite the other, so keep plenty like tf2's StaticListenerQoS.
TF_STATIC = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE)


RELIABLE_1 = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
VIZ_LAYERS_FILE = os.path.expanduser("~/.config/mercury_x1/viz_layers.json")   # written by web_server
# key -> (message type, topic, QoS, max rate [Hz]); entities live under world/nvblox/<key>
NVBLOX_LAYERS = {
    "mesh": (MeshTile, "/nvblox/mesh_tiles", QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE), 0.0),
    "esdf": (PointCloud2, "/nvblox_node/static_esdf_pointcloud", RELIABLE_1, 2.0),
    "depth": (PointCloud2, "/nvblox_node/back_projected_depth/camera_color_optical_frame", RELIABLE_1, 2.0),
    "tsdf": (Marker, "/nvblox_node/tsdf_layer_marker", RELIABLE_1, 1.0),
    "color": (Marker, "/nvblox_node/color_layer_marker", RELIABLE_1, 1.0),
    "occupancy": (OccupancyGrid, "/nvblox_node/static_occupancy_grid", RELIABLE_1, 2.0),
    # not nvblox: the whole RTAB-Map map as a coloured 3D cloud from its stored RGB-D frames, aligned to
    # the map frame (web_server re-exports it every 20 s while this is subscribed)
    "slam3d": (PointCloud2, "/slam/cloud3d", LATCHED, 0.0),
}
VIEW_LAYERS = ("camera",)   # switchable like the nvblox layers, but no subscription: camera frustums + image
# The viewer gets voxel grids, not raw clouds (mapping still uses the full-resolution depth):
DEPTH_VOXEL = 0.05         # [m] live depth points (~100k points/frame -> a few thousand voxels)
# Voxel map (whole RTAB-Map map): web_server publishes it at 3 cm; the bridge keeps that copy and re-voxelises
# it at `voxel_size`, drawn `voxel_shape` = round (points) | square (solid cubes), in 2 m static tiles of
# which only the changed ones are re-sent. Both are parameters (page settings), applied without a re-export.
VOXEL_SHAPES = ("round", "square")
VOXEL_SIZES = (0.03, 0.2)  # [m] allowed range
SLAM3D_TILE = 2.0          # [m]


def _turbo(t: np.ndarray) -> np.ndarray:
    """Turbo colormap (polynomial fit), t in [0, 1] -> uint8 RGB."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    c = np.hstack([
        0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))),
        0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))),
        0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))),
    ])
    return (np.clip(c, 0, 1) * 255).astype(np.uint8)


def _stamp(msg) -> float:
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


class Throttle:
    def __init__(self, hz: float):
        self.period = 1.0 / hz if hz > 0 else 0.0
        self.last: dict[str, float] = {}

    def __call__(self, key: str = "") -> bool:
        now = time.monotonic()
        if now - self.last.get(key, 0.0) < self.period:
            return False
        self.last[key] = now
        return True


def cloud_fields(msg: PointCloud2, names: str = "xyz") -> np.ndarray:
    """Fast extraction of float32 fields (columns in `names` order) from a PointCloud2; non-finite rows dropped."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in names):
        return np.zeros((0, len(names)), np.float32)
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    out = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in names], 1)
    return out[np.isfinite(out).all(1)]


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    return cloud_fields(msg, "xyz")


class RerunBridge(Node):
    def __init__(self) -> None:
        super().__init__("rerun_bridge")
        p = self.declare_parameter
        grpc_port = int(p("grpc_port", 9876).value)
        web_port = int(p("web_port", 9090).value)
        # the server keeps this much history and replays it to every newly connected viewer; newest first,
        # so a new viewer shows the live state at once and the history fills in behind it
        mem = p("memory_limit", "64MiB").value
        self.img_hz = float(p("image_rate", 4.0).value)
        self.cloud_hz = float(p("cloud_rate", 3.0).value)
        self.scan_hz = float(p("scan_rate", 10.0).value)
        # the bellybutton camera has no calibration file (camera_info K is zero): approximate it
        self.belly_hfov = math.radians(float(p("bellybutton_hfov_deg", 75.0).value))
        self.tf_hz = float(p("tf_rate", 10.0).value)            # odometry chain
        self.joint_tf_hz = float(p("joint_tf_rate", 5.0).value)  # robot links (joint states)

        rr.init("mercury_x1", recording_id="mercury_x1_live")
        self.log_q: queue.Queue = queue.Queue(maxsize=int(p("log_queue", 300).value))
        self.log_t: float | None = None      # "ros" timestamp for the next _log (set by _set_time)
        self.log_dropped = 0
        threading.Thread(target=self._log_worker, daemon=True, name="rerun_log").start()
        self.mesh_dir = os.path.join(get_package_share_directory("mercury_x1_web"), "meshes")
        self.robot_parts = self._robot_parts()
        uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit=mem, newest_first=True,
                            cors_allow_origin=["*"], default_blueprint=self._blueprint(self.robot_parts))
        # after a hard restart the old viewer port lingers in TIME_WAIT for ~60 s and rerun's web server
        # then hangs instead of failing: wait until the port can really be bound
        for _ in range(120):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("0.0.0.0", web_port))
                    break
                except OSError:
                    time.sleep(1.0)
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
        self.get_logger().info(f"rerun gRPC {uri}, viewer http://<robot>:{web_port}")

        # the 3D view's root: tie the ROS "map" frame to the view origin
        self._log("tf/map", rr.Transform3D(parent_frame="tf#/", child_frame="map"), static=True)
        self._log_robot_model()

        self.thr_tf = Throttle(self.tf_hz)
        self.thr_joint_tf = Throttle(self.joint_tf_hz)
        self.thr_img = Throttle(self.img_hz)
        self.thr_frustum = Throttle(2.0)
        self.thr_cloud = Throttle(self.cloud_hz)
        self.thr_scan = Throttle(self.scan_hz)
        self.thr_us = Throttle(5.0)
        self.thr_costmap = Throttle(1.0)
        self.thr_slow = Throttle(1.0)
        self.cam_info: dict[str, CameraInfo] = {}
        self.trail: list[list[float]] = []
        self.lock = threading.Lock()

        sub = self.create_subscription
        sub(TFMessage, "/tf", lambda m: self._on_tf(m, False), 100)
        sub(TFMessage, "/tf_static", lambda m: self._on_tf(m, True), TF_STATIC)
        sub(OccupancyGrid, "/map", self._on_map, LATCHED)
        sub(OccupancyGrid, "/local_costmap/costmap", self._on_costmap, 1)
        sub(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        sub(PointCloud2, "/camera/cloud", self._on_cloud, 2)
        # JPEG previews produced in C++ by mercury_x1_driver/image_preview_node (4 Hz)
        sub(CompressedImage, "/preview/orbbec", lambda m: self._on_image(m, "orbbec"), qos_profile_sensor_data)
        sub(CompressedImage, "/preview/bellybutton", lambda m: self._on_image(m, "bellybutton"),
            qos_profile_sensor_data)
        sub(CameraInfo, "/camera/color/camera_info", lambda m: self.cam_info.__setitem__("orbbec", m),
            qos_profile_sensor_data)
        sub(CameraInfo, "/bellybutton/camera_info", lambda m: self.cam_info.__setitem__("bellybutton", m),
            qos_profile_sensor_data)
        for name in ("left", "back", "right"):
            sub(Range, f"/ultrasonic/{name}", lambda m, n=name: self._on_range(m, n), qos_profile_sensor_data)
        sub(Path, "/plan", self._on_plan, 1)
        # cuVSLAM RGB-D visual odometry (mercury_x1_perception/cuvslam_rgbd_node), for comparison
        sub(Path, "/vslam/path", self._on_vslam_path, 1)
        sub(PoseArray, "/web/waypoints", self._on_waypoints, LATCHED)
        sub(PolygonStamped, "/local_costmap/published_footprint", self._on_footprint, 1)
        self.chain = PoseChain()
        self.create_timer(0.5, self._trail_tick)

        # nvblox layers: the mesh (pre-tiled in C++, mercury_x1_perception/nvblox_mesh_tiler), the ESDF
        # distance slice, the back-projected depth, TSDF / colour voxels, the occupancy grid
        self.nvblox_handlers = {"mesh": self._on_mesh_tile, "esdf": self._on_esdf, "depth": self._on_nvblox_depth,
                                "tsdf": lambda m: self._on_voxels(m, "tsdf"),
                                "color": lambda m: self._on_voxels(m, "color"), "occupancy": self._on_nvblox_occupancy,
                                "slam3d": self._on_slam3d}
        self.nvblox_subs: dict = {}
        self.thr_nvblox = {k: Throttle(v[3]) for k, v in NVBLOX_LAYERS.items()}
        self.mesh_tiles: set[str] = set()          # tile ids logged, blanked when the mesh layer goes off
        # voxel map: the latest 3 cm cloud (frame, xyz, rgb) and per shape: tile -> hash of what the viewer has
        self.slam3d_cloud: tuple | None = None
        self.slam3d_tiles: dict[str, dict[str, int]] = {k: {} for k in VOXEL_SHAPES}
        shape, size = self._saved_voxel()
        self.voxel_shape = p("voxel_shape", shape).value
        self.voxel_size = float(p("voxel_size", size).value)
        self.voxel_changed = False
        self.layers_wanted: list[str] | None = None
        self.show_camera = True                    # "camera" layer: frustums with the live image
        p("nvblox_layers", self._saved_layers())
        self.add_on_set_parameters_callback(self._on_params)
        self._apply_layers(list(self.get_parameter("nvblox_layers").value))
        self.create_timer(0.2, self._layers_tick)

    # ---- logging through the worker thread ---------------------------------------------
    def _set_time(self, t: float) -> None:
        self.log_t = t

    def _log(self, *args, **kwargs) -> None:
        item = (self.log_t, args, kwargs)
        try:
            self.log_q.put_nowait(item)
        except queue.Full:
            if kwargs.get("static"):            # meshes / robot model / map tiles: wait rather than lose them
                try:
                    self.log_q.put(item, timeout=2.0)
                    return
                except queue.Full:
                    pass
            self.log_dropped += 1

    def _log_worker(self) -> None:
        last_report = time.monotonic()
        while True:
            t, args, kwargs = self.log_q.get()
            try:
                if t is not None and not kwargs.get("static"):
                    rr.set_time("ros", timestamp=t)
                rr.log(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 - one bad message must not stop the stream
                self.get_logger().warn(f"rerun log {args[0] if args else '?'}: {e}", throttle_duration_sec=10.0)
            if self.log_dropped and time.monotonic() - last_report > 10.0:
                self.get_logger().warn(f"viewer stream backed up: dropped {self.log_dropped} updates "
                                       f"(a slow or stale viewer is connected)")
                self.log_dropped, last_report = 0, time.monotonic()

    # ---- nvblox layers -------------------------------------------------------------
    @staticmethod
    def _saved_layers() -> list[str]:
        try:
            with open(VIZ_LAYERS_FILE) as f:
                saved = json.load(f)
            layers = [("slam3d" if k == "voxmap" else k) for k in saved.get("nvblox", [])]   # voxmap: old name
            layers = list(dict.fromkeys(k for k in layers if k in NVBLOX_LAYERS or k in VIEW_LAYERS))
            if saved.get("version", 1) < 2:   # saved before the camera layer existed: on, like before
                layers.append("camera")
            return layers or [""]           # rclpy cannot infer the type of an empty list
        except (OSError, ValueError, AttributeError):
            return ["mesh", "depth", "camera"]

    @staticmethod
    def _saved_voxel() -> tuple[str, float]:
        try:
            with open(VIZ_LAYERS_FILE) as f:
                v = json.load(f).get("voxel", {})
            return (v.get("shape") if v.get("shape") in VOXEL_SHAPES else "square",
                    min(max(float(v.get("size", 0.05)), VOXEL_SIZES[0]), VOXEL_SIZES[1]))
        except (OSError, ValueError, AttributeError, TypeError):
            return "square", 0.05

    def _on_params(self, params) -> SetParametersResult:
        for prm in params:
            if prm.name == "voxel_shape":
                if prm.value not in VOXEL_SHAPES:
                    return SetParametersResult(successful=False, reason=f"voxel_shape: one of {VOXEL_SHAPES}")
                self.voxel_shape, self.voxel_changed = prm.value, True
            elif prm.name == "voxel_size":
                if not VOXEL_SIZES[0] <= float(prm.value) <= VOXEL_SIZES[1]:
                    return SetParametersResult(successful=False, reason=f"voxel_size: {VOXEL_SIZES} m")
                self.voxel_size, self.voxel_changed = float(prm.value), True
            elif prm.name == "nvblox_layers":
                bad = [k for k in prm.value if k and k not in NVBLOX_LAYERS and k not in VIEW_LAYERS]
                if bad:
                    return SetParametersResult(successful=False, reason=f"unknown layers {bad}")
                self.layers_wanted = [k for k in prm.value if k]   # applied by the timer, outside the callback
        return SetParametersResult(successful=True)

    def _layers_tick(self) -> None:
        if self.voxel_changed:                    # page changed shape / size: re-draw from the kept cloud
            self.voxel_changed = False
            self._render_slam3d()
        if self.layers_wanted is not None:
            wanted, self.layers_wanted = self.layers_wanted, None
            self._apply_layers(wanted)

    def _apply_layers(self, wanted: list[str]) -> None:
        camera = "camera" in wanted
        if self.show_camera and not camera:
            self._set_time(time.time())
            for name in ("orbbec", "bellybutton"):
                self._log(f"world/{name}_camera", rr.Clear(recursive=True))
        self.show_camera = camera
        wanted = [k for k in wanted if k in NVBLOX_LAYERS]
        for key in [k for k in self.nvblox_subs if k not in wanted]:
            self.destroy_subscription(self.nvblox_subs.pop(key))
            self._clear_layer(key)
        for key in wanted:
            if key not in self.nvblox_subs:
                ty, topic, qos, _ = NVBLOX_LAYERS[key]
                self.nvblox_subs[key] = self.create_subscription(ty, topic, self.nvblox_handlers[key], qos)
        self.get_logger().info(f"nvblox layers: {sorted(self.nvblox_subs) or 'none'}, camera {'on' if camera else 'off'}")

    def _clear_layer(self, key: str) -> None:
        if key == "slam3d":  # static tiles: blank each (the latched cloud is re-sent on resubscribe)
            for shape in VOXEL_SHAPES:
                self._blank_tiles(shape, list(self.slam3d_tiles[shape]))
            self.slam3d_cloud = None
            return
        if key == "mesh":   # static tiles: overwrite each with an empty mesh (the tiler resends all on resubscribe)
            for tid in self.mesh_tiles:
                self._log(f"world/nvblox/mesh/{tid}", rr.Mesh3D(vertex_positions=np.zeros((0, 3), np.float32)),
                       static=True)
            self.mesh_tiles.clear()
            return
        self._set_time(time.time())
        self._log(f"world/nvblox/{key}", rr.Clear(recursive=True))

    def _on_esdf(self, msg: PointCloud2) -> None:
        """ESDF slice: distance to the nearest obstacle at the slice height, red = touching, blue >= 1.5 m"""
        if not self.thr_nvblox["esdf"]():
            return
        pts = cloud_fields(msg, ("x", "y", "z", "intensity"))
        self._set_time(_stamp(msg))
        self._log("world/nvblox/esdf", rr.CoordinateFrame(msg.header.frame_id),
               rr.Points3D(pts[:, :3], radii=0.022, colors=_turbo(1.0 - pts[:, 3] / 1.5)))

    def _on_nvblox_depth(self, msg: PointCloud2) -> None:
        """the depth image nvblox integrates, back-projected to 3D (camera frame), coloured by range"""
        if not self.thr_nvblox["depth"]():
            return
        xyz, _ = voxel_down(cloud_xyz(msg), None, DEPTH_VOXEL)
        self._set_time(_stamp(msg))
        self._log("world/nvblox/depth", rr.CoordinateFrame(msg.header.frame_id),
               rr.Points3D(xyz, radii=DEPTH_VOXEL * 0.45, colors=_turbo(np.linalg.norm(xyz, axis=1) / 4.0)))

    def _on_voxels(self, msg: Marker, key: str) -> None:
        """TSDF / colour layer as voxel cubes (nvblox publishes CUBE_LIST markers with per-voxel colours)"""
        if not self.thr_nvblox[key]() or not msg.points:
            return
        centers = np.array([(q.x, q.y, q.z) for q in msg.points], np.float32)
        if len(msg.colors) == len(msg.points):
            colors = (np.array([(c.r, c.g, c.b) for c in msg.colors], np.float32) * 255).astype(np.uint8)
        else:
            colors = [[180, 180, 180]]
        h = 0.5 * float(msg.scale.x or 0.05)
        self._set_time(_stamp(msg))
        self._log(f"world/nvblox/{key}", rr.CoordinateFrame(msg.header.frame_id),
               rr.Boxes3D(centers=centers, half_sizes=[[h, h, h]], colors=colors, fill_mode="solid"))

    def _blank_tiles(self, shape: str, tids: list[str]) -> None:
        """static data cannot be cleared: overwrite tiles with an empty shape of their kind"""
        empty = np.zeros((0, 3), np.float32)
        for tid in tids:
            self.slam3d_tiles[shape].pop(tid, None)
            self._log(f"world/slam3d/{shape}/{tid}",
                      rr.Boxes3D(centers=empty, half_sizes=empty) if shape == "square" else rr.Points3D(empty),
                      static=True)

    def _on_slam3d(self, msg: PointCloud2) -> None:
        """whole map, 3 cm voxels in the map frame (web_server): kept, then drawn at the page's size / shape"""
        offs = {f.name: f.offset for f in msg.fields}
        n = msg.width * msg.height
        if n == 0 or not all(k in offs for k in ("x", "y", "z")):
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
        xyz = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
        if "rgb" in offs:
            packed = buf[:, offs["rgb"]:offs["rgb"] + 4].copy().view(np.uint32)[:, 0]
            rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255], 1).astype(np.uint8)
        else:
            rgb = np.full((len(xyz), 3), 200, np.uint8)
        self.slam3d_cloud = (msg.header.frame_id or "map", xyz, rgb)
        self._render_slam3d()

    def _render_slam3d(self) -> None:
        """2 m x 2 m static tiles (the viewer keeps only the latest of each); unchanged tiles are not re-sent,
        so a growing map streams only where it grew"""
        if self.slam3d_cloud is None:
            return
        t0 = time.monotonic()
        frame, xyz, rgb = self.slam3d_cloud
        shape, size = self.voxel_shape, self.voxel_size
        other = [t for s in VOXEL_SHAPES if s != shape for t in self.slam3d_tiles[s]]
        for s in VOXEL_SHAPES:
            if s != shape:
                self._blank_tiles(s, list(self.slam3d_tiles[s]))
        xyz, colors = voxel_down(xyz, rgb, size)
        tx = np.floor(xyz[:, 0] / SLAM3D_TILE).astype(np.int32)
        ty = np.floor(xyz[:, 1] / SLAM3D_TILE).astype(np.int32)
        order = np.lexsort((ty, tx))
        xyz, colors, tx, ty = xyz[order], colors[order], tx[order], ty[order]
        cuts = np.flatnonzero((np.diff(tx) != 0) | (np.diff(ty) != 0)) + 1
        tiles, h = self.slam3d_tiles[shape], size * 0.5
        seen, sent = set(), 0
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(xyz)]):
            tid = f"{tx[a]}_{ty[a]}"
            seen.add(tid)
            digest = hash((size, xyz[a:b].tobytes(), colors[a:b].tobytes()))
            if tiles.get(tid) == digest:
                continue
            tiles[tid] = digest
            if shape == "square":
                vis = rr.Boxes3D(centers=xyz[a:b], half_sizes=[[h, h, h]], colors=colors[a:b], fill_mode="solid")
            else:
                vis = rr.Points3D(xyz[a:b], radii=h, colors=colors[a:b])
            self._log(f"world/slam3d/{shape}/{tid}", rr.CoordinateFrame(frame), vis, static=True)
            sent += b - a
        self._blank_tiles(shape, [t for t in tiles if t not in seen])
        self.get_logger().info(f"voxel map: {len(xyz)} {shape} voxels of {size * 100:.0f} cm in {len(seen)} tiles, "
                               f"sent {sent} ({(time.monotonic() - t0) * 1000:.0f} ms)"
                               + (f", blanked {len(other)} old tiles" if other else ""))

    def _on_nvblox_occupancy(self, msg: OccupancyGrid) -> None:
        """nvblox' own 2D occupancy (from the depth camera), drawn just above the floor"""
        if not self.thr_nvblox["occupancy"]():
            return
        self._set_time(_stamp(msg))
        self._log("world/nvblox/occupancy", rr.CoordinateFrame(msg.header.frame_id),
               self._grid(msg, 0.6, rr.components.Colormap.RvizMap, 0.03))

    # ---- setup -----------------------------------------------------------------
    def _robot_parts(self) -> dict[str, list[str]]:
        """link -> GLB files (one entity per file under world/robot/<link>/part<k>)."""
        try:
            with open(os.path.join(self.mesh_dir, "index.json")) as f:
                index = json.load(f)
        except OSError:
            return {}
        return {link: info.get("files") or [info["file"]] for link, info in index.items()}

    @staticmethod
    def _blueprint(parts: dict[str, list[str]]):
        # The robot is many mesh entities (one per link chunk). Make them non-pickable and let the
        # transparent hit box on world/robot take the click: selecting world/robot highlights the
        # whole robot as one body instead of a single link.
        solid = rrb.EntityBehavior(interactive=False)
        overrides = {f"world/robot/{link}/part{k}": solid for link, files in parts.items() for k in range(len(files))}
        overrides["world/robot/base_axes"] = solid
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="World", origin="/",
                                  contents=["+ /**", "- /cams/**", "- /stats/**"],
                                  overrides=overrides,
                                  eye_controls=rrb.EyeControls3D(position=[-1.9, -1.5, 1.7],
                                                                 look_target=[0.3, 0.0, 0.5],
                                                                 eye_up=[0.0, 0.0, 1.0])),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Orbbec (jaw)", origin="/cams/orbbec"),
                    rrb.Spatial2DView(name="Bellybutton", origin="/cams/bellybutton"),
                    row_shares=[1, 1]),
                column_shares=[3, 1]),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
        )

    def _log_robot_model(self) -> None:
        if not self.robot_parts:
            self.get_logger().warn("no robot meshes installed")
            return
        for link, files in self.robot_parts.items():
            for k, fname in enumerate(files):
                # one GLB per message, each well under a few MB (see tools/convert_meshes.py --chunk)
                self._log(f"world/robot/{link}/part{k}", rr.Asset3D(path=os.path.join(self.mesh_dir, fname)),
                       rr.CoordinateFrame(link), static=True)
        # invisible pick target for the whole robot (floor to ~1.3 m around the base, axle frame)
        self._log("world/robot", rr.CoordinateFrame("base_footprint"),
               rr.Capsules3D(lengths=[0.72], radii=[0.3], translations=[[-0.15, 0.0, 0.3]],
                             colors=[[255, 255, 255, 0]], fill_mode="solid"), static=True)
        self._log("world/robot/base_axes", rr.TransformAxes3D(axis_length=0.3), rr.CoordinateFrame("base_footprint"),
               static=True)

    # ---- callbacks -----------------------------------------------------------------
    def _on_tf(self, msg: TFMessage, static: bool) -> None:
        if not static:
            self.chain.update(msg)
        for t in msg.transforms:
            child = t.child_frame_id.lstrip("/")
            thr = self.thr_tf if child in ("odom", "odom_wheel", "base_footprint") else self.thr_joint_tf
            if not static and not thr(child):
                continue
            tr, q = t.transform.translation, t.transform.rotation
            if not static:
                self._set_time(_stamp(t))
            self._log(f"tf/{child}", rr.Transform3D(translation=[tr.x, tr.y, tr.z], quaternion=[q.x, q.y, q.z, q.w],
                                                 parent_frame=t.header.frame_id.lstrip("/"), child_frame=child),
                   static=static)

    def _grid(self, msg: OccupancyGrid, opacity: float, cmap, z: float):
        info = msg.info
        # ROS row 0 is y-min; Rerun's GridMap row 0 is the image top (y-max) -> flip rows
        data = np.asarray(msg.data, dtype=np.int8).astype(np.uint8).reshape(info.height, info.width)[::-1]
        o, q = info.origin.position, info.origin.orientation
        return rr.GridMap(data=np.ascontiguousarray(data).tobytes(),
                          format=rr.components.ImageFormat(width=info.width, height=info.height,
                                                           color_model="L", channel_datatype="U8"),
                          cell_size=info.resolution, translation=[o.x, o.y, o.z + z],
                          quaternion=[q.x, q.y, q.z, q.w], opacity=opacity, colormap=cmap)

    @staticmethod
    def _map_mesh(msg: OccupancyGrid):
        """Occupancy grid as a textured floor quad. Unlike GridMap, a mesh is pickable, so a click on
        the floor in the viewer reports the exact (x, y) - that is what click-to-waypoint relies on.
        Cropped to the explored area + 1 m; RViz-like opaque colours read on dark and light themes."""
        info = msg.info
        h, w, res = info.height, info.width, info.resolution
        d = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        known = d >= 0
        if not known.any():
            return None
        rows, cols = np.where(known.any(1))[0], np.where(known.any(0))[0]
        m = int(round(1.0 / res))
        r0, r1 = max(rows[0] - m, 0), min(rows[-1] + m + 1, h)
        c0, c1 = max(cols[0] - m, 0), min(cols[-1] + m + 1, w)
        sub = d[r0:r1, c0:c1]
        occ = sub.clip(0, 100).astype(np.float32) / 100.0
        free_rgb, occ_rgb, unk_rgb = np.array([214, 218, 224]), np.array([28, 30, 36]), np.array([86, 92, 104])
        rgb = (free_rgb[None, None] * (1 - occ[..., None]) + occ_rgb[None, None] * occ[..., None])
        rgb[sub < 0] = unk_rgb
        tex = np.ascontiguousarray(rgb[::-1].astype(np.uint8))      # texture row 0 = y-max
        o, q = info.origin.position, info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        cy, sy = math.cos(yaw), math.sin(yaw)
        local = [(c0 * res, r0 * res), (c1 * res, r0 * res), (c1 * res, r1 * res), (c0 * res, r1 * res)]
        verts = [[o.x + cy * x - sy * y, o.y + sy * x + cy * y, o.z] for x, y in local]
        return rr.Mesh3D(vertex_positions=verts, triangle_indices=[[0, 1, 2], [0, 2, 3]],
                         vertex_normals=[[0.0, 0.0, 1.0]] * 4,
                         vertex_texcoords=[[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
                         albedo_texture=tex)

    def _on_map(self, msg: OccupancyGrid) -> None:
        self._set_time(_stamp(msg))
        mesh = self._map_mesh(msg)
        if mesh is not None:
            self._log("world/map", rr.CoordinateFrame(msg.header.frame_id or "map"), mesh)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        if not self.thr_costmap():
            return
        self._set_time(_stamp(msg))
        self._log("world/local_costmap", rr.CoordinateFrame(msg.header.frame_id),
               self._grid(msg, 0.45, rr.components.Colormap.RvizCostmap, 0.01))

    def _on_scan(self, msg: LaserScan) -> None:
        if not self.thr_scan():
            return
        r = np.asarray(msg.ranges, np.float32)
        a = msg.angle_min + np.arange(len(r), dtype=np.float32) * msg.angle_increment
        ok = np.isfinite(r) & (r > max(msg.range_min, 0.12)) & (r < min(msg.range_max, 25.0))
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum(), np.float32)], 1)
        self._set_time(_stamp(msg))
        self._log("world/scan", rr.CoordinateFrame(msg.header.frame_id), rr.Points3D(pts, radii=0.02, colors=[255, 60, 60]))

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.thr_cloud():
            return
        xyz = cloud_xyz(msg)
        if len(xyz) == 0:
            return
        d = np.linalg.norm(xyz, axis=1)
        t = np.clip((d - 0.3) / 3.0, 0, 1)
        colors = (np.stack([t, 0.6 + 0.4 * (1 - t), 1 - t], 1) * 255).astype(np.uint8)
        self._set_time(_stamp(msg))
        self._log("world/depth_cloud", rr.CoordinateFrame(msg.header.frame_id),
               rr.Points3D(xyz, radii=0.015, colors=colors))

    def _on_mesh_tile(self, msg: MeshTile) -> None:
        path = f"world/nvblox/mesh/{msg.id}"
        self.mesh_tiles.add(msg.id)
        tri = np.frombuffer(msg.triangles, dtype=np.uint32)
        if len(tri) < 3:
            self._log(path, rr.Mesh3D(vertex_positions=np.zeros((0, 3), np.float32)), static=True)
            return
        pos = np.frombuffer(msg.positions, dtype=np.float32).reshape(-1, 3)
        nrm = np.frombuffer(msg.normals, dtype=np.float32).reshape(-1, 3) if len(msg.normals) else None
        if len(msg.colors) == pos.size:
            rgb = np.frombuffer(msg.colors, dtype=np.uint8).reshape(-1, 3)
        else:  # no colour integrated yet: shade by height
            t = np.clip(pos[:, 2] / 1.6, 0, 1)[:, None]
            rgb = (np.array([70, 110, 200]) * (1 - t) + np.array([230, 200, 120]) * t).astype(np.uint8)
        # static: only the latest version of each tile is kept, so the memory-limited stream never
        # garbage-collects parts of the map that have not changed for a while
        self._log(path, rr.CoordinateFrame(msg.header.frame_id),
               rr.Mesh3D(vertex_positions=pos, vertex_normals=nrm, vertex_colors=rgb,
                         triangle_indices=tri.reshape(-1, 3)), static=True)

    def _on_image(self, msg: CompressedImage, name: str) -> None:
        if not self.thr_img(name):
            return
        jpg = bytes(msg.data)
        self._set_time(_stamp(msg))
        self._log(f"cams/{name}", rr.EncodedImage(contents=jpg, media_type="image/jpeg"))
        info = self.cam_info.get(name)
        if info is not None and self.show_camera and self.thr_frustum(name):
            # 3D frustum with the live image at 2 Hz (preview keeps the camera_info resolution)
            frame = info.header.frame_id or msg.header.frame_id
            K = np.array(info.k, dtype=np.float64).reshape(3, 3)
            w, h = (info.width or 640), (info.height or 480)
            if K[0, 0] <= 0:   # uncalibrated: pinhole from an assumed horizontal FOV
                f = (w / 2) / math.tan(self.belly_hfov / 2)
                K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            self._log(f"world/{name}_camera",
                   rr.Pinhole(image_from_camera=K, resolution=[w, h],
                              camera_xyz=rr.ViewCoordinates.RDF, parent_frame=frame, child_frame=f"{name}_image",
                              image_plane_distance=0.4),
                   rr.EncodedImage(contents=jpg, media_type="image/jpeg"),
                   rr.CoordinateFrame(f"{name}_image"))

    def _on_range(self, msg: Range, name: str) -> None:
        if not self.thr_us(name):
            return
        rng = msg.range if math.isfinite(msg.range) else msg.max_range
        color = [255, 170, 0] if math.isfinite(msg.range) else [90, 90, 90]
        self._set_time(_stamp(msg))
        self._log(f"world/ultrasonic/{name}", rr.CoordinateFrame(msg.header.frame_id),
               rr.Arrows3D(vectors=[[rng, 0, 0]], colors=[color], radii=0.01,
                           labels=[f"{rng:.2f} m" if math.isfinite(msg.range) else ""]))
        self._set_time(_stamp(msg))
        self._log(f"stats/ultrasonic/{name}", rr.Scalars(rng))

    def _on_plan(self, msg: Path) -> None:
        pts = [[p.pose.position.x, p.pose.position.y, 0.03] for p in msg.poses]
        self._set_time(_stamp(msg))
        self._log("world/plan", rr.CoordinateFrame(msg.header.frame_id or "map"),
               rr.LineStrips3D([pts] if pts else [], colors=[[60, 200, 255]], radii=0.02))

    def _on_vslam_path(self, msg: Path) -> None:
        if not self.thr_slow("vslam"):
            return
        pts = [[p.pose.position.x, p.pose.position.y, 0.05] for p in msg.poses]
        self._set_time(_stamp(msg))
        self._log("world/vslam_path", rr.CoordinateFrame(msg.header.frame_id or "vslam_odom"),
               rr.LineStrips3D([pts] if pts else [], colors=[[200, 90, 255]], radii=0.012))

    def _on_waypoints(self, msg: PoseArray) -> None:
        pos, vec, labels = [], [], []
        for i, p in enumerate(msg.poses):
            q = p.orientation
            yaw = 2 * math.atan2(q.z, q.w)
            pos.append([p.position.x, p.position.y, 0.05])
            vec.append([0.35 * math.cos(yaw), 0.35 * math.sin(yaw), 0.0])
            labels.append(str(i + 1))
        self._set_time(time.time())   # newest on the timeline: otherwise an older stamp can hide a deletion
        self._log("world/waypoints", rr.CoordinateFrame(msg.header.frame_id or "map"),
               rr.Arrows3D(origins=pos, vectors=vec, colors=[[255, 220, 0]] * len(pos), radii=0.03, labels=labels))

    def _on_footprint(self, msg: PolygonStamped) -> None:
        pts = [[p.x, p.y, 0.02] for p in msg.polygon.points]
        if pts:
            pts.append(pts[0])
        self._set_time(time.time())
        self._log("world/footprint", rr.CoordinateFrame(msg.header.frame_id),
               rr.LineStrips3D([pts], colors=[[0, 255, 140]], radii=0.01))

    def _trail_tick(self) -> None:
        pose = self.chain.pose()
        if pose is None or pose[0] != "map":
            return
        p = [pose[1], pose[2], 0.02]
        if not self.trail or math.dist(p, self.trail[-1]) > 0.03:
            self.trail = (self.trail + [p])[-3000:]
            self._set_time(time.time())
            self._log("world/trail", rr.CoordinateFrame("map"),
                   rr.LineStrips3D([self.trail], colors=[[255, 255, 255, 120]], radii=0.008))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RerunBridge()
    ex = rclpy.executors.SingleThreadedExecutor()   # rclpy multi-threaded spinning is CPU-heavy
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
