"""ROS 2 -> Rerun bridge for the Mercury X1.

Mirrors /tf into Rerun named frames (Transform3D parent_frame/child_frame) and logs every
data entity with CoordinateFrame(<ros frame_id>), so the viewer resolves poses the way tf2
does. Serves the data over gRPC (default :9876) and a stand-alone web viewer (:9090); the
control page served by web_server embeds the same stream.
"""

from __future__ import annotations

import json
import math
import os
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
from sensor_msgs.msg import BatteryState, CameraInfo, CompressedImage, LaserScan, PointCloud2, Range
from tf2_msgs.msg import TFMessage

from .pose_chain import PoseChain

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)
# /tf_static has several latched publishers (robot_state_publisher, orbbec driver, ...): with
# depth 1 one sample would overwrite the other, so keep plenty like tf2's StaticListenerQoS.
TF_STATIC = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE)


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


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    """Fast x/y/z extraction from a float32 PointCloud2."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in "xyz"):
        return np.zeros((0, 3), np.float32)
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    xyz = np.stack([buf[:, offs[k]:offs[k] + 4].copy().view(np.float32)[:, 0] for k in "xyz"], 1)
    return xyz[np.isfinite(xyz).all(1)]


class RerunBridge(Node):
    def __init__(self) -> None:
        super().__init__("rerun_bridge")
        p = self.declare_parameter
        grpc_port = int(p("grpc_port", 9876).value)
        web_port = int(p("web_port", 9090).value)
        mem = p("memory_limit", "150MiB").value   # replayed to every newly connected viewer
        self.img_hz = float(p("image_rate", 4.0).value)
        self.cloud_hz = float(p("cloud_rate", 3.0).value)
        self.scan_hz = float(p("scan_rate", 10.0).value)
        # the bellybutton camera has no calibration file (camera_info K is zero): approximate it
        self.belly_hfov = math.radians(float(p("bellybutton_hfov_deg", 75.0).value))
        self.tf_hz = float(p("tf_rate", 10.0).value)            # odometry chain
        self.joint_tf_hz = float(p("joint_tf_rate", 5.0).value)  # robot links (joint states)

        rr.init("mercury_x1", recording_id="mercury_x1_live")
        self.mesh_dir = os.path.join(get_package_share_directory("mercury_x1_web"), "meshes")
        self.robot_parts = self._robot_parts()
        uri = rr.serve_grpc(grpc_port=grpc_port, server_memory_limit=mem,
                            cors_allow_origin=["*"], default_blueprint=self._blueprint(self.robot_parts))
        rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=uri)
        self.get_logger().info(f"rerun gRPC {uri}, viewer http://<robot>:{web_port}")

        # the 3D view's root: tie the ROS "map" frame to the view origin
        rr.log("tf/map", rr.Transform3D(parent_frame="tf#/", child_frame="map"), static=True)
        self._log_robot_model()

        self.thr_tf = Throttle(self.tf_hz)
        self.thr_joint_tf = Throttle(self.joint_tf_hz)
        self.thr_img = Throttle(self.img_hz)
        self.thr_frustum = Throttle(1.0)
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
        # nvblox GPU reconstruction, pre-tiled in C++ (mercury_x1_perception/nvblox_mesh_tiler)
        sub(MeshTile, "/nvblox/mesh_tiles", self._on_mesh_tile,
            QoSProfile(depth=64, reliability=ReliabilityPolicy.RELIABLE))
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
        sub(BatteryState, "/battery", self._on_battery, 10)
        self.chain = PoseChain()
        self.create_timer(0.5, self._trail_tick)

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
                    rrb.TextDocumentView(name="Battery", origin="/stats/battery_text"),
                    row_shares=[4, 4, 1]),
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
                rr.log(f"world/robot/{link}/part{k}", rr.Asset3D(path=os.path.join(self.mesh_dir, fname)),
                       rr.CoordinateFrame(link), static=True)
        # invisible pick target for the whole robot (floor to ~1.3 m around the base, axle frame)
        rr.log("world/robot", rr.CoordinateFrame("base_footprint"),
               rr.Capsules3D(lengths=[0.72], radii=[0.3], translations=[[-0.15, 0.0, 0.3]],
                             colors=[[255, 255, 255, 0]]), static=True)
        rr.log("world/robot/base_axes", rr.TransformAxes3D(axis_length=0.3), rr.CoordinateFrame("base_footprint"),
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
                rr.set_time("ros", timestamp=_stamp(t))
            rr.log(f"tf/{child}", rr.Transform3D(translation=[tr.x, tr.y, tr.z], quaternion=[q.x, q.y, q.z, q.w],
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
        rr.set_time("ros", timestamp=_stamp(msg))
        mesh = self._map_mesh(msg)
        if mesh is not None:
            rr.log("world/map", rr.CoordinateFrame(msg.header.frame_id or "map"), mesh)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        if not self.thr_costmap():
            return
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log("world/local_costmap", rr.CoordinateFrame(msg.header.frame_id),
               self._grid(msg, 0.45, rr.components.Colormap.RvizCostmap, 0.01))

    def _on_scan(self, msg: LaserScan) -> None:
        if not self.thr_scan():
            return
        r = np.asarray(msg.ranges, np.float32)
        a = msg.angle_min + np.arange(len(r), dtype=np.float32) * msg.angle_increment
        ok = np.isfinite(r) & (r > max(msg.range_min, 0.12)) & (r < min(msg.range_max, 25.0))
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum(), np.float32)], 1)
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log("world/scan", rr.CoordinateFrame(msg.header.frame_id), rr.Points3D(pts, radii=0.02, colors=[255, 60, 60]))

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.thr_cloud():
            return
        xyz = cloud_xyz(msg)
        if len(xyz) == 0:
            return
        d = np.linalg.norm(xyz, axis=1)
        t = np.clip((d - 0.3) / 3.0, 0, 1)
        colors = (np.stack([t, 0.6 + 0.4 * (1 - t), 1 - t], 1) * 255).astype(np.uint8)
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log("world/depth_cloud", rr.CoordinateFrame(msg.header.frame_id),
               rr.Points3D(xyz, radii=0.015, colors=colors))

    def _on_mesh_tile(self, msg: MeshTile) -> None:
        path = f"world/nvblox/{msg.id}"
        tri = np.frombuffer(msg.triangles, dtype=np.uint32)
        if len(tri) < 3:
            rr.log(path, rr.Mesh3D(vertex_positions=np.zeros((0, 3), np.float32)), static=True)
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
        rr.log(path, rr.CoordinateFrame(msg.header.frame_id),
               rr.Mesh3D(vertex_positions=pos, vertex_normals=nrm, vertex_colors=rgb,
                         triangle_indices=tri.reshape(-1, 3)), static=True)

    def _on_image(self, msg: CompressedImage, name: str) -> None:
        if not self.thr_img(name):
            return
        jpg = bytes(msg.data)
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log(f"cams/{name}", rr.EncodedImage(contents=jpg, media_type="image/jpeg"))
        info = self.cam_info.get(name)
        if info is not None and self.thr_frustum(name):
            # 3D frustum with the live image at 1 Hz (preview keeps the camera_info resolution)
            frame = info.header.frame_id or msg.header.frame_id
            K = np.array(info.k, dtype=np.float64).reshape(3, 3)
            w, h = (info.width or 640), (info.height or 480)
            if K[0, 0] <= 0:   # uncalibrated: pinhole from an assumed horizontal FOV
                f = (w / 2) / math.tan(self.belly_hfov / 2)
                K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
            rr.log(f"world/{name}_camera",
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
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log(f"world/ultrasonic/{name}", rr.CoordinateFrame(msg.header.frame_id),
               rr.Arrows3D(vectors=[[rng, 0, 0]], colors=[color], radii=0.01,
                           labels=[f"{rng:.2f} m" if math.isfinite(msg.range) else ""]))
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log(f"stats/ultrasonic/{name}", rr.Scalars(rng))

    def _on_plan(self, msg: Path) -> None:
        pts = [[p.pose.position.x, p.pose.position.y, 0.03] for p in msg.poses]
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log("world/plan", rr.CoordinateFrame(msg.header.frame_id or "map"),
               rr.LineStrips3D([pts] if pts else [], colors=[[60, 200, 255]], radii=0.02))

    def _on_vslam_path(self, msg: Path) -> None:
        if not self.thr_slow("vslam"):
            return
        pts = [[p.pose.position.x, p.pose.position.y, 0.05] for p in msg.poses]
        rr.set_time("ros", timestamp=_stamp(msg))
        rr.log("world/vslam_path", rr.CoordinateFrame(msg.header.frame_id or "vslam_odom"),
               rr.LineStrips3D([pts] if pts else [], colors=[[200, 90, 255]], radii=0.012))

    def _on_waypoints(self, msg: PoseArray) -> None:
        pos, vec, labels = [], [], []
        for i, p in enumerate(msg.poses):
            q = p.orientation
            yaw = 2 * math.atan2(q.z, q.w)
            pos.append([p.position.x, p.position.y, 0.05])
            vec.append([0.35 * math.cos(yaw), 0.35 * math.sin(yaw), 0.0])
            labels.append(str(i + 1))
        rr.log("world/waypoints", rr.CoordinateFrame(msg.header.frame_id or "map"),
               rr.Arrows3D(origins=pos, vectors=vec, colors=[[255, 220, 0]] * len(pos), radii=0.03, labels=labels))

    def _on_footprint(self, msg: PolygonStamped) -> None:
        pts = [[p.x, p.y, 0.02] for p in msg.polygon.points]
        if pts:
            pts.append(pts[0])
        rr.log("world/footprint", rr.CoordinateFrame(msg.header.frame_id),
               rr.LineStrips3D([pts], colors=[[0, 255, 140]], radii=0.01))

    def _on_battery(self, msg: BatteryState) -> None:
        rr.set_time("ros", timestamp=_stamp(msg))
        pct = msg.percentage * 100 if math.isfinite(msg.percentage) else float("nan")
        text = f"## 🔋 {pct:.0f} %  \n{msg.voltage:.1f} V" if math.isfinite(pct) else f"## 🔋 {msg.voltage:.1f} V"
        rr.log("stats/battery_text", rr.TextDocument(text, media_type=rr.MediaType.MARKDOWN))
        rr.log("stats/battery", rr.Scalars(msg.voltage))

    def _trail_tick(self) -> None:
        pose = self.chain.pose()
        if pose is None or pose[0] != "map":
            return
        p = [pose[1], pose[2], 0.02]
        if not self.trail or math.dist(p, self.trail[-1]) > 0.03:
            self.trail = (self.trail + [p])[-3000:]
            rr.log("world/trail", rr.CoordinateFrame("map"),
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
