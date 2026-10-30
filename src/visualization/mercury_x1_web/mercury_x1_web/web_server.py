"""Mercury X1 control page: Rerun viewer + map / waypoints / teleop / SLAM & Nav2 supervisor.

  http://<robot>:8080

The page embeds the Rerun web viewer (stream from rerun_bridge on :9876) next to a control
panel. This process runs an rclpy node in a background thread and an aiohttp server on the
main asyncio loop; ROS actions are executed on the ROS thread through a small job queue.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
import urllib.request

import numpy as np
import rclpy
import rclpy.executors
from aiohttp import web
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from mercury_x1_msgs.msg import ArmState, BaseState, JointCommand
from nav2_msgs.action import FollowWaypoints, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from PIL import Image as PILImage
from rclpy.action import ActionClient
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.srv import GetMap
from sensor_msgs.msg import BatteryState, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

from .pose_chain import PoseChain
from . import slam3d
from .sysstats import SysStats
from .wifi import Wifi
from std_msgs.msg import Bool, ColorRGBA, String
from std_srvs.srv import SetBool, Trigger

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
MAPS_DIR = os.path.expanduser("~/maps")
VIEWER_VERSION = "0.39.0"   # must match rerun-sdk on the robot (newest release that still supports Python 3.10)
VIEWER_FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm")
MX1_WS = os.environ.get("MX1_WS", os.path.expanduser("~/mercury_x1_ros2"))
CONF_DIR = os.path.expanduser("~/.config/mercury_x1")
ROS_NETWORK_FILE = os.path.join(CONF_DIR, "ros_network")   # read by scripts/ros_env.sh at stack start
VIZ_LAYERS_FILE = os.path.join(CONF_DIR, "viz_layers.json")  # read by rerun_bridge at start
# = rerun_bridge.NVBLOX_LAYERS + "camera" (camera frustums with the live image in the 3D view)
NVBLOX_LAYERS = ("mesh", "esdf", "depth", "tsdf", "color", "occupancy", "slam3d", "camera")
DEFAULT_VIZ_LAYERS = ["mesh", "depth", "camera"]
TRASH_DIR = os.path.join(MAPS_DIR, ".trash")   # deleted maps, purged after TRASH_DAYS
TRASH_DAYS = 7
LIVE3D_VOXEL = 0.05         # [m] voxel size of the whole-map 3D layer
LIVE3D_PERIOD = 30.0        # [s] at most one re-export per this while the map grows (mapping)
WAYPOINT_PICK_RADIUS = 0.6   # [m] a "delete" click removes the nearest waypoint within this distance
# head pose while mapping: neck straight, Orbbec level (its pitch = -jaw - 5.7 deg, see the URDF notes)
MAPPING_HEAD = {"head": 0.0, "eye": -5.7}
VIEWER_CACHE = os.path.expanduser(f"~/.cache/mercury_x1_web/rerun-web-viewer/{VIEWER_VERSION}")


def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def pose_stamped(x: float, y: float, yaw: float, frame: str, stamp) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = frame
    p.header.stamp = stamp
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    _, _, p.pose.orientation.z, p.pose.orientation.w = yaw_to_quat(float(yaw))
    return p


def cloud_msg(xyz: np.ndarray, rgb: np.ndarray | None, frame: str, stamp) -> PointCloud2:
    """x, y, z + packed rgb (the usual ROS layout) PointCloud2"""
    pts = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    pts["x"], pts["y"], pts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        pts["rgb"] = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]
    msg = PointCloud2(height=1, width=len(xyz), is_bigendian=False, point_step=16, row_step=16 * len(xyz),
                      is_dense=True, data=pts.tobytes())
    msg.header.frame_id, msg.header.stamp = frame, stamp
    msg.fields = [PointField(name=k, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, k in enumerate(("x", "y", "z", "rgb"))]
    return msg


# ---------------------------------------------------------------------------------------------
class Supervisor:
    """Starts/stops `ros2 launch` children (SLAM, Nav2) in their own process groups."""

    def __init__(self, log_dir: str):
        self.procs: dict[str, subprocess.Popen] = {}
        self.args: dict[str, dict] = {}
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.lock = threading.Lock()

    def running(self, name: str) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def start(self, name: str, cmd: list[str], meta: dict) -> None:
        self.stop(name)
        with self.lock:
            log = open(os.path.join(self.log_dir, f"{name}.log"), "w")
            self.procs[name] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
            self.args[name] = meta

    def stop(self, name: str, timeout: float = 25.0) -> None:
        with self.lock:
            p = self.procs.pop(name, None)
            self.args.pop(name, None)
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGINT)   # rtabmap saves its database on SIGINT
            p.wait(timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def status(self) -> dict:
        return {n: {"running": self.running(n), **self.args.get(n, {})} for n in ("slam", "nav")}


# ---------------------------------------------------------------------------------------------
class WebNode(Node):
    def __init__(self) -> None:
        super().__init__("mercury_web")
        self.jobs: queue.Queue = queue.Queue()
        self.map_msg: OccupancyGrid | None = None
        self.map_version = 0
        self.map_png: bytes | None = None
        self.plan: list[list[float]] = []
        self.battery = float("nan")
        self.battery_pct = float("nan")
        self.charge = "unknown"
        self.base: BaseState | None = None
        self.arms: dict[str, ArmState] = {}
        self.waypoints: list[dict] = []
        self.nav = {"state": "idle", "goal": None, "current": None, "loop": False, "message": ""}
        self.wp_handle = None
        self.goal_handle = None
        self.cancel_requested = False
        # what a click in the 3D view (embedded Rerun viewer) does
        self.click_mode = "waypoint"          # waypoint | delete | goto | pose | off
        self.last_click: dict | None = None
        self.waypoints_changed = None         # set by App: persist + republish
        self.remote: dict | None = None       # Unitree BLE remote (mercury_x1_teleop/unitree_remote)

        self.chain = PoseChain()
        self.create_subscription(TFMessage, "/tf", self.chain.update, 50)

        self.create_subscription(OccupancyGrid, "/map", self._on_map, LATCHED)
        self.create_subscription(Path, "/plan", self._on_plan, 1)
        self.create_subscription(BatteryState, "/battery", self._on_battery, 10)
        self.create_subscription(BaseState, "/base/state", lambda m: setattr(self, "base", m), 10)
        for s in ("left", "right"):
            self.create_subscription(ArmState, f"/{s}_arm/state", lambda m, s=s: self.arms.__setitem__(s, m), 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)
        self.pub_wps = self.create_publisher(PoseArray, "/web/waypoints", LATCHED)
        self.pub_led = self.create_publisher(ColorRGBA, "/base/led", 10)
        self.pub_joint = self.create_publisher(JointCommand, "/upper_body/joint_command", 10)
        self.pub_remote = self.create_publisher(String, "/remote/cmd", 10)
        # whole map as a coloured 3D cloud (App._live3d_loop), built only while someone subscribes
        self.pub_cloud3d = self.create_publisher(PointCloud2, "/slam/cloud3d", LATCHED)
        self.cli_map_data = self.create_client(GetMap, "/rtabmap/get_map_data")
        # The physical E-stop cuts the base MCU's power (its USB serial device disappears) and the arm
        # servos' power (the arm controllers report powered=false, and their link may drop out too);
        # nothing reports the button itself. Base link lost while no arm reports power = E-stop.
        self.estop = False
        self.pub_estop = self.create_publisher(Bool, "/estop", LATCHED)
        self.pub_estop.publish(Bool(data=False))
        self.create_timer(0.5, self._check_estop)
        self.create_subscription(String, "/remote/status", self._on_remote, LATCHED)
        self.create_subscription(String, "/remote/button", self._on_remote_button, 10)
        self.ac_wps = ActionClient(self, FollowWaypoints, "/follow_waypoints")
        self.ac_goal = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self.srv = {
            "left_power": self.create_client(SetBool, "/left_arm/power"),
            "right_power": self.create_client(SetBool, "/right_arm/power"),
            "arms_stop": self.create_client(Trigger, "/upper_body/stop"),
            "freedrive": self.create_client(SetBool, "/upper_body/freedrive"),
            "bridge_params": self.create_client(SetParameters, "/rerun_bridge/set_parameters"),
        }
        self.create_timer(0.05, self._run_jobs)
        self.node_names: list[str] = []
        self.create_timer(2.0, self._refresh_nodes)

    def _refresh_nodes(self) -> None:
        """running ROS nodes for the head display (helper nodes of tf listeners / launch filtered out)"""
        names = set()
        for name, ns in self.get_node_names_and_namespaces():
            if name.startswith(("transform_listener_impl", "launch_ros_", "_ros2cli")) or name.endswith("_rclcpp_node"):
                continue
            names.add(name)
        self.node_names = sorted(names)

    # -- run a callable on the ROS thread ------------------------------------------------------
    def post(self, fn) -> None:
        self.jobs.put(fn)

    def _run_jobs(self) -> None:
        while not self.jobs.empty():
            try:
                self.jobs.get_nowait()()
            except Exception as e:
                self.get_logger().error(f"job failed: {e}")

    # -- topics --------------------------------------------------------------------------------
    def _on_map(self, msg: OccupancyGrid) -> None:
        self.map_msg = msg
        w, h = msg.info.width, msg.info.height
        d = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        img = np.full((h, w), 205, np.uint8)                   # unknown
        known = d >= 0
        img[known] = (254 - (d[known].clip(0, 100) * 254 // 100)).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(img[::-1]).save(buf, format="PNG")  # row 0 = top in the PNG
        self.map_png = buf.getvalue()
        self.map_version += 1

    def _on_battery(self, msg: BatteryState) -> None:
        self.battery = msg.voltage
        self.battery_pct = msg.percentage * 100 if math.isfinite(msg.percentage) else float("nan")
        # estimated by base_node from the voltage trend (the MCU does not report it)
        self.charge = {BatteryState.POWER_SUPPLY_STATUS_CHARGING: "charging",
                       BatteryState.POWER_SUPPLY_STATUS_DISCHARGING: "discharging",
                       BatteryState.POWER_SUPPLY_STATUS_FULL: "full"}.get(msg.power_supply_status, "unknown")

    def _on_remote(self, msg: String) -> None:
        try:
            st = json.loads(msg.data)
        except ValueError:
            return
        was_driving = bool(self.remote and self.remote.get("driving"))
        self.remote = st
        if st.get("driving") and not was_driving and self.nav["state"] == "navigating":
            self.cancel(stop_base=False)        # driving with the remote overrides navigation

    def _on_remote_button(self, msg: String) -> None:
        if msg.data == "B":                     # B on the remote = STOP (cancel navigation, stop the base)
            self.cancel()
        elif msg.data == "Select":              # Select = freedrive on / off (both arms)
            self.set_freedrive(not self.freedrive_on())

    def freedrive_on(self) -> bool:
        return any(getattr(a, "freedrive", False) for a in self.arms.values())

    def set_freedrive(self, on: bool) -> None:
        self.get_logger().info(f"freedrive {'ON' if on else 'off'}")
        self.call("freedrive", SetBool.Request(data=on))

    def _on_plan(self, msg: Path) -> None:
        self.plan = [[round(p.pose.position.x, 3), round(p.pose.position.y, 3)] for p in msg.poses[::3]]

    def robot_pose(self):
        p = self.chain.pose()
        if p is None:
            return None
        return {"frame": p[0], "x": p[1], "y": p[2], "yaw": p[3]}

    # -- commands --------------------------------------------------------------------------------
    def publish_waypoints(self) -> None:
        pa = PoseArray()
        pa.header.frame_id = "map"
        pa.header.stamp = self.get_clock().now().to_msg()
        for w in self.waypoints:
            p = Pose()
            p.position.x, p.position.y = float(w["x"]), float(w["y"])
            _, _, p.orientation.z, p.orientation.w = yaw_to_quat(float(w["yaw"]))
            pa.poses.append(p)
        self.pub_wps.publish(pa)

    IGNORED_CLICK_PREFIXES = ("world/robot", "cams", "world/orbbec_camera", "world/bellybutton_camera")

    def handle_click(self, x: float, y: float, z: float, entity: str, source: str) -> None:
        """A point picked in a 3D view -> waypoint / goal / initial pose (runs on the ROS thread)."""
        entity = (entity or "").lstrip("/")
        result = ""
        if self.click_mode == "off":
            result = "ignored (view only)"
        elif entity.startswith(self.IGNORED_CLICK_PREFIXES):
            result = f"ignored (clicked {entity})"
        else:
            pose = self.robot_pose()
            here = (pose["x"], pose["y"]) if pose and pose["frame"] == "map" else None
            if self.click_mode == "waypoint":
                prev = (self.waypoints[-1]["x"], self.waypoints[-1]["y"]) if self.waypoints else here
                yaw = math.atan2(y - prev[1], x - prev[0]) if prev else 0.0
                self.waypoints.append({"x": float(x), "y": float(y), "yaw": yaw})
                if self.waypoints_changed:
                    self.waypoints_changed()
                result = f"waypoint {len(self.waypoints)}"
            elif self.click_mode == "delete":
                result = self.delete_waypoint_near(x, y)
            elif self.click_mode == "goto":
                yaw = math.atan2(y - here[1], x - here[0]) if here else 0.0
                self.go_to(x, y, yaw)
                result = "navigating"
            elif self.click_mode == "pose":
                self.set_initial_pose(x, y, pose["yaw"] if pose else 0.0)
                result = "initial pose set"
        self.last_click = {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2), "entity": entity,
                           "source": source, "result": result, "t": time.time()}

    def delete_waypoint_near(self, x: float, y: float) -> str:
        if not self.waypoints:
            return "no waypoints"
        k = min(range(len(self.waypoints)), key=lambda i: math.hypot(self.waypoints[i]["x"] - x, self.waypoints[i]["y"] - y))
        d = math.hypot(self.waypoints[k]["x"] - x, self.waypoints[k]["y"] - y)
        if d > WAYPOINT_PICK_RADIUS:
            return f"no waypoint within {WAYPOINT_PICK_RADIUS} m"
        del self.waypoints[k]
        if self.waypoints_changed:
            self.waypoints_changed()
        return f"deleted waypoint {k + 1}"

    def cmd_vel(self, vx: float, wz: float) -> None:
        t = Twist()
        t.linear.x, t.angular.z = float(vx), float(wz)
        self.pub_cmd.publish(t)

    def set_initial_pose(self, x: float, y: float, yaw: float) -> None:
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose = pose_stamped(x, y, yaw, "map", m.header.stamp).pose
        m.pose.covariance[0] = m.pose.covariance[7] = 0.25
        m.pose.covariance[35] = 0.07
        self.pub_init.publish(m)

    def go_to(self, x: float, y: float, yaw: float) -> None:
        self.cancel(stop_base=False)
        if not self.ac_goal.wait_for_server(timeout_sec=1.0):
            self.nav.update(state="error", message="navigate_to_pose not available - start Nav2")
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose_stamped(x, y, yaw, "map", self.get_clock().now().to_msg())
        self.nav.update(state="navigating", goal={"x": x, "y": y, "yaw": yaw}, current=None, loop=False, message="")
        fut = self.ac_goal.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._accepted(f, "goal"))

    def run_waypoints(self, loop: bool) -> None:
        self.cancel(stop_base=False)
        if not self.waypoints:
            self.nav.update(state="error", message="no waypoints")
            return
        if not self.ac_wps.wait_for_server(timeout_sec=1.0):
            self.nav.update(state="error", message="follow_waypoints not available - start Nav2")
            return
        self.cancel_requested = False
        self.nav.update(state="navigating", goal=None, current=0, loop=loop, message="")
        self._send_waypoints()

    def _send_waypoints(self) -> None:
        goal = FollowWaypoints.Goal()
        stamp = self.get_clock().now().to_msg()
        goal.poses = [pose_stamped(w["x"], w["y"], w["yaw"], "map", stamp) for w in self.waypoints]
        fut = self.ac_wps.send_goal_async(goal, feedback_callback=self._wp_feedback)
        fut.add_done_callback(lambda f: self._accepted(f, "wps"))

    def _wp_feedback(self, fb) -> None:
        self.nav["current"] = int(fb.feedback.current_waypoint)

    def _accepted(self, fut, kind: str) -> None:
        h = fut.result()
        if not h.accepted:
            self.nav.update(state="error", message=f"{kind} rejected")
            return
        if kind == "wps":
            self.wp_handle = h
        else:
            self.goal_handle = h
        h.get_result_async().add_done_callback(lambda f: self._done(f, kind, h))

    def _done(self, fut, kind: str, handle) -> None:
        if handle is not (self.wp_handle if kind == "wps" else self.goal_handle):
            return  # result of a goal we already cancelled/replaced
        res = fut.result()
        ok = res.status == GoalStatus.STATUS_SUCCEEDED
        if kind == "wps":
            self.wp_handle = None
            missed = list(res.result.missed_waypoints) if ok else []
            if self.nav.get("loop") and not self.cancel_requested and res.status != GoalStatus.STATUS_CANCELED:
                self.nav["message"] = f"lap done (missed {missed})" if missed else "lap done"
                self._send_waypoints()
                return
            msg = f"missed waypoints {missed}" if missed else ""
        else:
            self.goal_handle = None
            msg = ""
        state = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_CANCELED: "canceled",
                 GoalStatus.STATUS_ABORTED: "aborted"}.get(res.status, "idle")
        self.nav.update(state=state, message=msg)

    def cancel(self, stop_base: bool = True) -> None:
        self.cancel_requested = True
        for h in (self.wp_handle, self.goal_handle):
            if h is not None:
                h.cancel_goal_async()
        self.wp_handle = self.goal_handle = None
        if self.nav["state"] == "navigating":
            self.nav.update(state="canceled")
        if stop_base:
            self.cmd_vel(0.0, 0.0)

    def set_nvblox_layers(self, layers: list[str]) -> None:
        val = ParameterValue(type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=layers or [""])
        self.call("bridge_params", SetParameters.Request(parameters=[Parameter(name="nvblox_layers", value=val)]))

    def live_graph(self, timeout: float = 15.0) -> dict[int, tuple[float, float, float]] | None:
        """node id -> optimized position in the map frame, from rtabmap (graph only, no sensor data).
        Called from a worker thread; the executor thread completes the future."""
        if not self.cli_map_data.wait_for_service(timeout_sec=2.0):
            return None
        fut = self.cli_map_data.call_async(GetMap.Request(global_map=True, optimized=True, graph_only=True))
        done = threading.Event()
        fut.add_done_callback(lambda _: done.set())
        if not done.wait(timeout) or fut.result() is None:
            return None
        g = fut.result().data.graph
        return {int(i): (p.position.x, p.position.y, p.position.z) for i, p in zip(g.poses_id, g.poses)}

    def call(self, key: str, req) -> None:
        cli = self.srv[key]
        if cli.wait_for_service(timeout_sec=0.5):
            cli.call_async(req)

    def _check_estop(self) -> None:
        base_up = self.base is not None and self.base.connected
        arms = list(self.arms.values())
        estop = self.base is not None and not base_up and bool(arms) and not any(a.powered for a in arms)
        if estop == self.estop:
            return
        self.estop = estop
        self.pub_estop.publish(Bool(data=estop))
        if estop:
            self.get_logger().warn("E-STOP: base MCU off and arm power cut")
            self.cancel(stop_base=False)        # the base is unpowered; drop goals so nothing resumes on release
        else:
            self.get_logger().info("E-stop released (base MCU back or arms powered)")

    def status(self) -> dict:
        base = self.base
        arms = {s: {"powered": a.powered, "connected": a.connected, "moving": a.moving,
                    "freedrive": bool(getattr(a, "freedrive", False)),
                    "rate": round(a.state_rate_hz, 1)} for s, a in self.arms.items()}
        return {
            "pose": self.robot_pose(),
            "battery": None if math.isnan(self.battery) else round(self.battery, 2),
            "battery_pct": None if math.isnan(self.battery_pct) else round(self.battery_pct),
            "charge": self.charge,
            "base": None if base is None else {
                "connected": base.connected, "fw": base.firmware_version,
                "ultrasonic": {n: round(v, 2) for n, v in zip(base.ultrasonic_names, base.ultrasonic_range) if n},
                "obstacle_hold": base.obstacle_hold, "cmd_timeout": base.cmd_timeout,
                "flag": base.stop_flag, "aux": list(getattr(base, "mcu_aux", []))},
            "arms": arms,
            "nav": dict(self.nav),
            "map_version": self.map_version,
            "plan": self.plan,
            "waypoints": self.waypoints,
            "click_mode": self.click_mode,
            "last_click": self.last_click,
            "remote": self.remote,
            "estop": self.estop,
        }


# ---------------------------------------------------------------------------------------------
class App:
    def __init__(self, node: WebNode, port: int, grpc_port: int, viewer_port: int, head_port: int = 8081):
        self.node, self.port = node, port
        self.grpc_port, self.viewer_port = grpc_port, viewer_port
        self.sup = Supervisor(os.path.expanduser("~/.ros/mercury_x1_web"))
        self.map_name = "default"
        self.restarting = False
        self.static_dir = os.path.join(get_package_share_directory("mercury_x1_web"), "web")
        self.viewer_ready = all(os.path.exists(os.path.join(VIEWER_CACHE, f)) for f in VIEWER_FILES)
        self.viz_layers = self.load_viz_layers()
        self.head_hold = True        # hold the head level while mapping (switch on the page)
        self.wifi = Wifi()
        self.export3d = {"busy": "", "message": "", "files": slam3d.list_exports(), "live": ""}
        self._maps_cache: list[dict] = []
        self._maps_t = 0.0
        threading.Thread(target=self._live3d_loop, daemon=True, name="live3d").start()
        self.head_port = head_port
        self.sysstats = SysStats()           # CPU / GPU load and temperatures for the status page
        node.waypoints_changed = self.save_waypoints
        os.makedirs(MAPS_DIR, exist_ok=True)

    # -- vendored Rerun web viewer (fetched once from jsDelivr) ----------------------------------
    def fetch_viewer(self) -> None:
        if self.viewer_ready:
            return
        os.makedirs(VIEWER_CACHE, exist_ok=True)
        try:
            for f in VIEWER_FILES:
                url = f"https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@{VIEWER_VERSION}/{f}"
                tmp = os.path.join(VIEWER_CACHE, f + ".part")
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, os.path.join(VIEWER_CACHE, f))
            self.viewer_ready = True
            self.node.get_logger().info(f"rerun web viewer {VIEWER_VERSION} cached in {VIEWER_CACHE}")
        except Exception as e:
            self.node.get_logger().warn(f"could not fetch the rerun web viewer ({e}); page falls back to iframe")

    async def vendor(self, req: web.Request) -> web.StreamResponse:
        name = req.match_info["name"]
        if name == "re_viewer":
            name = "re_viewer.js"        # index.js imports "./re_viewer" without an extension
        if name not in VIEWER_FILES or not self.viewer_ready:
            raise web.HTTPNotFound()
        ctype = "application/wasm" if name.endswith(".wasm") else "text/javascript"
        return web.FileResponse(os.path.join(VIEWER_CACHE, name), headers={
            "Content-Type": ctype, "Cache-Control": "public, max-age=86400"})

    def teleop(self, vx: float, wz: float) -> None:
        n = self.node
        if (vx or wz) and n.nav["state"] == "navigating":
            n.post(lambda: n.cancel(stop_base=False))   # driving by hand overrides navigation
        n.post(lambda: n.cmd_vel(max(-0.4, min(0.4, vx)), max(-1.0, min(1.0, wz))))

    # -- waypoints persistence -----------------------------------------------------------------
    def wp_file(self) -> str:
        return os.path.join(MAPS_DIR, f"{self.map_name}.waypoints.json")

    def load_waypoints(self) -> None:
        try:
            with open(self.wp_file()) as f:
                self.node.waypoints = json.load(f)
        except (OSError, ValueError):
            self.node.waypoints = []
        self.node.post(self.node.publish_waypoints)

    def save_waypoints(self) -> None:
        with open(self.wp_file(), "w") as f:
            json.dump(self.node.waypoints, f, indent=1)
        self.node.post(self.node.publish_waypoints)

    # -- routes --------------------------------------------------------------------------------
    async def index(self, req):
        return web.FileResponse(os.path.join(self.static_dir, "index.html"))

    # -- head display (:8081): lightweight touch UI, no 3D -------------------------------------
    async def head_index(self, req):
        return web.FileResponse(os.path.join(self.static_dir, "head.html"))

    def head_status(self) -> dict:
        st = self.node.status()
        return {"battery": st["battery"], "battery_pct": st["battery_pct"], "charge": st["charge"],
                "base": st["base"], "arms": st["arms"], "remote": st["remote"], "nav": st["nav"]["state"], "estop": st["estop"],
                "procs": self.sup.status(), "map_name": self.map_name, "system": self.sysstats.snapshot(),
                "nodes": self.node.node_names, "wifi": self.wifi.snapshot(), "ros": self.ros_network(),
                "time": time.strftime("%H:%M")}

    async def head_ws(self, req):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(req)
        try:
            while not ws.closed:
                await ws.send_str(json.dumps(self.head_status()))
                await asyncio.sleep(1.0)      # 1 Hz: the panel only shows slow-changing state
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return ws

    async def config(self, req):
        return web.json_response({"grpc_port": self.grpc_port, "viewer_port": self.viewer_port,
                                  "viewer_local": self.viewer_ready, "viewer_version": VIEWER_VERSION})

    async def map_png(self, req):
        if self.node.map_png is None:
            raise web.HTTPNotFound()
        return web.Response(body=self.node.map_png, content_type="image/png",
                            headers={"Cache-Control": "no-store"})

    async def map_info(self, req):
        m = self.node.map_msg
        if m is None:
            return web.json_response(None)
        i = m.info
        return web.json_response({"version": self.node.map_version, "width": i.width, "height": i.height,
                                  "resolution": i.resolution, "origin": [i.origin.position.x, i.origin.position.y],
                                  "frame": m.header.frame_id})

    @staticmethod
    def load_viz_layers() -> list[str]:
        try:
            with open(VIZ_LAYERS_FILE) as f:
                saved = json.load(f)
            layers = [k for k in saved.get("nvblox", []) if k in NVBLOX_LAYERS]
            if saved.get("version", 1) < 2:      # saved before the camera layer existed: on, like before
                layers.append("camera")
            return layers
        except (OSError, ValueError, AttributeError):
            return list(DEFAULT_VIZ_LAYERS)

    def mapping(self) -> bool:
        st = self.sup.status()["slam"]
        return st["running"] and st.get("mode") == "mapping"

    def head_locked(self) -> bool:
        return self.head_hold and self.mapping()

    def level_head(self) -> None:
        jc = JointCommand(name=list(MAPPING_HEAD), position=[math.radians(v) for v in MAPPING_HEAD.values()],
                          speed=30)
        self.node.post(lambda: self.node.pub_joint.publish(jc))

    def ros_network(self) -> dict:
        """network = what this stack runs with, configured = what the next (re)start uses"""
        try:
            with open(ROS_NETWORK_FILE) as f:
                configured = f.read().strip()
        except OSError:
            configured = "localhost"
        return {"domain": int(os.environ.get("ROS_DOMAIN_ID", 0)), "network": os.environ.get("MX1_ROS_NETWORK", "?"),
                "configured": configured if configured == "lan" else "localhost", "restarting": self.restarting}

    def restart_stack(self) -> None:
        """restart every ROS process (DDS transports are fixed when a process starts); SLAM / Nav2
        resume as they are now via resume.env (read once by start.sh)"""
        slam, nav = self.sup.status()["slam"], self.sup.status()["nav"]
        with open(os.path.join(CONF_DIR, "resume.env"), "w") as f:
            f.write(f"MX1_SLAM={'true' if slam['running'] else 'false'}\n"
                    f"MX1_NAV={'true' if nav['running'] else 'false'}\n")
            if slam["running"]:
                f.write(f"MX1_MAP={slam.get('map', self.map_name)}\nMX1_MODE={slam.get('mode', 'mapping')}\n")
        self.restarting = True
        # setsid -f: detached from this process, which the restart stops
        subprocess.Popen(["setsid", "-f", os.path.join(MX1_WS, "scripts", "mx1_restart.sh")],
                         stdout=open("/tmp/mx1_restart.log", "w"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)

    # -- 3D map: live layer + exports (slam3d) -------------------------------------------------
    def _db_path(self) -> str:
        return os.path.join(MAPS_DIR, f"{self.map_name}.db")

    def _live3d_loop(self) -> None:
        """While the viewer's 'Whole map (3D)' layer is on: export the map database as a coloured cloud plus
        node poses, fit the export's frame to rtabmap's live map frame on the nodes both share, and publish
        the aligned cloud as a voxel grid (LIVE3D_VOXEL, one point per voxel). rtabmap-export optimises the
        graph on its own (another origin), and in a multi-session database it may pick a different connected
        piece than the live map: then there is nothing to align and the layer says so.
        Re-exported only when the live graph changed (never while localizing on a finished map), and at most
        every LIVE3D_PERIOD while mapping; the latched topic keeps the last cloud for new subscribers."""
        n = self.node
        last_key, last_export, subs = None, 0.0, 0
        while True:
            time.sleep(5.0)
            nsubs = n.pub_cloud3d.get_subscription_count()
            if nsubs == 0 or not self.sup.status()["slam"]["running"]:
                subs = nsubs
                continue
            db = self._db_path()
            if not os.path.exists(db):
                continue
            live = n.live_graph()
            if live is None:
                self.export3d["live"] = "3D layer: rtabmap did not return its graph"
                continue
            # graph unchanged and nobody new (the bridge re-subscribing gets the latched cloud anyway)
            key = (self.map_name, len(live), max(live, default=0), round(sum(p[0] + p[1] for p in live.values()), 1))
            if key == last_key and nsubs <= subs:
                subs = nsubs
                continue
            subs = nsubs
            if last_key is not None and key[0] == last_key[0] and time.time() - last_export < LIVE3D_PERIOD:
                continue
            last_key, last_export = key, time.time()
            try:
                out_dir = os.path.join(slam3d.SNAP_DIR, "live")
                ply = slam3d.export_cloud(db, out_dir, "live", voxel=0.03, poses=True)
                exp = slam3d.read_poses(os.path.join(out_dir, "live_poses.txt"))
                ids = sorted(set(exp) & set(live))
                if len(ids) < 3:
                    self.export3d["live"] = (f"3D layer: the export ({len(exp)} nodes) and the live map ({len(live)} nodes) "
                                             f"share {len(ids)} nodes - the database has disconnected sessions; re-map")
                    continue
                r, t, err = slam3d.fit_rigid(np.array([exp[i] for i in ids]), np.array([live[i] for i in ids]))
                if err > 0.25:
                    self.export3d["live"] = f"3D layer: export and live map disagree ({err:.2f} m mean on {len(ids)} nodes)"
                    continue
                xyz, rgb = slam3d.read_ply(ply)
                vox, vrgb = slam3d.voxel_down((xyz @ r.T + t).astype(np.float32), rgb, LIVE3D_VOXEL)
                msg = cloud_msg(vox, vrgb, "map", n.get_clock().now().to_msg())
                n.post(lambda m=msg: n.pub_cloud3d.publish(m))
                self.export3d["live"] = (f"3D layer: {len(vox)} voxels ({LIVE3D_VOXEL * 100:.0f} cm) from {len(xyz)} points, "
                                         f"{len(exp)} nodes, fit {err * 100:.0f} cm, {time.strftime('%H:%M:%S')}")
            except Exception as e:  # noqa: BLE001 - keep the loop alive
                self.export3d["live"] = f"3D layer: {e}"

    def _export3d_job(self, kind: str) -> None:
        name = f"{self.map_name}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.export3d.update(busy=f"exporting {kind} of '{self.map_name}'", message="")
        try:
            fn = slam3d.export_mesh if kind == "mesh" else slam3d.export_cloud
            out = fn(self._db_path(), slam3d.EXPORT_DIR, name)
            self.export3d["message"] = f"saved {os.path.basename(out)}"
        except Exception as e:  # noqa: BLE001
            self.export3d["message"] = f"export failed: {e}"
        finally:
            self.export3d.update(busy="", files=slam3d.list_exports())

    async def export_file(self, req):
        name = os.path.basename(req.match_info["name"])
        path = os.path.join(slam3d.EXPORT_DIR, name)
        if not os.path.isfile(path):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    def map_list(self) -> list[dict]:
        slam = self.sup.status()["slam"]
        out = []
        for f in os.listdir(MAPS_DIR):
            if not f.endswith(".db"):
                continue
            name, st = f[:-3], os.stat(os.path.join(MAPS_DIR, f))
            try:
                with open(os.path.join(MAPS_DIR, f"{name}.waypoints.json")) as wf:
                    nwp = len(json.load(wf))
            except (OSError, ValueError, TypeError):
                nwp = 0
            out.append({"name": name, "size_mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime,
                        "time": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), "waypoints": nwp,
                        "active": slam.get("mode") if slam["running"] and slam.get("map") == name else None})
        return sorted(out, key=lambda m: -m["mtime"])

    async def maps(self, req):
        maps = self.map_list()
        return web.json_response({"maps": [m["name"] for m in maps], "details": maps, "current": self.map_name})

    def delete_map(self, name: str) -> str:
        """move <name>.db and its side files to ~/maps/.trash/<name>-<time>/ (3D exports stay)"""
        dest = os.path.join(TRASH_DIR, f"{name}-{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(dest, exist_ok=True)
        for ext in (".db", ".waypoints.json", ".pgm", ".yaml"):
            src = os.path.join(MAPS_DIR, name + ext)
            if os.path.exists(src):
                shutil.move(src, os.path.join(dest, name + ext))
        cutoff = time.time() - TRASH_DAYS * 86400
        for d in os.listdir(TRASH_DIR):
            path = os.path.join(TRASH_DIR, d)
            if os.path.getmtime(path) < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        if name == self.map_name:
            self.node.waypoints = []
            self.node.post(self.node.publish_waypoints)
        return dest

    async def ws(self, req):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(req)
        try:
            while not ws.closed:
                st = self.node.status()
                st["procs"] = self.sup.status()
                st["map_name"] = self.map_name
                st["viz_layers"] = {"nvblox": self.viz_layers}
                st["head_hold"] = self.head_hold
                st["export3d"] = {**self.export3d, "files": self.export3d["files"][:6]}
                st["system"] = self.sysstats.snapshot()
                st["ros"] = self.ros_network()
                if time.monotonic() - self._maps_t > 3.0:     # saved maps: cheap, but no need at 5 Hz
                    self._maps_cache, self._maps_t = self.map_list(), time.monotonic()
                st["maps"] = self._maps_cache
                await ws.send_str(json.dumps(st))
                await asyncio.sleep(0.2)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return ws

    async def api(self, req):
        action = req.match_info["action"]
        body = await req.json() if req.can_read_body else {}
        n = self.node
        loop = asyncio.get_running_loop()
        if action not in ("cmd_vel", "click", "head", "led"):     # frequent ones stay out of the log
            shown = {k: v for k, v in body.items() if k != "password"} if isinstance(body, dict) else body
            n.get_logger().info(f"api {action} {json.dumps(shown)[:200]}")
        if action == "cmd_vel":
            self.teleop(float(body.get("vx", 0)), float(body.get("wz", 0)))
        elif action == "click":       # point picked in the embedded Rerun viewer
            n.post(lambda: n.handle_click(float(body["x"]), float(body["y"]), float(body.get("z", 0.0)),
                                          str(body.get("entity", "")), "web"))
        elif action == "click_mode":
            mode = body.get("mode", "waypoint")
            if mode not in ("waypoint", "delete", "goto", "pose", "off"):
                return web.json_response({"ok": False, "error": f"bad mode {mode}"})
            n.click_mode = mode
        elif action == "stop":
            n.post(n.cancel)
        elif action == "goto":
            n.post(lambda: n.go_to(body["x"], body["y"], body.get("yaw", 0.0)))
        elif action == "map_delete":    # {"name"}: to ~/maps/.trash (kept TRASH_DAYS days)
            name = str(body.get("name", ""))
            slam = self.sup.status()["slam"]
            if not name or "/" in name or name.startswith(".") or not os.path.exists(os.path.join(MAPS_DIR, f"{name}.db")):
                return web.json_response({"ok": False, "error": f"no map '{name}'"})
            if slam["running"] and slam.get("map") == name:
                return web.json_response({"ok": False, "error": f"'{name}' is in use - stop SLAM first"})
            dest = await loop.run_in_executor(None, lambda: self.delete_map(name))
            self._maps_t = 0.0
            return web.json_response({"ok": True, "trash": dest})
        elif action == "waypoint_delete":   # {"index"}
            k = int(body.get("index", -1))
            if not 0 <= k < len(n.waypoints):
                return web.json_response({"ok": False, "error": "no such waypoint"})
            del n.waypoints[k]
            self.save_waypoints()
        elif action == "waypoints":
            n.waypoints = [{"x": float(w["x"]), "y": float(w["y"]), "yaw": float(w.get("yaw", 0))}
                           for w in body.get("waypoints", [])]
            self.save_waypoints()
        elif action == "run":
            n.post(lambda: n.run_waypoints(bool(body.get("loop", False))))
        elif action == "initialpose":
            n.post(lambda: n.set_initial_pose(body["x"], body["y"], body.get("yaw", 0.0)))
        elif action == "wifi_scan":       # results arrive in the head status (wifi.networks)
            threading.Thread(target=self.wifi.scan, daemon=True).start()
        elif action == "wifi_connect":    # {"ssid", "password"?}; progress / result in wifi.busy / wifi.message
            ssid = str(body.get("ssid", "")).strip()
            if not ssid or self.wifi.busy:
                return web.json_response({"ok": False, "error": "no network given" if not ssid else self.wifi.busy})
            threading.Thread(target=self.wifi.connect, args=(ssid, str(body.get("password", ""))),
                             daemon=True).start()
        elif action == "export3d":       # {"kind": "cloud" | "mesh"}: files land in ~/maps/exports
            kind = body.get("kind", "cloud")
            if kind not in ("cloud", "mesh") or self.export3d["busy"]:
                return web.json_response({"ok": False, "error": self.export3d["busy"] or f"bad kind {kind}"})
            if not os.path.exists(self._db_path()):
                return web.json_response({"ok": False, "error": f"no map database '{self.map_name}'"})
            threading.Thread(target=self._export3d_job, args=(kind,), daemon=True).start()
        elif action == "viz_layers":     # {"nvblox": [...]} -> rerun_bridge parameter, saved for restarts
            layers = [k for k in NVBLOX_LAYERS if k in body.get("nvblox", [])]
            self.viz_layers = layers
            os.makedirs(CONF_DIR, exist_ok=True)
            with open(VIZ_LAYERS_FILE, "w") as f:
                json.dump({"nvblox": layers, "version": 2}, f)
            n.post(lambda: n.set_nvblox_layers(layers))
        elif action == "ros_network":   # {"network": "localhost" | "lan"} -> saved, then the stack restarts
            net = body.get("network")
            if net not in ("localhost", "lan"):
                return web.json_response({"ok": False, "error": f"bad network {net}"})
            os.makedirs(CONF_DIR, exist_ok=True)
            with open(ROS_NETWORK_FILE, "w") as f:
                f.write(net + "\n")
            n.post(n.cancel)
            await loop.run_in_executor(None, self.restart_stack)
        elif action == "remote":      # {"op": scan|connect|disconnect|autoconnect|forget, ...} -> unitree_remote
            if body.get("op") not in ("scan", "connect", "disconnect", "autoconnect", "forget"):
                return web.json_response({"ok": False, "error": "bad remote op"})
            n.post(lambda: n.pub_remote.publish(String(data=json.dumps(body))))
        elif action == "led":
            c = ColorRGBA(r=float(body.get("r", 0)), g=float(body.get("g", 0)), b=float(body.get("b", 0)), a=1.0)
            n.post(lambda: n.pub_led.publish(c))
        elif action == "head_hold":     # {"on": bool}; turning it on while mapping levels the head again
            self.head_hold = bool(body.get("on", True))
            if self.head_locked():
                self.level_head()
        elif action == "head":
            if self.head_locked():
                return web.json_response({"ok": False, "error": "head is held level while mapping (switch off 'Hold level while mapping')"})
            jc = JointCommand()
            for k in ("eye", "head"):
                if k in body:
                    jc.name.append(k)
                    jc.position.append(math.radians(float(body[k])))
            jc.speed = int(body.get("speed", 30))
            n.post(lambda: n.pub_joint.publish(jc))
        elif action == "freedrive":
            on = bool(body.get("on", False))
            n.post(lambda: n.set_freedrive(on))
        elif action == "arms_power":
            on = bool(body.get("on", False))
            for s in ("left", "right"):
                req_ = SetBool.Request(data=on)
                n.post(lambda s=s, r=req_: n.call(f"{s}_power", r))
        elif action == "slam":
            mode = body.get("mode", "mapping")
            name = "".join(c for c in body.get("map", self.map_name) if c.isalnum() or c in "-_") or "default"
            fresh = bool(body.get("fresh", False))
            if mode == "localization" and not os.path.exists(os.path.join(MAPS_DIR, f"{name}.db")):
                return web.json_response({"ok": False, "error": f"no map '{name}' - build it in mapping mode first"})
            if mode == "mapping" and fresh and os.path.exists(os.path.join(MAPS_DIR, f"{name}.db")):
                slam = self.sup.status()["slam"]
                if slam["running"] and slam.get("map") == name:
                    await loop.run_in_executor(None, lambda: self.sup.stop("slam"))
                # replacing a map: the old one (and its waypoints) goes to the trash, not lost
                await loop.run_in_executor(None, lambda: self.delete_map(name))
            self.map_name = name
            self.load_waypoints()
            self._maps_t = 0.0
            if mode == "mapping" and self.head_hold:   # the depth camera feeds the map: look straight ahead, level
                self.level_head()
            cmd = ["ros2", "launch", "mercury_x1_bringup", "slam.launch.py", f"mode:={mode}", f"map:={name}",
                   f"fresh:={'true' if fresh else 'false'}"]
            await loop.run_in_executor(None, lambda: self.sup.start("slam", cmd, {"mode": mode, "map": name}))
        elif action == "slam_stop":
            n.post(n.cancel)
            await loop.run_in_executor(None, lambda: self.sup.stop("slam"))
        elif action == "nav":
            await loop.run_in_executor(None, lambda: self.sup.start(
                "nav", ["ros2", "launch", "mercury_x1_bringup", "nav.launch.py"], {}))
        elif action == "nav_stop":
            n.post(n.cancel)
            await loop.run_in_executor(None, lambda: self.sup.stop("nav"))
        elif action == "export_map":
            if n.map_msg is None:
                return web.json_response({"ok": False, "error": "no map yet"})
            prefix = os.path.join(MAPS_DIR, self.map_name)
            r = await loop.run_in_executor(None, lambda: subprocess.run(
                ["ros2", "run", "nav2_map_server", "map_saver_cli", "-f", prefix, "--ros-args",
                 "-p", "map_subscribe_transient_local:=true"], capture_output=True, timeout=30))
            return web.json_response({"ok": r.returncode == 0, "file": prefix + ".pgm"})
        else:
            raise web.HTTPNotFound()
        return web.json_response({"ok": True})

    def run(self) -> None:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/config", self.config)
        app.router.add_get("/ws", self.ws)
        app.router.add_get("/map.png", self.map_png)
        app.router.add_get("/map.json", self.map_info)
        app.router.add_get("/maps", self.maps)
        app.router.add_get("/vendor/rerun/{name}", self.vendor)
        app.router.add_post("/api/{action}", self.api)
        app.router.add_static("/static", self.static_dir)
        app.router.add_get("/exports/{name}", self.export_file)
        threading.Thread(target=self.fetch_viewer, daemon=True).start()
        self.load_waypoints()

        head = web.Application()          # head display: same API, its own page and a 1 Hz status
        head.router.add_get("/", self.head_index)
        head.router.add_get("/ws", self.head_ws)
        head.router.add_post("/api/{action}", self.api)
        head.router.add_static("/static", self.static_dir)

        async def serve() -> None:
            runners = []
            for application, port in ((app, self.port), (head, self.head_port)):
                runner = web.AppRunner(application)
                await runner.setup()
                await web.TCPSite(runner, port=port).start()
                runners.append(runner)
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop.set)
            await stop.wait()
            for name in ("nav", "slam"):     # rtabmap saves its database on SIGINT
                await loop.run_in_executor(None, lambda n=name: self.sup.stop(n))
            for runner in runners:
                await runner.cleanup()

        asyncio.run(serve())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WebNode()
    port = int(node.declare_parameter("port", 8080).value)
    grpc_port = int(node.declare_parameter("grpc_port", 9876).value)
    viewer_port = int(node.declare_parameter("viewer_port", 9090).value)
    head_port = int(node.declare_parameter("head_port", 8081).value)
    ex = rclpy.executors.SingleThreadedExecutor()   # rclpy multi-threaded spinning is CPU-heavy
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    node.get_logger().info(f"control page on http://0.0.0.0:{port}, head display on :{head_port}")
    try:
        App(node, port, grpc_port, viewer_port, head_port).run()
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
