"""Body language for voice chat: small waist / head moves while talking, head toward the person who said the wake
word, nod / shake, and a handshake (right arm forward through MoveIt, a few small pumps, back to rest).

Conservative by design: slow firmware speeds, a few degrees of amplitude, positions relative to where the joints
are now (read from /joint_states), arm moves only through MoveIt (self-collision checked). Nothing runs unless
voice chat's "gestures" switch is on (voice.py checks it); stop() cancels everything.
Sequences run on their own thread; MoveIt calls go to the ROS thread (node.post).
"""
from __future__ import annotations

import math
import threading
import time

from mercury_x1_msgs.msg import JointCommand

SPEED = 12                 # firmware speed 1..100 for head / waist moves
WAIST_SWAY_DEG = 4.0
HEAD_SWAY_DEG = 6.0
HEAD_LOOK_LIMIT_DEG = 60.0  # never turn the head further than this towards a voice
HANDSHAKE_POSE = "reach_forward_R"
REST_POSE = "rest"
PUMP_JOINT = "joint4_R"     # elbow; a handshake = small up/down here
PUMP_DEG = 4.0


class Gestures:
    def __init__(self, node, arm_motion) -> None:
        self.node, self.arm = node, arm_motion
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.busy = ""
        self.home: dict[str, float] = {}   # waist / head / eye [deg] when talking started

    # -- public ----------------------------------------------------------------------------------
    def names(self) -> list[str]:
        return ["nod", "shake_head", "handshake"]

    def run(self, name: str) -> tuple[bool, str]:
        seq = {"nod": self._nod, "shake_head": self._shake, "handshake": self._handshake}.get(name)
        if seq is None:
            return False, f"unknown gesture {name}"
        if name == "handshake" and not self.arm.client.server_is_ready():
            return False, "arm planning (MoveIt) is not running"
        return self._start(name, seq)

    def talking(self, on: bool) -> None:
        """gentle waist + head sway while the robot speaks; back to where it was when it stops"""
        if on:
            if not self.busy:
                self._start("talk", self._talk)
        elif self.busy == "talk":
            self._stop.set()

    def look_at(self, angle_deg: float, front_deg: float = 90.0, sign: float = 1.0) -> None:
        """turn the head towards a voice direction from the mic array (linear array: front / back ambiguous)"""
        if self.busy and self.busy != "talk":
            return
        yaw = max(-HEAD_LOOK_LIMIT_DEG, min(HEAD_LOOK_LIMIT_DEG, sign * (angle_deg - front_deg)))
        self._send({"head": yaw}, SPEED + 8)

    def stop(self) -> None:
        self._stop.set()
        if self.busy == "handshake":
            self.node.post(self.arm.stop)

    # -- internals -------------------------------------------------------------------------------
    def _start(self, name: str, fn) -> tuple[bool, str]:
        if self.busy:
            if self.busy != "talk":
                return False, f"busy with {self.busy}"
            self._stop.set()
            if self._thread:
                self._thread.join(2.0)
        self._stop.clear()
        self.busy = name
        self._thread = threading.Thread(target=self._wrap, args=(fn,), daemon=True, name=f"gesture-{name}")
        self._thread.start()
        return True, f"{name.replace('_', ' ')} started"

    def _wrap(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - logged, never fatal
            self.node.get_logger().warn(f"gesture failed: {e}")
        finally:
            self.busy = ""

    def _now(self, *names: str) -> dict[str, float]:
        bj = self.node.body_joints   # degrees, from /joint_states
        return {n: bj.get(n, 0.0) for n in names}

    def _send(self, deg: dict[str, float], speed: int = SPEED) -> None:
        jc = JointCommand(name=list(deg), position=[math.radians(v) for v in deg.values()])
        jc.speed = max(1, min(40, int(speed)))
        self.node.pub_joint.publish(jc)

    def _wait(self, s: float) -> bool:
        """sleep; False if stopped meanwhile"""
        return not self._stop.wait(s)

    def _talk(self) -> None:
        home = self._now("body", "head")
        k = 0
        while not self._stop.is_set():
            k += 1
            sgn = 1 if k % 2 else -1
            self._send({"body": home["body"] + sgn * WAIST_SWAY_DEG, "head": home["head"] - sgn * HEAD_SWAY_DEG})
            if not self._wait(1.8):
                break
        self._send(home)

    def _shake(self) -> None:
        home = self._now("head")
        for sgn in (1, -1, 1, -1):
            self._send({"head": home["head"] + sgn * 12.0}, SPEED + 10)
            if not self._wait(0.5):
                break
        self._send(home)

    def _nod(self) -> None:
        home = self._now("eye")
        for _ in range(2):
            self._send({"eye": home["eye"] + 8.0}, SPEED + 10)
            if not self._wait(0.45):
                break
            self._send(home, SPEED + 10)
            if not self._wait(0.45):
                break
        self._send(home)

    def _moveit(self, pose: str, speed: float, timeout: float = 25.0) -> bool:
        t_start = time.time()
        self.node.post(lambda: self.arm.goto(pose, speed))
        while time.time() - t_start < timeout:
            if self._stop.is_set():
                return False
            if self.arm.t > t_start and self.arm.state in ("done", "failed", "stopped"):
                return self.arm.state == "done"
            time.sleep(0.1)
        return False

    def _handshake(self) -> None:
        if not self._moveit(HANDSHAKE_POSE, 0.15):
            self.node.get_logger().warn(f"handshake: {self.arm.message}")
            return
        base = math.degrees(self.arm_joint(PUMP_JOINT))
        for sgn in (1, -1, 1, -1, 1, -1):
            self._send({PUMP_JOINT: base + sgn * PUMP_DEG}, 20)
            if not self._wait(0.5):
                break
        self._send({PUMP_JOINT: base}, 20)
        self._wait(1.0)
        self._moveit(REST_POSE, 0.15)

    def arm_joint(self, name: str) -> float:
        pos = getattr(self.node, "arm_joints", {}).get(name)
        return pos if pos is not None else -1.2
