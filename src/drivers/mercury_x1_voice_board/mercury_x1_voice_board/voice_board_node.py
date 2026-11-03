"""ROS 2 driver for the Mercury X1 head voice board (iFlytek AIUI mic-array board, USB 2207:0001).

The board does wake word + direction of arrival + beamforming/noise suppression on board. Its processed
16 kHz mono audio is a plain USB microphone (ALSA card XFMDPV0018) and is NOT handled here; this node
talks to the board's control UART (CH343 USB-serial, /dev/xf_mic via udev).

Topics:   ~/wake (mercury_x1_msgs/VoiceWake), ~/state (VoiceBoardState), ~/events (std_msgs/String, every
          board event as JSON)
Services: ~/command (VoiceBoardCommand, raw JSON), ~/set_wake_word (SetWakeWord), ~/set_beam (SetBeam)
"""
import json
import math
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String

from mercury_x1_msgs.msg import VoiceBoardState, VoiceWake
from mercury_x1_msgs.srv import SetBeam, SetWakeWord, VoiceBoardCommand

from . import aiui_protocol as proto
from .serial_link import VoiceBoardLink, event_text

# re-init trigger after a wake word change: writing user_cfg.ini makes the board app reload everything
RELOAD_SETTING = {"type": "setting", "content": [{"group": "vtn", "key": "log", "value": "0"}]}
VOLUME_LIMITS = {"mic_vol": 30, "ref_vol": 30, "output_vol": 7}


class VoiceBoardNode(Node):
    def __init__(self) -> None:
        super().__init__("voice_board")
        p = self.declare_parameter
        self.port = p("port", "/dev/xf_mic").value
        self.frame_id = p("frame_id", "voice_board_link").value
        self.angle_front_deg = float(p("angle_front_deg", 90.0).value)
        self.angle_sign = float(p("angle_sign", 1.0).value)
        self.reply_timeout = float(p("reply_timeout", 3.0).value)
        self.keyword_timeout = float(p("keyword_timeout", 30.0).value)
        self.heartbeat_period = float(p("heartbeat_period", 30.0).value)
        self.alive_timeout = float(p("alive_timeout", 75.0).value)
        self.repeat_window = float(p("repeat_window", 2.0).value)
        self.startup = {name: int(p(name, -1).value) for name in ("mic_vol", "ref_vol", "output_vol", "beam")}

        self.wake_pub = self.create_publisher(VoiceWake, "~/wake", 10)
        self.state_pub = self.create_publisher(VoiceBoardState, "~/state", 10)
        self.event_pub = self.create_publisher(String, "~/events", 50)
        group = ReentrantCallbackGroup()
        self.create_service(VoiceBoardCommand, "~/command", self._srv_command, callback_group=group)
        self.create_service(SetWakeWord, "~/set_wake_word", self._srv_set_wake_word, callback_group=group)
        self.create_service(SetBeam, "~/set_beam", self._srv_set_beam, callback_group=group)
        self.add_on_set_parameters_callback(self._on_params)

        self.firmware_version = ""
        self.wake_count = 0
        self.last_wake_angle = float("nan")
        self.last_error = ""
        self._last_wake_key = None
        self._last_wake_time = 0.0
        self._configure = threading.Event()

        self.link = VoiceBoardLink(self.port, self._on_event, self._on_link_state)
        self.link.start()
        threading.Thread(target=self._config_worker, name="voice_board_cfg", daemon=True).start()
        self.create_timer(1.0, self._publish_state)
        self.create_timer(max(self.heartbeat_period, 5.0), self._heartbeat, callback_group=group)
        self.get_logger().info(f"voice board on {self.port}")

    # -- board -> host -------------------------------------------------------------------------
    def _on_link_state(self, text: str) -> None:
        if text == "open":
            self.get_logger().info(f"{self.port} open")
            self.last_error = ""
            self._configure.set()
        elif text == "closed":
            self.get_logger().warn(f"{self.port} closed")
        elif text != self.last_error:
            self.last_error = text
            self.get_logger().warn(text, throttle_duration_sec=30.0)

    def _on_event(self, event: dict) -> None:
        self.event_pub.publish(String(data=event_text(event)))
        kind = event.get("type")
        if kind in ("version", "started"):
            content = str(event.get("content", ""))
            self.firmware_version = content.split("version:", 1)[-1].strip()
            if kind == "started":  # the board app (re)started: re-apply our settings
                self.get_logger().info(f"board app started: {self.firmware_version}")
                self._configure.set()
            return
        wake = proto.parse_wakeup(event)
        if wake is not None:
            self._publish_wake(wake)
        elif kind not in ("manual_wakeup", "mic_vol", "ref_vol", "output_vol", "setting", "wakeup_keywords"):
            self.get_logger().info(f"board event: {event_text(event)}")

    def _publish_wake(self, wake: proto.WakeInfo) -> None:
        now = time.monotonic()
        key = (wake.start_ms, wake.keyword) if wake.start_ms >= 0 else (wake.keyword,)
        if key == self._last_wake_key and now - self._last_wake_time < self.repeat_window:
            return  # the firmware repeats wake events
        self._last_wake_key, self._last_wake_time = key, now
        msg = VoiceWake()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.keyword = wake.keyword
        msg.angle_deg = float(wake.angle_deg)
        msg.yaw = math.radians(self.angle_sign * (wake.angle_deg - self.angle_front_deg)) \
            if math.isfinite(wake.angle_deg) else float("nan")
        msg.beam = wake.beam
        msg.score = float(wake.score)
        msg.start_ms = wake.start_ms
        msg.raw = wake.raw
        self.wake_pub.publish(msg)
        self.wake_count += 1
        self.last_wake_angle = msg.angle_deg
        self.get_logger().info(f"wake '{wake.keyword}' angle {wake.angle_deg:.0f} deg beam {wake.beam} "
                               f"score {wake.score}")

    # -- host -> board -------------------------------------------------------------------------
    def _request(self, message: dict, timeout: float = 0.0):
        return self.link.request(message, timeout if timeout > 0 else self.reply_timeout)

    def _config_worker(self) -> None:
        """After every (re)open or board restart: read the version, apply configured settings."""
        while rclpy.ok():
            self._configure.wait()
            self._configure.clear()
            time.sleep(0.5)  # let the first handshakes be confirmed
            if self._request({"type": "version"}) is None:
                self.get_logger().warn("no reply to version request")
            for name, limit in VOLUME_LIMITS.items():
                value = self.startup[name]
                if 0 <= value <= limit:
                    self._apply_volume(name, value)
            if self.startup["beam"] >= 0:
                self._request({"type": "manual_wakeup", "content": {"beam": self.startup["beam"]}})

    def _apply_volume(self, name: str, value: int) -> bool:
        reply = self._request({"type": name, "content": {"volume": value}})
        ok = reply is not None and reply.get("code") == 0
        if not ok:
            self.get_logger().warn(f"{name}={value}: {reply}")
        return ok

    def _heartbeat(self) -> None:
        if self.link.is_open and time.monotonic() - self.link.last_rx > self.heartbeat_period:
            self._request({"type": "version"})

    def _on_params(self, params) -> SetParametersResult:
        for param in params:
            if param.name in VOLUME_LIMITS:
                value = int(param.value)
                if not -1 <= value <= VOLUME_LIMITS[param.name]:
                    return SetParametersResult(successful=False,
                                               reason=f"{param.name} must be 0..{VOLUME_LIMITS[param.name]}")
                self.startup[param.name] = value
                if value >= 0 and self.link.is_open:
                    threading.Thread(target=self._apply_volume, args=(param.name, value), daemon=True).start()
            elif param.name == "beam":
                self.startup["beam"] = int(param.value)
            elif param.name == "angle_front_deg":
                self.angle_front_deg = float(param.value)
            elif param.name == "angle_sign":
                self.angle_sign = float(param.value)
        return SetParametersResult(successful=True)

    # -- services -------------------------------------------------------------------------------
    def _srv_command(self, req, res):
        try:
            message = json.loads(req.json)
        except json.JSONDecodeError as exc:
            res.success, res.code, res.content = False, -1, f"invalid JSON: {exc}"
            return res
        if not isinstance(message, dict) or "type" not in message:
            res.success, res.code, res.content = False, -1, 'need a JSON object with "type"'
            return res
        reply = self._request(message, req.timeout)
        if reply is None:
            res.success, res.code, res.content = False, -1, "no reply from the board"
            return res
        res.code = int(reply.get("code", 0)) if isinstance(reply.get("code", 0), int) else -1
        res.content = str(reply.get("content", ""))
        res.reply = event_text(reply)
        res.success = res.code == 0
        return res

    def _srv_set_wake_word(self, req, res):
        keyword = req.keyword.strip()
        if not keyword:
            res.success, res.message = False, "empty keyword"
            return res
        threshold = req.threshold.strip() or ";".join(["900"] * len(keyword.split(";")))
        reply = self._request({"type": "wakeup_keywords",
                               "content": {"keyword": keyword, "threshold": threshold}}, self.keyword_timeout)
        res.success = reply is not None and reply.get("code", 0) == 0
        res.message = event_text(reply) if reply else "no reply (the board may still be compiling the model)"
        if req.reload:
            reload_reply = self._request(RELOAD_SETTING, self.reply_timeout)
            res.message += f"; reload: {event_text(reload_reply) if reload_reply else 'sent'}"
        return res

    def _srv_set_beam(self, req, res):
        reply = self._request({"type": "manual_wakeup", "content": {"beam": int(req.beam)}})
        res.success = reply is not None and reply.get("code") == 0
        res.message = str(reply.get("content")) if reply else "no reply from the board"
        return res

    # -- state ------------------------------------------------------------------------------------
    def _publish_state(self) -> None:
        msg = VoiceBoardState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.port = self.port
        msg.port_open = self.link.is_open
        msg.alive = self.link.is_open and self.link.last_rx > 0 \
            and time.monotonic() - self.link.last_rx < self.alive_timeout
        msg.firmware_version = self.firmware_version
        msg.handshakes = self.link.handshakes
        msg.wake_count = self.wake_count
        msg.last_wake_angle_deg = self.last_wake_angle
        msg.dropped_bytes = self.link.parser.dropped_bytes
        msg.last_error = self.last_error
        self.state_pub.publish(msg)

    def destroy_node(self):
        self.link.stop()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoiceBoardNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
