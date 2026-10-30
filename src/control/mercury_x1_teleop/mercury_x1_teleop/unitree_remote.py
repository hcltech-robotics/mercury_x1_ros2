"""Unitree BLE handheld remote -> /joy + /cmd_vel, over the robot's built-in Bluetooth (BlueZ via bleak).

Interface (the web page drives it through mercury_x1_web):
  /remote/cmd     std_msgs/String JSON  {"op": "scan"} | {"op": "connect", "address": "AA:.."} |
                                        {"op": "disconnect"} | {"op": "forget"} |
                                        {"op": "autoconnect", "enabled": true[, "address", "name"]}
  /remote/status  std_msgs/String JSON  (latched, 5 Hz) state, devices found, saved remote, battery, ...
  /remote/button  std_msgs/String       button name on press
  /joy            sensor_msgs/Joy       axes [lx, ly, rx, ry, dpad_x, dpad_y] (left/up = +1), 16 buttons
  /cmd_vel        geometry_msgs/Twist   only while driving: hold the dead-man button (L1) and move
                                        the left stick (forward/back) / right stick (turn)

Autoconnect keeps scanning for the saved remote (any "Unitree*" remote if none is saved yet) and
reconnects when it drops. "Disconnect" also turns autoconnect off. The saved remote and the
autoconnect flag persist in ~/.config/mercury_x1/unitree_remote.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time

import rclpy
from bleak import BleakClient, BleakScanner
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import String

from .remote_protocol import BUTTONS, HANDSHAKE, NAME_PREFIX, NOTIFY_UUID, WRITE_UUID, RemoteState, decode, joy_axes

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


class UnitreeRemote(Node):
    def __init__(self) -> None:
        super().__init__("unitree_remote")
        p = self.declare_parameter
        self.adapter = p("adapter", "hci0").value
        self.max_vx = float(p("max_linear", 0.35).value)
        self.max_vx_rev = float(p("max_linear_reverse", 0.15).value)
        self.max_wz = float(p("max_angular", 0.8).value)
        self.deadzone = float(p("deadzone", 0.08).value)
        self.deadman = str(p("deadman_button", "L1").value)        # "" = sticks drive directly
        self.stale = float(p("stale_timeout", 1.0).value)           # no packets -> stop driving
        self.lost = float(p("lost_timeout", 4.0).value)             # no packets -> drop the link
        self.store = os.path.expanduser(p("state_file", "~/.config/mercury_x1/unitree_remote.json").value)

        saved = self._load()
        self.saved = {"address": saved.get("address"), "name": saved.get("name")}
        self.autoconnect = bool(saved.get("autoconnect", False))

        self.lock = threading.Lock()
        self.state = "idle"          # idle | scanning | connecting | connected
        self.message = ""
        self.devices: list[dict] = []
        self.ble_devices: dict = {}
        self.scan_time = 0.0
        self.client: BleakClient | None = None
        self.conn = {"address": None, "name": None}
        self.rs: RemoteState | None = None
        self.last_rx = 0.0
        self.driving = False
        self.stop_burst = 0
        self.busy = False
        self.next_try = 0.0
        self.failures = 0

        self.pub_joy = self.create_publisher(Joy, "/joy", 10)
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_status = self.create_publisher(String, "/remote/status", LATCHED)
        self.pub_button = self.create_publisher(String, "/remote/button", 10)
        self.create_subscription(String, "/remote/cmd", self._on_cmd, 10)
        self.create_timer(0.05, self._teleop_tick)
        self.create_timer(0.2, self._publish_status)

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="ble").start()
        self._run(self._supervisor())
        self.get_logger().info(f"Unitree remote on {self.adapter}: saved={self.saved['address']} "
                               f"autoconnect={self.autoconnect} dead-man={self.deadman or 'none'}")

    # ---- persistence ------------------------------------------------------------------------
    def _load(self) -> dict:
        try:
            with open(self.store) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store), exist_ok=True)
        with open(self.store, "w") as f:
            json.dump({**self.saved, "autoconnect": self.autoconnect}, f, indent=1)

    # ---- ROS side -----------------------------------------------------------------------------
    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def _on_cmd(self, msg: String) -> None:
        try:
            cmd = json.loads(msg.data)
        except ValueError:
            return
        op = cmd.get("op")
        if op == "scan":
            self._run(self._manual_scan())
        elif op == "connect" and cmd.get("address"):
            self._run(self._connect(str(cmd["address"]).upper(), cmd.get("name")))
        elif op == "disconnect":
            self.autoconnect = False          # otherwise it would reconnect right away
            self._save()
            self._run(self._disconnect())
        elif op == "autoconnect":           # optional address/name: the remote to look for
            self.autoconnect = bool(cmd.get("enabled", True))
            if self.autoconnect and cmd.get("address"):
                self.saved = {"address": str(cmd["address"]).upper(), "name": cmd.get("name")}
            self.next_try = 0.0
            self._save()
        elif op == "forget":
            self.saved = {"address": None, "name": None}
            self._save()

    def _set(self, state: str | None = None, message: str | None = None) -> None:
        if state is not None:
            self.state = state
        if message is not None:
            self.message = message
            if message:
                self.get_logger().info(message)

    def _publish_status(self) -> None:
        with self.lock:
            rs, age = self.rs, time.monotonic() - self.last_rx
        live = rs is not None and self.state == "connected"
        st = {
            "state": self.state, "message": self.message, "autoconnect": self.autoconnect,
            "saved": self.saved, "connected": self.conn if self.state == "connected" else None,
            "devices": self.devices, "driving": self.driving, "deadman": self.deadman,
            "battery": rs.battery if live else None, "rssi": rs.rssi if live else None,
            "rx_age": round(age, 2) if live else None,
            "sticks": [round(v, 2) for v in (rs.lx, rs.ly, rs.rx, rs.ry)] if live else None,
            "pressed": [b for b in BUTTONS if rs.buttons[b]] if live else [],
        }
        self.pub_status.publish(String(data=json.dumps(st)))

    def _teleop_tick(self) -> None:
        with self.lock:
            rs, age = self.rs, time.monotonic() - self.last_rx
        active, vx, wz = False, 0.0, 0.0
        if rs is not None and self.state == "connected" and age < self.stale:
            dz = self.deadzone
            shape = lambda v: 0.0 if abs(v) < dz else (abs(v) - dz) / (1.0 - dz) * (1 if v > 0 else -1)
            fwd = shape(rs.ly)
            turn = shape(-rs.rx) or shape(-rs.lx)        # right stick turns; left stick x as fallback
            held = not self.deadman or rs.buttons.get(self.deadman, False)
            if held and (fwd or turn):
                active = True
                vx = fwd * (self.max_vx if fwd > 0 else self.max_vx_rev)
                wz = turn * self.max_wz
        if active:
            t = Twist()
            t.linear.x, t.angular.z = vx, wz
            self.pub_cmd.publish(t)
            self.stop_burst = 4
        elif self.stop_burst > 0:                        # released: stop once, then stay quiet for Nav2
            self.pub_cmd.publish(Twist())
            self.stop_burst -= 1
        self.driving = active

    # ---- BLE side (asyncio thread) ------------------------------------------------------------
    async def _scan(self, timeout: float) -> list[dict]:
        found = await BleakScanner.discover(timeout=timeout, return_adv=True, adapter=self.adapter)
        devs, ble = [], {}
        for dev, adv in found.values():
            name = dev.name or adv.local_name or ""
            if name.startswith(NAME_PREFIX):
                devs.append({"address": dev.address.upper(), "name": name, "rssi": adv.rssi})
                ble[dev.address.upper()] = dev
        devs.sort(key=lambda d: -d["rssi"])
        self.devices, self.ble_devices, self.scan_time = devs, ble, time.monotonic()
        return devs

    async def _manual_scan(self) -> None:
        if self.busy:
            return
        self.busy = True
        prev = self.state
        self._set("scanning", "")
        try:
            devs = await self._scan(6.0)
            self._set(message=f"found {len(devs)} remote(s)" if devs else
                      "no Unitree remote found - switch it on (it advertises only when not connected)")
        except Exception as e:
            self._set(message=f"scan failed: {e}")
        finally:
            self._set("connected" if self.client is not None and prev == "connected" else "idle")
            self.busy = False

    async def _connect(self, address: str, name: str | None = None) -> bool:
        if self.busy:
            return False
        if self.client is not None:
            if self.conn["address"] == address:
                return True
            await self._disconnect()
        self.busy = True
        self._set("connecting", f"connecting to {name or address}")
        client = None
        try:
            # a scan result is only good while BlueZ still has the device object (it drops / we remove it)
            dev = self.ble_devices.get(address) if time.monotonic() - self.scan_time < 10.0 else None
            if dev is None:
                dev = await BleakScanner.find_device_by_address(address, timeout=8.0, adapter=self.adapter)
            if dev is None:
                raise RuntimeError("not advertising - is the remote on and not paired to another robot?")
            name = name or dev.name or address
            client = BleakClient(dev, disconnected_callback=self._on_disconnect, adapter=self.adapter)
            await client.connect(timeout=15.0)
            try:
                await client.write_gatt_char(WRITE_UUID, HANDSHAKE, response=False)
            except Exception:
                await client.write_gatt_char(WRITE_UUID, HANDSHAKE, response=True)
            await client.start_notify(NOTIFY_UUID, self._on_notify)
            self.client = client
            self.conn = {"address": address, "name": name}
            self.saved = {"address": address, "name": name}
            self._save()
            with self.lock:
                self.last_rx = time.monotonic()
            self.failures = 0
            self._set("connected", f"connected to {name} ({address})")
            return True
        except Exception as e:
            self.failures += 1
            self.ble_devices.pop(address, None)
            hint = (" - the adapter must be LE-only (ControllerMode = le in /etc/bluetooth/main.conf)"
                    if "br-connection" in str(e) else "")
            self._set("idle", f"connect to {address} failed: {e}{hint}")
            if client is not None:
                try:
                    await client.disconnect()
                except Exception:
                    pass
            if self.failures % 2 == 0:   # stale BlueZ device entries can block new LE connections
                subprocess.run(["bluetoothctl", "remove", address], capture_output=True, timeout=5, check=False)
            self.next_try = time.monotonic() + min(2.0 * self.failures, 10.0)
            return False
        finally:
            self.busy = False

    async def _disconnect(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.disconnect()
            except Exception:
                pass
        with self.lock:
            self.rs = None
        self._set("idle", "disconnected")

    def _on_disconnect(self, client: BleakClient) -> None:
        if client is not self.client:
            return
        self.client = None
        with self.lock:
            self.rs = None
        self.next_try = time.monotonic() + 2.0
        self._set("idle", f"{self.conn['name'] or 'remote'} disconnected")

    def _on_notify(self, _sender, data: bytearray) -> None:
        rs = decode(bytes(data))
        if rs is None:
            return
        with self.lock:
            prev = self.rs
            self.rs = rs
            self.last_rx = time.monotonic()
        for b in BUTTONS:
            if rs.buttons[b] and not (prev and prev.buttons[b]):
                self.pub_button.publish(String(data=b))
        joy = Joy()
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.header.frame_id = "unitree_remote"
        joy.axes = joy_axes(rs)
        joy.buttons = [int(rs.buttons[b]) for b in BUTTONS]
        self.pub_joy.publish(joy)

    async def _supervisor(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                if self.client is not None and time.monotonic() - self.last_rx > self.lost:
                    self._set(message="no data from the remote - dropping the link")
                    await self._disconnect()
                if (self.autoconnect and self.client is None and not self.busy
                        and time.monotonic() >= self.next_try):
                    await self._auto_attempt()
            except Exception as e:
                self.get_logger().warn(f"supervisor: {e}")

    async def _auto_attempt(self) -> None:
        self.busy = True
        self._set("scanning")
        try:
            devs = await self._scan(4.0)
        except Exception as e:
            self._set("idle", f"scan failed: {e}")
            self.next_try = time.monotonic() + 5.0
            return
        finally:
            self.busy = False
        want = self.saved.get("address")
        pick = next((d for d in devs if d["address"] == want), None) if want else (devs[0] if devs else None)
        if pick is None:
            self._set("idle", f"waiting for {self.saved.get('name') or want or 'a Unitree remote'}")
            self.next_try = time.monotonic() + 2.0
            return
        await self._connect(pick["address"], pick["name"])

    def shutdown(self) -> None:
        fut = self._run(self._disconnect())
        try:
            fut.result(timeout=3.0)
        except Exception:
            pass
        self.pub_cmd.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnitreeRemote()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
