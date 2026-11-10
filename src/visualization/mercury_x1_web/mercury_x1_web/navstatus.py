"""Where the navigation stack is: one row per layer for the page's Navigate tab (and the header chip).

  sensors       lidar and odometry rates, visual features in the camera frame (RTAB-Map's own count)
  slam          off / mapping / localizing, whether RTAB-Map is updating
  localization  localized (last map match, by camera or lidar) / uncertain / not localized
  nav2          off / starting / ready
  goal          idle / navigating / aborted ... (WebNode.nav)
  keepout       zones handed to Nav2's keep-out filter

Localization is judged from RTAB-Map's /info: every update says whether it matched the current frame against
the map (loop_closure_id = camera, proximity_detection_id = lidar scan near the current pose). Its
/localization_pose carries covariance 9999 until the first match. Nothing else in the stack says "lost":
the robot's pose on the map is then plain odometry from wherever it started, and looks just as valid.
"""
from __future__ import annotations

import math
import time
from collections import deque

INFO_STALE_S = 5.0          # RTAB-Map updates at ~1.5 Hz
SLAM_STARTUP_S = 40.0       # loading a map database takes a while
UNCERTAIN_AFTER_M = 4.0     # driven this far without a map match: the pose may have drifted
NOT_LOCALIZED_COV = 1000.0  # RTAB-Map publishes 9999 before its first match
ODOM_HOP_M = 0.10           # distance driven is counted in hops this long (ignores odometry jitter)


class Rate:
    def __init__(self, window: float = 3.0) -> None:
        self.window = window
        self.t: deque[float] = deque(maxlen=400)

    def tick(self) -> None:
        self.t.append(time.monotonic())

    def hz(self) -> float:
        now = time.monotonic()
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()
        return len(self.t) / self.window


class NavStatus:
    def __init__(self) -> None:
        self.scan = Rate()
        self.odom = Rate()
        self.odom_xy: tuple[float, float] | None = None
        self.reset()

    def reset(self) -> None:
        """a new SLAM run (started, restarted, other map)"""
        now = time.monotonic()
        self.start_t = now
        self.info_t = 0.0
        self.words: int | None = None
        self.nodes: int | None = None
        self.match_t: float | None = None
        self.match_kind = ""
        self.matches = 0
        self.dist_since_match = 0.0
        self.cov: float | None = None
        self.manual_t: float | None = None

    # ---- inputs ---------------------------------------------------------------------------------
    def on_info(self, msg) -> None:
        st = dict(zip(msg.stats_keys, msg.stats_values))
        self.info_t = time.monotonic()
        w = st.get("Keypoint/Current_frame/words")
        self.words = None if w is None else int(w)
        wm = st.get("Memory/Working_memory_size/")
        self.nodes = None if wm is None else int(wm + st.get("Memory/Short_time_memory_size/", 0))
        if msg.loop_closure_id > 0 or msg.proximity_detection_id > 0:
            self.match_t = self.info_t
            self.match_kind = "camera" if msg.loop_closure_id > 0 else "lidar"
            self.matches += 1
            self.dist_since_match = 0.0

    def on_localization_pose(self, msg) -> None:
        self.cov = float(msg.pose.covariance[0])

    def on_odom(self, msg) -> None:
        self.odom.tick()
        # distance in 10 cm hops from an anchor: lidar odometry jitters by millimetres per update even when the
        # robot stands still, and summing every step made a parked robot "drive" metres
        p = msg.pose.pose.position
        if self.odom_xy is None:
            self.odom_xy = (p.x, p.y)
            return
        d = math.hypot(p.x - self.odom_xy[0], p.y - self.odom_xy[1])
        if d >= ODOM_HOP_M:
            if d < 1.0:                       # odometry resets jump: ignore
                self.dist_since_match += d
            self.odom_xy = (p.x, p.y)

    def on_scan(self, _msg) -> None:
        self.scan.tick()

    def manual_pose(self) -> None:
        """the user told the localizer where the robot is ("Robot is here")"""
        self.manual_t = time.monotonic()
        self.dist_since_match = 0.0

    # ---- output ---------------------------------------------------------------------------------
    @staticmethod
    def _ago(s: float) -> str:
        return f"{s:.0f} s ago" if s < 90 else f"{s / 60:.0f} min ago"

    def snapshot(self, slam: dict, nav2_running: bool, nav2_ready: bool, nav: dict, keepout: int) -> dict:
        now = time.monotonic()
        rows = []

        def row(key, label, state, text, hint=""):
            rows.append({"key": key, "label": label, "state": state, "text": text, "hint": hint})

        # sensors
        scan_hz, odom_hz = self.scan.hz(), self.odom.hz()
        slam_on = bool(slam.get("running"))
        info_fresh = slam_on and now - self.info_t < INFO_STALE_S
        parts = [f"lidar {scan_hz:.0f} Hz", f"odometry {odom_hz:.0f} Hz"]
        dark = info_fresh and self.words == 0
        if info_fresh and self.words is not None:
            parts.append(f"camera {self.words} features")
        state = "bad" if scan_hz < 3 else "warn" if (slam_on and odom_hz < 3) or dark else "ok"
        hint = ("no lidar scans - check the lidar driver" if scan_hz < 3 else
                "no odometry - SLAM's lidar odometry is not running" if slam_on and odom_hz < 3 else
                "the camera sees no features (too dark?) - only the lidar can find the map, and only near the right pose"
                if dark else "")
        row("sensors", "Sensors", state, " · ".join(parts), hint)

        # SLAM
        mode, name = slam.get("mode"), slam.get("map", "")
        starting = slam_on and now - self.start_t < SLAM_STARTUP_S
        if not slam_on:
            row("slam", "SLAM", "off", "off", "▶ a map below to localize on it, or start a new map")
        elif not info_fresh:
            row("slam", "SLAM", "warn" if starting else "bad",
                f"{mode} {name} · " + ("starting" if starting else "RTAB-Map not updating"),
                "" if starting else "no update for a while - stop SLAM and start it again")
        else:
            row("slam", "SLAM", "ok" if mode == "localization" else "warn",
                ("localizing on " if mode == "localization" else "mapping ") + name
                + (f" · {self.nodes} nodes" if self.nodes else ""))

        # localization
        loc = "off"
        if not slam_on:
            row("loc", "Localization", "off", "off")
        elif mode == "mapping":
            loc = "mapping"
            last = f" · last loop closure {self._ago(now - self.match_t)}" if self.match_t else ""
            row("loc", "Localization", "ok", "mapping: the robot's position is the new map's own" + last)
        elif not info_fresh:
            loc = "unknown"
            row("loc", "Localization", "warn", "waiting for RTAB-Map")
        elif self.match_t is None:
            if self.manual_t is not None:
                loc = "manual"
                row("loc", "Localization", "warn",
                    f"pose set by hand {self._ago(now - self.manual_t)} · no map match yet",
                    "drive slowly for a metre or two: the lidar should match the map near that pose")
            else:
                loc = "no"
                row("loc", "Localization", "bad", "NOT localized · no map match since SLAM started",
                    "the position on the map is just odometry. Click the map where the robot really is "
                    "→ 📍 Robot is here, then drive a little")
        else:
            age = now - self.match_t
            text = f"last map match {self._ago(age)} ({self.match_kind}) · {self.dist_since_match:.1f} m driven since"
            if self.dist_since_match > UNCERTAIN_AFTER_M:
                loc = "uncertain"
                row("loc", "Localization", "warn", "uncertain · " + text,
                    "no map match for a while: the pose may have drifted. If the robot is off its walls, "
                    "use 📍 Robot is here")
            else:
                loc = "yes"
                row("loc", "Localization", "ok", "localized · " + text)

        # Nav2
        if not nav2_running:
            row("nav2", "Nav2", "off", "off", "start it under \"Nav2 and map file\" below to drive to goals")
        elif not nav2_ready:
            row("nav2", "Nav2", "warn", "starting (waiting for the navigator)")
        else:
            warn = loc in ("no", "manual", "uncertain", "unknown")
            row("nav2", "Nav2", "warn" if warn else "ok",
                "ready" + (" · but the robot is not reliably localized" if warn else ""))

        # goal
        st = nav.get("state", "idle")
        gstate = {"navigating": "warn", "error": "bad", "aborted": "bad"}.get(st, "ok" if nav2_ready else "off")
        text = st + (" · cruising" if nav.get("loop") else "") + (f" · {nav['message']}" if nav.get("message") else "")
        row("goal", "Goal", gstate, text)

        # keep-out
        row("keepout", "Keep-out", "ok" if keepout else "off",
            f"{keepout} zone{'s' if keepout != 1 else ''}" + (" · in Nav2's costmaps" if keepout and nav2_running else ""),
            "" if keepout else "draw glass walls with ⛔ Keep-out zone before driving")
        return {"rows": rows, "localized": loc}
