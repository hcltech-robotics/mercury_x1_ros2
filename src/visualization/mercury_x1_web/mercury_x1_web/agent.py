"""Agent brain: the network vision-language model (pennyroyal = Qwen on 192.168.11.51, voice.env VOICE_LLM_URL)
drives the robot through tools until a task is done: "find the sink", "find a person and tell them lunch is
ready", "go and check whether the door is open".

The model gets an annotated picture of the 2D map (1 m grid in map coordinates, the robot, the named places,
keep-out zones) and the head camera when it looks around; it answers with tool calls, one step at a time:

  get_map             the map picture again (after driving: where am I now)
  look(direction)     turn the head front / left / right / down, return the camera picture
  go_to(place | x, y) Nav2 goal, waits for the result; a point inside an obstacle moves to the nearest free spot
  turn(degrees)       turn the base in place (to look behind)
  say(text)           speak through the robot's speaker
  listen(seconds)     record the mic beam and transcribe it (an answer from a person)
  gesture(name)       wave, nod, thank_you, ... (gestures.py)
  remember_place(name) save the robot's spot as a named place (it found the sink: next time it drives there)
  finish(answer)      done; the answer is shown on the page (and spoken when the task came by voice)

Started from the page's Agent tab or by the voice chat's do_task tool; Stop (page, voice "stop", the Stop
button) cancels the model loop and the Nav2 goal. Pictures stay in memory (last MAX_IMAGES) for the page log;
the model sees the last KEEP_IMAGES of them, older ones are replaced by a note to keep the prompt small."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import re
import time

import aiohttp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .voice import LOCAL_SPEECH, load_conf, network_llms

MAX_STEPS = 40
MAX_S = 900.0              # whole task
NAV_TIMEOUT_S = 120.0
STUCK_S = 30.0             # no 0.3 m of travel and no 20 deg of turn for this long: cancel, report "stuck"
KEEP_IMAGES = 3            # pictures the model still sees
MAX_IMAGES = 30            # pictures kept for the page
HEAD_DEG = {"front": 0.0, "left": 60.0, "right": -60.0}
EYE_DOWN_DEG = 20.0        # "down": head pitch ("eye" + = down, as in the nod)
SAY_SENTENCES = 2          # spoken per say call (the rest is dropped; the model is told)
CLEARANCE_M = 0.35         # a go_to point closer than this to an obstacle moves away from it

SYSTEM = """You are the planning brain of Mercury, a wheeled humanoid robot (two arms, a head with a colour camera)
working in an office lab. You get a task from a person and complete it by calling tools, one step at a time.

The map: a top-down picture of the floor. White = free floor, black = walls and furniture, grey = never seen.
Map coordinates are in metres; the blue grid has one line per metre and labels on the edges. The robot is the
green circle with an arrow (its heading). Purple dots are named places, red areas are keep-out zones.

How to work:
- Before each tool call, write one short sentence of what you think and why.
- If a named place matches the task, go there first. Otherwise look around (left, front, right) before driving.
- To search, pick points in the free (white) areas of the map, in rooms or corners you have not seen yet, and go
  there; then look around. Prefer points 1 to 4 m away. Never pick points in grey or black areas.
- The camera sees about 60 degrees wide and objects up to a few metres away. Look at each picture carefully.
- When you find what you were asked for, drive close to it (about 1 m), look to confirm, and call remember_place
  with a short name (e.g. "sink") so it is known next time.
- To tell a person something: go near them, say it, and if a reply is needed, listen.
- Everything you say is spoken aloud: at most two short sentences per say call. Details go in finish.
- If something fails twice, try another way. If the task is impossible, say why in finish.
- Finish with finish(answer): one or two sentences about what you did and found."""


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _jpeg(img: Image.Image, q: int = 80) -> bytes:
    b = io.BytesIO()
    img.convert("RGB").save(b, "JPEG", quality=q)
    return b.getvalue()


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Agent:
    def __init__(self, app) -> None:
        self.app, self.node = app, app.node
        self.state, self.task, self.answer, self.source = "idle", "", "", ""
        self.log: list[dict] = []
        self.images: dict[int, bytes] = {}
        self._img_id = 0
        self._job: asyncio.Task | None = None
        self._frame: tuple[float, bytes] | None = None      # (time, jpeg) from /preview/orbbec
        self.t0 = 0.0
        self._eye_home: float | None = None
        self.node.post(self._subscribe)

    # -- camera ------------------------------------------------------------------------------------
    def _subscribe(self) -> None:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        self.node.create_subscription(CompressedImage, "/preview/orbbec",
                                      lambda m: setattr(self, "_frame", (time.time(), bytes(m.data))), qos_profile_sensor_data)

    async def _fresh_frame(self, after: float, timeout: float = 3.0) -> bytes | None:
        t = time.time()
        while time.time() - t < timeout:
            f = self._frame
            if f and f[0] > after:
                return f[1]
            await asyncio.sleep(0.1)
        return self._frame[1] if self._frame else None

    # -- page --------------------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"state": self.state, "task": self.task, "answer": self.answer, "source": self.source,
                "elapsed": round(time.time() - self.t0, 1) if self.t0 else 0, "log": self.log[-80:]}

    def _add(self, kind: str, text: str = "", img: bytes | None = None) -> None:
        e = {"t": round(time.time() - self.t0, 1), "kind": kind, "text": text[:600]}
        if img is not None:
            self._img_id += 1
            self.images[self._img_id] = img
            for k in sorted(self.images)[:-MAX_IMAGES]:
                del self.images[k]
            e["img"] = self._img_id
        self.log.append(e)
        self.log = self.log[-200:]

    # -- control -----------------------------------------------------------------------------------
    def start(self, task: str, source: str = "page") -> tuple[bool, str]:
        task = task.strip()
        if not task:
            return False, "no task"
        if self._job and not self._job.done():
            return False, f"busy with: {self.task}"
        if not network_llms(load_conf()):
            return False, "no network model configured (VOICE_LLM_URL in voice.env)"
        self.task, self.answer, self.source, self.log, self.t0 = task, "", source, [], time.time()
        self.state, self._eye_home = "running", None
        self._job = asyncio.create_task(self._run())
        return True, "started"

    def stop(self) -> None:
        if self._job and not self._job.done():
            self._job.cancel()
        self.node.post(self.node.cancel)

    # -- the loop ----------------------------------------------------------------------------------
    async def _run(self) -> None:
        conf = load_conf()
        srv = network_llms(conf)[0]
        sel = self.app.voice.settings.get("llm_server")
        srv = next((s for s in network_llms(conf) if s["id"] == sel), srv)
        self._add("task", f"{self.task}  (model: {srv['name']})")
        mimg, mtext = self.map_image()
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": f"Task: {self.task}\n\n{self.status_text()}\n{mtext}"}]
                 + ([self._img_part(mimg)] if mimg else [])}]
        if mimg:
            self._add("image", "map", mimg)
        try:
            async with aiohttp.ClientSession(headers={"Authorization": f"Bearer {srv['key']}"}) as http:
                for step in range(MAX_STEPS):
                    if time.time() - self.t0 > MAX_S:
                        raise RuntimeError(f"out of time ({MAX_S:.0f} s)")
                    self._trim_images(msgs)
                    body = {"model": srv["model"], "messages": msgs, "tools": TOOLS, "max_tokens": 700, "temperature": 0.2,
                            "chat_template_kwargs": {"enable_thinking": False}}
                    async with http.post(srv["url"], json=body, timeout=aiohttp.ClientTimeout(total=90)) as r:
                        if r.status != 200:
                            raise RuntimeError(f"model: HTTP {r.status} {(await r.text())[:160]}")
                        msg = (await r.json())["choices"][0]["message"]
                    text = (msg.get("content") or "").strip()
                    calls = msg.get("tool_calls") or []
                    if text:
                        self._add("thought", text)
                    msgs.append({"role": "assistant", "content": text or None,
                                 **({"tool_calls": calls} if calls else {})})
                    if not calls:     # no tool: nudge once, then treat the text as the answer
                        if step and msgs[-2].get("role") == "user" and "call a tool" in str(msgs[-2].get("content")):
                            self.answer = text
                            break
                        msgs.append({"role": "user", "content": "Continue: call a tool (finish when done)."})
                        continue
                    done = False
                    for c in calls:
                        name = c["function"]["name"]
                        try:
                            args = json.loads(c["function"].get("arguments") or "{}")
                        except ValueError:
                            args = {}
                        self._add("tool", f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                        result, img = await self._call(name, args)
                        self._add("result", json.dumps(result, ensure_ascii=False)[:400], img)
                        msgs.append({"role": "tool", "tool_call_id": c.get("id") or name, "content": json.dumps(result, ensure_ascii=False)})
                        if img is not None:
                            msgs.append({"role": "user", "content": [{"type": "text", "text": f"Picture from {name}:"}, self._img_part(img)]})
                        if name == "finish":
                            self.answer, done = str(args.get("answer", "")), True
                    if done:
                        break
                else:
                    self.answer = self.answer or f"stopped after {MAX_STEPS} steps"
            self.state = "done"
            self._add("done", self.answer)
            if self.source == "voice" and self.answer:
                await self.app.voice.announce(" ".join(re.split(r"(?<=[.!?])\s+", self.answer)[:SAY_SENTENCES]))
        except asyncio.CancelledError:
            self.state = "stopped"
            self._add("done", "stopped")
            self.node.post(self.node.cancel)
        except Exception as e:  # noqa: BLE001 - shown on the page
            self.state = "error"
            self._add("error", str(e)[:300] or type(e).__name__)
            self.node.post(self.node.cancel)
        finally:
            self._head(0.0)

    @staticmethod
    def _img_part(jpeg: bytes) -> dict:
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}

    @staticmethod
    def _trim_images(msgs: list[dict]) -> None:
        """only the last KEEP_IMAGES pictures stay in the prompt"""
        seen = 0
        for m in reversed(msgs):
            if isinstance(m.get("content"), list):
                for k, part in enumerate(m["content"]):
                    if part.get("type") == "image_url":
                        seen += 1
                        if seen > KEEP_IMAGES:
                            m["content"][k] = {"type": "text", "text": "[older picture removed]"}

    # -- tools -------------------------------------------------------------------------------------
    async def _call(self, name: str, a: dict) -> tuple[dict, bytes | None]:
        try:
            if name == "get_map":
                img, text = self.map_image()
                return {"ok": img is not None, "map": text, "robot": self.status_text()}, img
            if name == "look":
                return await self.look(str(a.get("direction", "front")))
            if name == "go_to":
                return await self.go_to(a), None
            if name == "turn":
                return await self.turn(float(a.get("degrees", 0))), None
            if name == "say":
                text = " ".join(re.split(r"(?<=[.!?])\s+", str(a.get("text", "")).strip())[:SAY_SENTENCES])
                await self.app.voice.announce(text)
                return {"ok": True, "said": text}, None
            if name == "listen":
                return await self.listen(float(a.get("seconds", 5))), None
            if name == "gesture":
                ok, msg = self.app.gestures.run(str(a.get("name", "")))
                if ok:
                    await asyncio.sleep(0.5)
                    while self.app.gestures.busy:
                        await asyncio.sleep(0.3)
                return {"ok": ok, "message": msg}, None
            if name == "remember_place":
                err = self.app.place_save(str(a.get("name", "")))
                return {"ok": not err, "error": err} if err else {"ok": True, "saved": str(a.get("name"))}, None
            if name == "finish":
                return {"ok": True}, None
            return {"ok": False, "error": f"unknown tool {name}"}, None
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the model sees the error and can try something else
            return {"ok": False, "error": str(e)[:200] or type(e).__name__}, None

    def status_text(self) -> str:
        p = self.node.robot_pose()
        pose = (f"The robot is at x={p['x']:.1f}, y={p['y']:.1f} m, heading {math.degrees(p['yaw']):.0f} degrees "
                f"(0 = +x, 90 = +y)." if p and p["frame"] == "map" else "The robot is not localized on the map.")
        places = ", ".join(f"{q['name']} ({q['x']:.1f}, {q['y']:.1f})" for q in self.node.places) or "none"
        return f"{pose} Named places: {places}. Battery {self.node.battery_pct:.0f} %." if math.isfinite(self.node.battery_pct) \
            else f"{pose} Named places: {places}."

    def _head(self, yaw_deg: float, eye_deg: float | None = None) -> None:
        from mercury_x1_msgs.msg import JointCommand
        names, pos = ["head"], [math.radians(yaw_deg)]
        if eye_deg is not None:
            names.append("eye")
            pos.append(math.radians(eye_deg))
        self.node.pub_joint.publish(JointCommand(name=names, position=pos, speed=25))

    async def look(self, direction: str) -> tuple[dict, bytes | None]:
        direction = direction.lower().strip()
        eye_now = self.node.body_joints.get("eye", 0.0)          # degrees
        if direction not in HEAD_DEG and direction != "down":
            return {"ok": False, "error": "direction: front, left, right or down"}, None
        if self._eye_home is None:            # the head pitch when the task started = level enough to look around
            self._eye_home = eye_now
        if direction == "down":
            self._head(0.0, self._eye_home + EYE_DOWN_DEG)
        else:
            self._head(HEAD_DEG[direction], self._eye_home)
        t = time.time()
        await asyncio.sleep(1.2)                                  # the head moves, then a fresh frame
        img = await self._fresh_frame(t + 1.0)
        if direction == "down":
            self._head(0.0, self._eye_home)
        if img is None:
            return {"ok": False, "error": "no camera picture (is the Orbbec running?)"}, None
        p = self.node.robot_pose()
        facing = (math.degrees(p["yaw"]) + HEAD_DEG.get(direction, 0.0)) if p else None
        return {"ok": True, "looking": direction,
                "camera_heading_deg": None if facing is None else round(_wrap(math.radians(facing)) * 180 / math.pi)}, img

    async def go_to(self, a: dict) -> dict:
        n = self.node
        if not self.app.sup.status()["nav"]["running"]:
            return {"ok": False, "error": "navigation (Nav2) is not running"}
        p = n.robot_pose()
        if p is None or p["frame"] != "map":
            return {"ok": False, "error": "the robot is not localized on the map"}
        from . import places as placelib
        q = placelib.resolve(str(a["place"]), n.places) if a.get("place") else None
        if a.get("place") and q is None and a.get("x") is None:
            return {"ok": False, "error": f"no place called {a['place']!r}; known: {[x['name'] for x in n.places]}"}
        if q is not None:
            gx, gy, gyaw, label = q["x"], q["y"], q["yaw"], q["name"]
        else:
            try:
                gx, gy = float(a["x"]), float(a["y"])
            except (KeyError, TypeError, ValueError):
                return {"ok": False, "error": "give a place name or x and y in metres"}
            fx, fy, note = self.free_spot(gx, gy)
            if fx is None:
                return {"ok": False, "error": note}
            gx, gy = fx, fy
            gyaw = math.atan2(gy - p["y"], gx - p["x"])
            label = f"({gx:.1f}, {gy:.1f})" + (f" ({note})" if note else "")
        self._head(0.0)
        t = time.time()
        n.post(lambda: n.go_to(gx, gy, gyaw))
        await asyncio.sleep(1.0)
        ref, ref_t, state = n.robot_pose() or p, time.time(), ""
        while n.nav["state"] == "navigating":
            await asyncio.sleep(0.5)
            q = n.robot_pose() or ref
            if math.hypot(q["x"] - ref["x"], q["y"] - ref["y"]) > 0.3 or abs(_wrap(q["yaw"] - ref["yaw"])) > math.radians(20):
                ref, ref_t = q, time.time()
            if time.time() - ref_t > STUCK_S:
                state = "stuck (no movement for 30 s)"
                break
            if time.time() - t > NAV_TIMEOUT_S:
                state = "timed out"
                break
        if state:
            n.post(n.cancel)
        else:
            state = n.nav["state"]
        q = n.robot_pose() or p
        left = math.hypot(gx - q["x"], gy - q["y"])
        if state != "succeeded" and left < 0.6:
            state = "succeeded (close enough)"
        return {"ok": state.startswith("succeeded"), "result": state, "goal": label, "distance_left_m": round(left, 2),
                "seconds": round(time.time() - t), "now": self.status_text(), **({"message": n.nav["message"]} if n.nav.get("message") else {})}

    def free_spot(self, x: float, y: float):
        """(x, y, note) of the nearest free map cell at least CLEARANCE_M from obstacles, within 1.5 m"""
        m = self.node.map_msg
        if m is None:
            return None, None, "no map"
        i = m.info
        res, ox, oy, w, h = i.resolution, i.origin.position.x, i.origin.position.y, i.width, i.height
        grid = np.frombuffer(np.array(m.data, np.int8).tobytes(), np.int8).reshape(h, w)
        cx, cy = int((x - ox) / res), int((y - oy) / res)
        r = int(1.5 / res)
        x0, x1, y0, y1 = max(0, cx - r), min(w, cx + r + 1), max(0, cy - r), min(h, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return None, None, f"({x:.1f}, {y:.1f}) is outside the map"
        win = grid[y0:y1, x0:x1]
        occ = np.argwhere((win > 50) | (win < 0))          # walls and unknown count as obstacles
        free = np.argwhere((win >= 0) & (win <= 20))
        if not len(free):
            return None, None, f"no free floor near ({x:.1f}, {y:.1f})"
        if len(occ):
            d = np.sqrt(((free[:, None, :] - occ[None, :, :]) ** 2).sum(-1)).min(1) * res if len(free) * len(occ) < 4e6 \
                else np.full(len(free), CLEARANCE_M)
            free = free[d >= CLEARANCE_M]
        if not len(free):
            return None, None, f"no free floor with {CLEARANCE_M} m clearance near ({x:.1f}, {y:.1f})"
        k = np.argmin((free[:, 0] + y0 - cy) ** 2 + (free[:, 1] + x0 - cx) ** 2)
        fy, fx = free[k]
        gx, gy = ox + (fx + x0 + 0.5) * res, oy + (fy + y0 + 0.5) * res
        moved = math.hypot(gx - x, gy - y)
        return gx, gy, (f"moved {moved:.1f} m to free floor" if moved > 0.15 else "")

    async def turn(self, degrees: float) -> dict:
        n = self.node
        p = n.robot_pose()
        if p is None:
            return {"ok": False, "error": "no robot pose"}
        degrees = max(-360.0, min(360.0, degrees))
        n.post(lambda: n.cancel(stop_base=False))
        target, turned, last = math.radians(degrees), 0.0, p["yaw"]
        t = time.time()
        try:
            while time.time() - t < 30.0:
                q = n.robot_pose() or p
                turned += _wrap(q["yaw"] - last)
                last = q["yaw"]
                err = target - turned
                if abs(err) < math.radians(4):
                    break
                wz = math.copysign(min(0.8, max(0.35, 1.5 * abs(err))), err)
                n.post(lambda wz=wz: n.cmd_vel(0.0, wz))
                await asyncio.sleep(0.1)
        finally:
            n.post(lambda: n.cmd_vel(0.0, 0.0))
        return {"ok": abs(target - turned) < math.radians(10), "turned_deg": round(math.degrees(turned)), "now": self.status_text()}

    async def listen(self, seconds: float) -> dict:
        seconds = max(1.0, min(8.0, seconds))
        proc = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", "mx1_beam", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw", "-d", str(int(round(seconds))),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        pcm, _ = await proc.communicate()
        if len(pcm) < 8000:
            return {"ok": False, "error": "no audio from the microphone"}
        url = load_conf().get("VOICE_SPEECH_URL", LOCAL_SPEECH).rstrip("/") + "/stt"
        async with aiohttp.ClientSession() as s, s.post(url, data=pcm, timeout=aiohttp.ClientTimeout(total=15)) as r:
            text = str((await r.json()).get("text", "")).strip()
        return {"ok": True, "heard": text or "(nothing)"}

    # -- the map picture ---------------------------------------------------------------------------
    def map_image(self) -> tuple[bytes | None, str]:
        m = self.node.map_msg
        if m is None:
            return None, "No map yet."
        i = m.info
        res, ox, oy, w, h = i.resolution, i.origin.position.x, i.origin.position.y, i.width, i.height
        g = np.frombuffer(np.array(m.data, np.int8).tobytes(), np.int8).reshape(h, w)
        known = np.argwhere(g >= 0)
        if not len(known):
            return None, "The map is empty."
        pad = int(1.0 / res)
        r0, c0 = np.maximum(known.min(0) - pad, 0)
        r1, c1 = np.minimum(known.max(0) + pad + 1, [h, w])
        g = g[r0:r1, c0:c1]
        rgb = np.full(g.shape + (3,), 150, np.uint8)
        rgb[(g >= 0) & (g <= 25)] = 255
        rgb[g > 65] = 0
        rgb[(g > 25) & (g <= 65)] = 200
        img = Image.fromarray(rgb[::-1])                       # row 0 = highest y
        x0, y0 = ox + c0 * res, oy + r0 * res
        x1, y1 = x0 + g.shape[1] * res, y0 + g.shape[0] * res
        S = min(1100 / img.width, 1100 / img.height, 8.0)
        M = 60                                                  # margin for the labels
        img = img.resize((int(img.width * S), int(img.height * S)), Image.NEAREST)
        canvas = Image.new("RGB", (img.width + 2 * M, img.height + 2 * M), (255, 255, 255))
        canvas.paste(img, (M, M))
        d = ImageDraw.Draw(canvas, "RGBA")
        px = S / res                                            # pixels per metre

        def to(x, y):
            return M + (x - x0) * px, M + (y1 - y) * px
        f, fb = _font(16), _font(18)
        step = 1 if (x1 - x0) * (y1 - y0) < 900 else 2
        for gx in range(math.ceil(x0), math.floor(x1) + 1):
            X, _ = to(gx, y0)
            d.line([(X, M), (X, M + img.height)], fill=(60, 120, 255, 110 if gx % 5 else 200), width=1)
            if gx % step == 0:
                d.text((X - 8, M - 22), str(gx), fill=(20, 60, 200), font=f)
                d.text((X - 8, M + img.height + 4), str(gx), fill=(20, 60, 200), font=f)
        for gy in range(math.ceil(y0), math.floor(y1) + 1):
            _, Y = to(x0, gy)
            d.line([(M, Y), (M + img.width, Y)], fill=(60, 120, 255, 110 if gy % 5 else 200), width=1)
            if gy % step == 0:
                d.text((M - 34, Y - 9), str(gy), fill=(20, 60, 200), font=f)
                d.text((M + img.width + 6, Y - 9), str(gy), fill=(20, 60, 200), font=f)
        d.text((M, 4), "x (m) →", fill=(20, 60, 200), font=f)
        d.text((4, M + img.height + 28), "y (m) ↑", fill=(20, 60, 200), font=f)
        for z in self.node.keepout:
            pts = [to(*p) for p in z["points"]]
            if len(pts) == 2:
                d.line(pts, fill=(255, 0, 0, 220), width=max(3, int(0.1 * px)))
            else:
                d.polygon(pts, fill=(255, 0, 0, 90), outline=(255, 0, 0, 255))
        for q in self.node.places:
            X, Y = to(q["x"], q["y"])
            d.ellipse([X - 7, Y - 7, X + 7, Y + 7], fill=(150, 60, 220))
            d.text((X + 10, Y - 10), q["name"], fill=(110, 30, 180), font=fb, stroke_width=3, stroke_fill=(255, 255, 255))
        p = self.node.robot_pose()
        if p and p["frame"] == "map":
            X, Y = to(p["x"], p["y"])
            rr = max(8, 0.3 * px)
            d.ellipse([X - rr, Y - rr, X + rr, Y + rr], fill=(0, 200, 80, 150), outline=(0, 130, 50), width=3)
            L = max(24, 0.8 * px)
            d.line([(X, Y), (X + L * math.cos(p["yaw"]), Y - L * math.sin(p["yaw"]))], fill=(0, 130, 50), width=5)
        text = (f"Map picture: x from {x0:.1f} to {x1:.1f} m (left to right), y from {y0:.1f} to {y1:.1f} m "
                f"(bottom to top), one grid line per metre.")
        return _jpeg(canvas, 85), text


def _fn(name: str, desc: str, props: dict | None = None, req: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props or {}, "required": req or []}}}


TOOLS = [
    _fn("get_map", "Get the map picture again with the robot's current position."),
    _fn("look", "Turn the head and take a camera picture.",
        {"direction": {"type": "string", "enum": ["front", "left", "right", "down"]}}, ["direction"]),
    _fn("go_to", "Drive to a named place, or to map coordinates x, y in metres (free floor). Waits until arrived.",
        {"place": {"type": "string", "description": "a named place"}, "x": {"type": "number"}, "y": {"type": "number"}}),
    _fn("turn", "Turn the robot in place; positive = left (counter-clockwise).",
        {"degrees": {"type": "number"}}, ["degrees"]),
    _fn("say", "Say something out loud (short, plain words).", {"text": {"type": "string"}}, ["text"]),
    _fn("listen", "Record what a person says for a few seconds and return the transcript.",
        {"seconds": {"type": "number", "description": "1 to 8"}}, ["seconds"]),
    _fn("gesture", "Do a gesture: nod, shake_head, handshake, thank_you (Japanese bow), wave, high_five, point, "
        "welcome, shrug, look_around.", {"name": {"type": "string"}}, ["name"]),
    _fn("remember_place", "Save the robot's current spot as a named place (something you found).",
        {"name": {"type": "string"}}, ["name"]),
    _fn("finish", "The task is done (or impossible).", {"answer": {"type": "string"}}, ["answer"]),
]
