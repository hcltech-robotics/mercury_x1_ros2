"""Hands-free voice chat for the head display: mic -> speech (robot) -> Qwen (11.51) -> speech -> speaker.

Runs on web_server's asyncio loop (all I/O, no threads):
  * `arecord` streams the XFM-DP mic array (VOICE_MIC, `mx1_beam` = the board's beam, 16 kHz mono) to the robot's own speech
    service (`mercury-speech`, server/robot_speech, 127.0.0.1:8097, VOICE_SPEECH_URL): /listen finds the
    phrases (Silero VAD) and transcribes them (Parakeet, CPU). No audio leaves the robot; only text goes to Qwen.
  * Each phrase goes to Qwen (the pennyroyal proxy, streamed, thinking off); the reply is cut into
    sentences, each one is synthesised by Kokoro (/tts) and played with `aplay` on the C-Media speaker
    while the next is being made (Kokoro on the robot's GPU, same service: /tts).
  * The mic is muted while the robot talks (no echo loop) and for a short tail after it.
  * Knowledge: every phrase is looked up in the local knowledge files (knowledge.py, BM25) and the best
    paragraphs go into the prompt. Actions: the model may call tools (robot status, named places, drive to a
    place, stop, gestures); driving and gestures only when switched on in the settings (kiosk Voice tab).
  * Language model: the lab server (VOICE_LLM_URL) or the robot's own (llama-server on the Jetson,
    VOICE_LOCAL_LLM_URL, server/robot_llm). Setting "llm": auto (lab, the robot when the lab does not answer),
    lab, or robot (fully on board).
Config: ~/.config/mercury_x1/voice.env (model URL + bearer token, ALSA devices; chmod 600).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from urllib.parse import urlparse

import aiohttp

from .knowledge import Knowledge, as_prompt

CONF = os.path.expanduser("~/.config/mercury_x1/voice.env")
SETTINGS_FILE = os.path.expanduser("~/.config/mercury_x1/voice_settings.json")
VOICES = {"af_heart": "Heart (US)", "af_bella": "Bella (US)", "am_michael": "Michael (US)",
          "bf_emma": "Emma (UK)", "bm_george": "George (UK)", "am_puck": "Puck (US)"}
DEFAULT_SETTINGS = {"volume": 46, "voice": "af_heart", "speed": 1.0,
                    "drive": False,       # may drive to named places when asked (Nav2); off until tested
                    "gestures": False,    # may move waist / neck / arms for gestures; off until tested
                    "llm": "auto",        # auto: network model, robot's own when it does not answer | lab (network) | robot
                    "llm_server": "1",    # which network model (network_llms id)
                    "wake_listen": True,  # "hey mercury" takes one question (when not already listening)
                    "wake_verify": True}  # ... after the speech recognizer confirmed it heard "mercury" (wake_verify.py)
TEST_PHRASE = "Hello, I am Mercury. This is how loud I speak."
SYSTEM_PROMPT = (
    "You are Mercury, a friendly wheeled humanoid robot with two arms, made by Elephant Robotics and working in the "
    "HCLTech robotics lab as a lab companion. You are talking out loud with people standing in front of you, so "
    "answer in at most three short spoken sentences (never more than four), in plain words: no lists, no markdown, no emojis, no code. Say "
    "numbers the way a person would. If you did not understand, ask briefly. For questions about HCLTech, the lab or "
    "yourself, answer only from the facts you are given: copy names and numbers exactly, say the period of a figure, "
    "prefer the most recent date, and if the facts do not say it, say you do not know. Call a tool only when the "
    "person asks for what it does (your status, places, driving somewhere, stopping, a gesture). You move only "
    "through tools: never say you are moving or doing something unless a tool result confirms it, and if no tool "
    "fits, say you cannot do that yet.")
# the robot's own (small) model needs the tool rules spelled out; appended to its system prompt only
LOCAL_RULES = (
    " Rules: for your battery, the time, the date or where you are, call get_robot_status and say what it returns; "
    "never guess a number. To go somewhere or show the way, call go_to_place. To nod, shake your head or shake hands, "
    "call gesture. When someone says stop or wait, call stop_moving. Never say you are moving, going or doing a "
    "gesture unless a tool result in this conversation says it worked; if it says something is switched off, say so. "
    "General knowledge questions that are not about HCLTech, the lab or you, answer briefly from what you know.")
MAX_TOOL_ROUNDS = 3
HISTORY_TURNS = 8          # user + assistant pairs kept as context
IDLE_RESET_S = 180         # forget the conversation after this long without speech
WAKE_ASK_S = 8.0           # after "hey mercury": back to waiting for the wake word if no question starts within this
# the wake word itself as speech-to-text writes it ("Hey, Mark Jury", "Pay marks"): stripped from the start of a question
WAKE_ECHO = re.compile(r"^\W*(?:hey|hay|hi|hello|ok(?:ay)?|pay)\W+(?:mercur\w*|merc\w*|mark\W*jury|marks?)\b\W*", re.I)
UNMUTE_TAIL_S = 0.35       # keep the mic muted this long after the speaker stops
LOCAL_SPEECH = "http://127.0.0.1:8097"   # server/robot_speech: Parakeet STT + Kokoro TTS
LOCAL_LLM = "http://127.0.0.1:8091/v1/chat/completions"   # server/robot_llm: llama-server on the Jetson
LLM_MODES = ("auto", "lab", "robot")
LLM_ENV = os.path.expanduser("~/.config/mercury_x1/llm.env")   # server/robot_llm: which model the robot runs
LLM_CHECK_MIN_S = 5.0      # reachability checks for the page: at most this often
LAB_HEADERS_S = 3.0        # auto: the lab server must start answering within this, else the robot's model answers
LAB_RETRY_S = 60.0         # auto: after a lab failure, ask the robot's model first for this long
MAX_SENTENCES = 4          # spoken answer cap (user, 2026-11-09: "not more than 3-4 sentences"); the rest is dropped
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")
# a short "stop" phrase cancels motion directly, before (and whatever) the model answers
STOP_PHRASE = re.compile(r"\b(stop|halt|freeze|wait|stay|hold on|don'?t move|stand still)\b", re.I)
FIRST_CLAUSE = re.compile(r"^((?:\S+\s+){3,}?\S+,)\s+")   # a reply's opening clause of 4+ words, up to its comma


def load_conf() -> dict:
    conf = {}
    try:
        with open(CONF) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    conf[k.strip()] = v.strip()
    except OSError:
        pass
    return conf


def network_llms(conf: dict) -> list[dict]:
    """language models on the local network, from voice.env: VOICE_LLM_URL (+ VOICE_LLM_MODEL, VOICE_LLM_NAME,
    VOICE_LLM_KEY) and more as VOICE_LLM_URL_2 .. VOICE_LLM_URL_9 with the same suffix. Key defaults to VOICE_API_KEY."""
    out = []
    for n in range(1, 10):
        sfx = "" if n == 1 else f"_{n}"
        url = conf.get(f"VOICE_LLM_URL{sfx}", "")
        if not url:
            continue
        model = conf.get(f"VOICE_LLM_MODEL{sfx}", "pennyroyal" if n == 1 else "")
        out.append({"id": str(n), "name": conf.get(f"VOICE_LLM_NAME{sfx}") or model or urlparse(url).hostname or url,
                    "url": url, "model": model or "default", "key": conf.get(f"VOICE_LLM_KEY{sfx}", conf.get("VOICE_API_KEY", ""))})
    return out


def robot_llm_name() -> str:
    """the robot's own model, as server/robot_llm runs it (llm.env LLM_MODEL, else its default)"""
    name = "Qwen3.5-4B-Q4_K_M.gguf"
    try:
        with open(LLM_ENV) as f:
            for line in f:
                if line.strip().startswith("LLM_MODEL="):
                    name = line.split("=", 1)[1].strip().strip("\"'") or name
    except OSError:
        pass
    return re.sub(r"(-Q\d\w*)?\.gguf$", "", os.path.basename(name))


def speakable(text: str) -> str:
    """strip what TTS would read out literally (markdown, emoji, URLs, stray think tags)"""
    text = re.sub(r"</?think>?", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~\[\]]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class VoiceChat:
    def __init__(self, on_change=None, actions=None) -> None:
        self.on_change = on_change or (lambda: None)
        self.actions = actions        # web_server App: voice_places / voice_go / voice_stop / voice_status / voice_gesture
        self.kb = Knowledge()
        self.enabled = False
        self.wake_turn = False     # listening for the one question after "hey mercury" (not the Listen switch)
        self._wake_t = 0.0         # wake word or last speech activity during a wake turn
        self.state = "off"        # off | connecting | listening | hearing | thinking | speaking | error
        self.message = ""
        self.transcript: list[dict] = []      # {"role": "user"|"robot"|"note", "text", "t"}
        self.history: list[dict] = []         # chat messages for the model
        self.last_speech = 0.0
        self.stats: dict = {}
        self._tasks: list[asyncio.Task] = []
        self._reply_task: asyncio.Task | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._session: aiohttp.ClientSession | None = None
        self._arecord: asyncio.subprocess.Process | None = None
        self._aplay: asyncio.subprocess.Process | None = None
        self._muted = False
        self.settings = self._load_settings()
        self.llm_health: dict = {}     # "robot" / network id -> {"ok", "ms" | "error"}
        self._llm_checked = 0.0
        self._vol_applied = False

    # ---- settings: speaker volume (ALSA hardware PCM control), voice, speed; saved ----------
    @staticmethod
    def _load_settings() -> dict:
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            saved = {}
        st = {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}
        if st["voice"] not in VOICES:
            st["voice"] = DEFAULT_SETTINGS["voice"]
        if st["llm"] not in LLM_MODES:
            st["llm"] = DEFAULT_SETTINGS["llm"]
        return st

    @staticmethod
    def _speaker_card(conf: dict) -> str:
        m = re.search(r"CARD=([^,]+)", conf.get("VOICE_SPEAKER", ""))
        return m.group(1) if m else "Device"

    async def apply_volume(self) -> None:
        card = self._speaker_card(load_conf())
        p = await asyncio.create_subprocess_exec("amixer", "-q", "-c", card, "sset", "PCM", f"{self.settings['volume']}%",
                                                 stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        err = (await p.communicate())[1]
        if p.returncode != 0:
            raise RuntimeError(f"volume: {err.decode(errors='replace').strip()[:100]}")
        self._vol_applied = True

    async def update_settings(self, changes: dict) -> None:
        if "volume" in changes:
            self.settings["volume"] = max(0, min(100, int(changes["volume"])))
        if changes.get("voice") in VOICES:
            self.settings["voice"] = changes["voice"]
        if "speed" in changes:
            self.settings["speed"] = round(max(0.7, min(1.4, float(changes["speed"]))), 2)
        for key in ("drive", "gestures", "wake_listen", "wake_verify"):
            if key in changes:
                self.settings[key] = bool(changes[key])
        if changes.get("llm") in LLM_MODES:
            self.settings["llm"] = changes["llm"]
        if str(changes.get("llm_server", "")) in {s["id"] for s in network_llms(load_conf())}:
            self.settings["llm_server"] = str(changes["llm_server"])
            self._lab_down_until = 0.0      # a new choice is tried first again
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.settings, f)
        if "volume" in changes:
            await self.apply_volume()
        self.on_change()

    async def test_voice(self) -> None:
        """say a fixed phrase with the current settings (to hear the level); not while answering"""
        if self._reply_task and not self._reply_task.done():
            return
        await self.announce(TEST_PHRASE, log=False)

    async def announce(self, text: str, log: bool = True) -> None:
        """say `text` outside a conversation (the agent's say tool, the voice test); mic muted meanwhile"""
        conf = load_conf()
        if log:
            self._say("robot", text)
        if not self._vol_applied:
            await self.apply_volume()
        own = self._session is None
        session = self._session or aiohttp.ClientSession(headers={"Authorization": f"Bearer {conf.get('VOICE_API_KEY', '')}"})
        prev = self._session
        self._session = session
        try:
            await self._mute(True)
            self._talking(True)
            for sentence in [x for x in SENTENCE_END.split(text) if speakable(x)] or [text]:
                wav = await self._tts(conf, speakable(sentence) or sentence)
                await self._play(wav, conf)
        except Exception as e:  # noqa: BLE001 - shown in the conversation
            self._say("note", f"speaking failed: {str(e)[:120] or type(e).__name__}")
        finally:
            self._talking(False)
            await self._mute(False)
            if own:
                await session.close()
                self._session = prev

    # ---- state for the page ---------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "message": self.message,
                "transcript": self.transcript[-14:], "stats": self.stats, "settings": self.settings, "voices": VOICES,
                "llms": self.llm_choices()}

    # ---- language models for the page: the robot's own and the network ones (no keys), reachability ---------
    def llm_choices(self) -> dict:
        conf = load_conf()
        net = [{"id": s["id"], "name": s["name"], "host": urlparse(s["url"]).hostname or "", "model": s["model"],
                "health": self.llm_health.get(s["id"])} for s in network_llms(conf)]
        return {"robot": {"name": robot_llm_name(), "health": self.llm_health.get("robot")}, "network": net}

    async def check_llms(self) -> None:
        """GET <base>/v1/models on each model server (2.5 s), for the page's status dots"""
        if time.time() - self._llm_checked < LLM_CHECK_MIN_S:
            return
        self._llm_checked = time.time()
        conf = load_conf()
        targets = [("robot", conf.get("VOICE_LOCAL_LLM_URL", LOCAL_LLM), "")] + \
                  [(s["id"], s["url"], s["key"]) for s in network_llms(conf)]

        async def probe(session: aiohttp.ClientSession, key: str, url: str, token: str) -> None:
            base = url.split("/v1/")[0].rstrip("/")
            t = time.time()
            try:
                async with session.get(base + "/v1/models", headers={"Authorization": f"Bearer {token}"} if token else None,
                                       timeout=aiohttp.ClientTimeout(total=2.5)) as r:
                    ok = r.status == 200
                    self.llm_health[key] = {"ok": ok, "ms": round((time.time() - t) * 1000)} if ok else \
                        {"ok": False, "error": f"HTTP {r.status}"}
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
                self.llm_health[key] = {"ok": False, "error": "no answer" if isinstance(e, asyncio.TimeoutError)
                                        else "not running" if key == "robot" else "unreachable"}

        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(probe(session, k, u, tok) for k, u, tok in targets))
        self.on_change()

    def _set(self, state: str, message: str = "") -> None:
        if (state, message) != (self.state, self.message):
            self.state, self.message = state, message
            self.on_change()

    def _say(self, role: str, text: str) -> None:
        self.transcript = (self.transcript + [{"role": role, "text": text, "t": time.strftime("%H:%M")}])[-40:]
        self.on_change()

    # ---- control ---------------------------------------------------------------------------
    async def enable(self, on: bool, wake_turn: bool = False) -> None:
        if on and self.enabled and not wake_turn:
            self.wake_turn = False          # the Listen switch turns a wake-word question into a conversation
        if on == self.enabled:
            return
        self.enabled = on
        self.wake_turn = on and wake_turn
        if on:
            self._tasks = [asyncio.create_task(self._run())]
        else:
            await self.stop_speaking()
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []
            self._set("off")

    async def wake(self) -> None:
        """the wake word was heard: chime and take one question, then wait for the wake word again.
        Ignored while the Listen switch keeps a conversation open (every phrase is answered then)."""
        if self.enabled:
            if self.wake_turn:
                self._wake_t = time.time()  # said again while waiting for the question: wait a bit longer
            return
        self._wake_t = time.time()
        await self.enable(True, wake_turn=True)
        self._say("note", "heard 'hey mercury' - listening")
        asyncio.create_task(self._chime())
        self._tasks.append(asyncio.create_task(self._wake_watch()))

    async def _chime(self) -> None:
        """short two-tone cue on the speaker (the mic is muted meanwhile)"""
        import math
        import struct
        rate, frames = 24000, []
        for f, dur in ((880, 0.09), (1320, 0.12)):
            n = int(rate * dur)
            frames += [int(9000 * math.sin(2 * math.pi * f * i / rate) * min(1, i / 240, (n - i) / 240)) for i in range(n)]
        data = struct.pack(f"<{len(frames)}h", *frames)
        wav = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) \
            + b"data" + struct.pack("<I", len(data)) + data
        try:
            await self._mute(True)
            await self._play(wav, load_conf())
        finally:
            await asyncio.sleep(0.15)
            await self._mute(False)

    async def _wake_watch(self) -> None:
        """no question after the wake word: stop listening"""
        while self.enabled and self.wake_turn:
            await asyncio.sleep(0.5)
            busy = self.state in ("thinking", "speaking", "hearing") or (self._reply_task and not self._reply_task.done())
            if not busy and time.time() - self._wake_t > WAKE_ASK_S:
                self._say("note", "no question heard - waiting for 'hey mercury'")
                asyncio.create_task(self.enable(False))
                return

    async def _end_wake_turn(self) -> None:
        if self.enabled and self.wake_turn:
            self._say("note", "waiting for 'hey mercury'")
            await self.enable(False)

    def clear(self) -> None:
        self.transcript, self.history = [], []
        self.on_change()

    async def stop_speaking(self) -> None:
        """cancel the current answer (model stream, speech, playback) and listen again"""
        if self._reply_task and not self._reply_task.done():
            self._reply_task.cancel()
            await asyncio.gather(self._reply_task, return_exceptions=True)
        await self._kill(self._aplay)
        await self._mute(False)

    # ---- main loop: (re)connect mic + speech service ----------------------------------------
    async def _run(self) -> None:
        while self.enabled:
            conf = load_conf()
            try:
                self._set("connecting")
                await self._session_run(conf)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - shown on the page, retried
                self._set("error", str(e)[:160] or type(e).__name__)
            finally:
                await self._kill(self._arecord)
                if self._ws is not None:
                    await self._ws.close()
                if self._session is not None:
                    await self._session.close()
                self._ws = self._session = self._arecord = None
            if self.enabled:
                await asyncio.sleep(3.0)

    async def _session_run(self, conf: dict) -> None:
        if not network_llms(conf) and self.settings.get("llm") != "robot":
            raise RuntimeError(f"no network model in {CONF} (or choose: on robot)")
        if not self._vol_applied:
            await self.apply_volume()
        self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {conf.get('VOICE_API_KEY', '')}"})
        url = conf.get("VOICE_SPEECH_URL", LOCAL_SPEECH).rstrip("/")
        try:
            self._ws = await self._session.ws_connect(url.replace("http", "ws", 1) + "/listen", heartbeat=15,
                                                      timeout=aiohttp.ClientWSTimeout(ws_close=5))
        except aiohttp.ClientConnectionError as e:
            raise RuntimeError("speech service is not running (sudo systemctl start mercury-speech)") from e
        self._arecord = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", conf.get("VOICE_MIC", "default"), "-f", "S16_LE", "-r", "16000", "-c", "1",
            "-t", "raw", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self._muted = False
        self._set("listening")
        pump = asyncio.create_task(self._pump_mic())
        # a dead microphone ends the session now, with its error (before, it sat "listening" and deaf until the
        # speech service closed the socket)
        pump.add_done_callback(lambda t: self._ws is not None and not self._ws.closed and asyncio.ensure_future(self._ws.close()))
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                ev = json.loads(msg.data)
                kind = ev.get("type")
                if kind in ("speech_start", "speech_end"):
                    self._wake_t = time.time()      # a question is under way: the wake turn waits for its text
                if kind == "speech_start" and self.state == "listening":
                    self._set("hearing")
                elif kind == "speech_end" and self.state == "hearing":
                    self._set("listening")
                elif kind == "final":
                    self.stats["stt_ms"] = ev.get("stt_ms")
                    await self._on_phrase(ev["text"], conf)
            if pump.done() and pump.exception():
                raise pump.exception()
            raise RuntimeError("speech service closed the connection")
        finally:
            pump.cancel()

    async def _pump_mic(self) -> None:
        assert self._arecord and self._arecord.stdout
        while True:
            chunk = await self._arecord.stdout.read(3200)    # 100 ms
            if not chunk:
                err = (await self._arecord.stderr.read()).decode(errors="replace").strip()
                raise RuntimeError(f"microphone stopped: {err[:120] or 'arecord exited'}")
            if not self._muted and self._ws is not None and not self._ws.closed:
                await self._ws.send_bytes(chunk)

    async def _mute(self, on: bool) -> None:
        self._muted = on
        if self._ws is not None and not self._ws.closed:
            await self._ws.send_json({"type": "mute", "on": on})

    # ---- one exchange ------------------------------------------------------------------------
    async def _on_phrase(self, text: str, conf: dict) -> None:
        if self._reply_task and not self._reply_task.done():
            return                          # still answering (mic is muted; should not happen)
        question = WAKE_ECHO.sub("", text).strip()
        if not re.search(r"\w", question):  # only the wake word (said again, or its tail): keep listening
            self._wake_t = time.time()
            return
        text = question
        now = time.time()
        if now - self.last_speech > IDLE_RESET_S:
            self.history = []
        self.last_speech = now
        self._say("user", text)
        if self.actions is not None and len(text.split()) <= 6 and STOP_PHRASE.search(text):
            self._say("note", self.actions.voice_stop())
            text += "  [the robot has already stopped all motion]"
        await self._mute(True)
        self._reply_task = asyncio.create_task(self._reply(text, conf))

    async def _reply(self, text: str, conf: dict) -> None:
        self._set("thinking")
        t0 = time.time()
        answer, queue = "", asyncio.Queue()
        speaker = asyncio.create_task(self._speak_queue(queue, conf))
        try:
            msgs = self._messages(text)
            tools = self._tools()
            first, spoken, n_sent, capped = None, False, 0, False
            for round_ in range(MAX_TOOL_ROUNDS + 1):
                body = {"messages": msgs, "stream": True, "max_tokens": 300, "temperature": 0.3,
                        "chat_template_kwargs": {"enable_thinking": False}}
                if tools and round_ < MAX_TOOL_ROUNDS:
                    body["tools"] = tools
                pending, calls, part = "", {}, ""
                async with await self._llm_open(body, conf) as r:
                    async for raw in r.content:
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data:") or line.endswith("[DONE]"):
                            continue
                        delta = json.loads(line[5:])["choices"][0].get("delta", {})
                        for tc in delta.get("tool_calls") or []:      # streamed in pieces, keyed by index
                            c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                            c["id"] = tc.get("id") or c["id"]
                            fn = tc.get("function") or {}
                            c["name"] += fn.get("name") or ""
                            c["arguments"] += fn.get("arguments") or ""
                        piece = delta.get("content") or ""
                        if not piece:
                            continue
                        first = first or time.time() - t0
                        part += piece
                        pending += piece
                        parts = SENTENCE_END.split(pending)
                        clause = FIRST_CLAUSE.match(parts[0]) if not spoken and len(parts) == 1 else None
                        if clause and speakable(clause.group(1)):    # speak the opening clause now: earlier first word
                            await queue.put(speakable(clause.group(1)))
                            spoken, pending = True, pending[clause.end():]
                            continue
                        for sentence in parts[:-1]:      # complete sentences go to speech right away
                            if speakable(sentence) and not capped:
                                await queue.put(speakable(sentence))
                                spoken, n_sent = True, n_sent + 1
                                capped = n_sent >= MAX_SENTENCES
                        pending = parts[-1]
                        if capped:                       # long answer: stop reading, say no more
                            part = part[:len(part) - len(pending)]
                            pending = ""
                            break
                if speakable(pending) and not capped:
                    await queue.put(speakable(pending))
                answer += part
                if not calls or capped:
                    break
                ordered = [calls[i] for i in sorted(calls)]
                msgs.append({"role": "assistant", "content": part or None, "tool_calls": [
                    {"id": c["id"] or f"call_{round_}_{i}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}} for i, c in enumerate(ordered)]})
                for i, c in enumerate(ordered):
                    result = await self._run_tool(c["name"], c["arguments"])
                    self.stats["last_tool"] = f"{c['name']} {c['arguments'][:60]}"
                    msgs.append({"role": "tool", "tool_call_id": c["id"] or f"call_{round_}_{i}",
                                 "content": json.dumps(result, ensure_ascii=False)})
                if answer.strip():   # already said something this turn: continue, do not repeat
                    msgs.append({"role": "system", "content": f"You already said aloud: \"{answer.strip()}\". "
                                 "Reply with only what the tool result adds, in one short sentence, or nothing."})

            self.stats["llm_first_s"] = round(first or 0, 2)
            self.stats["facts"] = self._facts_used
            answer = answer.strip()
            self._say("robot", answer)
            self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": answer}]
            await queue.put(None)
            await speaker
        except asyncio.CancelledError:
            speaker.cancel()
            if answer.strip():
                self._say("robot", answer.strip() + " …")
            raise
        except Exception as e:  # noqa: BLE001
            speaker.cancel()
            self._say("note", f"error: {str(e)[:140]}")
        finally:
            self._talking(False)
            await asyncio.sleep(UNMUTE_TAIL_S)
            await self._mute(False)
            self.last_speech = time.time()
            if self.enabled and self.state in ("thinking", "speaking"):
                self._set("listening")
            if self.wake_turn:      # one question per wake word (last statement: this task is done before it runs)
                asyncio.create_task(self._end_wake_turn())

    # ---- language model: lab server or the robot's own -------------------------------------------
    _lab_down_until = 0.0

    def _llm_targets(self, conf: dict) -> list[tuple[str, str, str, str, str]]:
        """(where, url, model, key, name) in the order to try: where = lab (a network model) | robot"""
        servers = network_llms(conf)
        pick = next((s for s in servers if s["id"] == self.settings.get("llm_server")), servers[0] if servers else None)
        robot = ("robot", conf.get("VOICE_LOCAL_LLM_URL", LOCAL_LLM), conf.get("VOICE_LOCAL_LLM_MODEL", "local"), "", "robot")
        mode = self.settings.get("llm", "auto")
        if mode == "robot" or pick is None:
            return [robot]
        lab = ("lab", pick["url"], pick["model"], pick["key"], pick["name"])
        if mode == "lab":
            return [lab]
        return [robot, lab] if time.time() < self._lab_down_until else [lab, robot]

    @staticmethod
    def _leading_system_only(msgs: list[dict]) -> list[dict]:
        """Qwen / most chat templates in llama.cpp reject a system message after the start: fold the facts (and the
        after-tool note) into the next user turn instead, or send a trailing note as a user turn"""
        out: list[dict] = []
        pending: list[str] = []
        msgs = [{**msgs[0], "content": msgs[0]["content"] + LOCAL_RULES}, *msgs[1:]] if msgs and msgs[0]["role"] == "system" else msgs
        for i, m in enumerate(msgs):
            if m["role"] == "system" and i > 0:
                pending.append(m["content"])
                continue
            if pending and m["role"] == "user":
                m = {**m, "content": "\n\n".join(pending) + "\n\nThe person said: " + m["content"]}
                pending = []
            out.append(m)
        if pending:
            out.append({"role": "user", "content": "(note) " + "\n\n".join(pending)})
        return out

    async def _llm_open(self, body: dict, conf: dict) -> aiohttp.ClientResponse:
        """streaming chat request to the first model that starts answering (failover only before the reply starts)"""
        errors = []
        for where, url, model, key, name in self._llm_targets(conf):
            try:
                timeout = aiohttp.ClientTimeout(total=60, sock_connect=1.5 if where == "lab" else 5)
                req = {**body, "model": model}
                hdr = {"Authorization": f"Bearer {key}"} if key else None
                if where == "robot":   # llama-server: strict templates, and it drops kept-alive connections
                    req["messages"] = self._leading_system_only(body["messages"])
                    hdr = {"Connection": "close"}
                for attempt in range(2):
                    try:
                        r = await asyncio.wait_for(self._session.post(url, json=req, timeout=timeout, headers=hdr),
                                                   LAB_HEADERS_S if where == "lab" else 30)
                        break
                    except aiohttp.ServerDisconnectedError:
                        if attempt:
                            raise
                if r.status != 200:
                    text = (await r.text())[:100]
                    r.release()
                    raise RuntimeError(f"HTTP {r.status} {text}")
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, OSError) as e:
                errors.append(f"{name}: {str(e)[:80] or type(e).__name__}")
                if where == "lab":
                    self._lab_down_until = time.time() + LAB_RETRY_S
                continue
            if where == "lab":
                self._lab_down_until = 0.0
            self.stats["llm_on"] = where
            self.stats["llm_name"] = name if where == "lab" else robot_llm_name()
            return r
        raise RuntimeError("model: " + "; ".join(errors))

    # ---- knowledge + tools ---------------------------------------------------------------------
    _facts_used = 0

    def _messages(self, text: str) -> list[dict]:
        """system prompt + history + retrieved facts + the new phrase"""
        query = text
        if len(text.split()) < 6:   # a short follow-up ("and the margin?"): the previous question gives context
            prev = [m["content"] for m in self.history[-4:] if m["role"] == "user"]
            query = " ".join(prev[-1:] + [text])
        hits = self.kb.search(query)
        self._facts_used = len(hits)
        system = SYSTEM_PROMPT
        if self.actions is not None:
            places = self.actions.voice_places()
            system += (f" Named places on the current map: {', '.join(places)}." if places else
                       " No named places are saved on the current map yet.")
            system += (" Driving by voice is switched ON." if self.settings.get("drive") else
                       " Driving by voice is switched OFF (it can be switched on in the kiosk Voice tab).")
        msgs = [{"role": "system", "content": system}] + self.history[-2 * HISTORY_TURNS:]
        if hits:   # right before the question: the model weighs it most
            msgs.append({"role": "system", "content": as_prompt(hits)})
        return msgs + [{"role": "user", "content": text}]

    def _talking(self, on: bool) -> None:
        """body language while speaking (the App only moves when the gestures switch is on)"""
        fn = getattr(self.actions, "voice_talking", None)
        if fn is not None:
            try:
                fn(on)
            except Exception:  # noqa: BLE001 - never let gestures break the conversation
                pass

    def _tools(self) -> list[dict]:
        if self.actions is None:
            return []

        def fn(name: str, desc: str, props: dict | None = None, req: list | None = None) -> dict:
            return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
                "type": "object", "properties": props or {}, "required": req or []}}}
        tools = [
            fn("get_robot_status", "Only when the person asks about your battery, the time or date, or where you "
               "are: battery level, navigation state, map, nearest named place, time, date."),
            fn("list_places", "Names of the places the robot knows on the current map."),
            fn("go_to_place", "Drive to a named place on the map, e.g. when asked to go to or show the way to the "
               "kitchen. Only for places in the list.", {"place": {"type": "string", "description": "place name"}},
               ["place"]),
            fn("stop_moving", "Stop driving immediately (cancel navigation). Use it when someone says stop or wait."),
        ]
        if hasattr(self.actions, "voice_task"):
            tools.append(fn("do_task", "Hand a task that needs searching or several steps to your planning brain, which "
                            "drives and looks around on its own: find something or someone (find the sink, find a "
                            "person and tell them something), go somewhere that is not a named place, check on "
                            "something. Say in one short sentence that you are on it; the result is said when done.",
                            {"task": {"type": "string", "description": "the task in the person's words"}}, ["task"]))
        gestures = getattr(self.actions, "voice_gesture_names", lambda: [])()
        if gestures:
            from .gestures import GESTURES
            tools.append(fn("gesture", "Do a body gesture with the waist, head and arms. " +
                            "; ".join(f"{k} = {v}" for k, v in GESTURES.items() if k in gestures) +
                            ". Use thank_you when someone thanks you or you thank them, wave for hello and goodbye.",
                            {"name": {"type": "string", "enum": gestures}}, ["name"]))
        return tools

    async def _run_tool(self, name: str, arguments: str) -> dict:
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            args = {}
        a = self.actions
        try:
            if name == "get_robot_status":
                return a.voice_status()
            if name == "list_places":
                return {"places": a.voice_places()}
            if name == "go_to_place":
                if not self.settings.get("drive"):
                    return {"ok": False, "error": "driving by voice is switched off (kiosk Voice tab)"}
                ok, msg = a.voice_go(str(args.get("place", "")))
                self._say("note", msg)
                return {"ok": ok, "result" if ok else "error": msg}
            if name == "stop_moving":
                return {"ok": True, "result": a.voice_stop()}
            if name == "do_task":
                if not self.settings.get("drive"):
                    return {"ok": False, "error": "driving by voice is switched off (kiosk Voice tab)"}
                ok, msg = a.voice_task(str(args.get("task", "")))
                self._say("note", f"agent: {msg}")
                return {"ok": ok, "result" if ok else "error": msg}
            if name == "gesture":
                if not self.settings.get("gestures"):
                    return {"ok": False, "error": "gestures are switched off (kiosk Voice tab)"}
                ok, msg = await a.voice_gesture(str(args.get("name", "")))
                return {"ok": ok, "result" if ok else "error": msg}
        except Exception as e:  # noqa: BLE001 - the model gets the error and can say so
            return {"ok": False, "error": str(e)[:120]}
        return {"ok": False, "error": f"unknown tool {name}"}

    async def _speak_queue(self, queue: asyncio.Queue, conf: dict) -> None:
        """synthesise sentence n+1 while sentence n plays"""
        nxt: asyncio.Task | None = None
        sentence = await queue.get()
        while sentence is not None:
            wav_task = nxt or asyncio.create_task(self._tts(conf, sentence))
            following = await queue.get()
            nxt = asyncio.create_task(self._tts(conf, following)) if following is not None else None
            wav = await wav_task
            self.stats["tts_ms"], self.stats["tts_on"] = self._last_tts_ms, self._last_tts_on
            if self.state != "speaking":
                self._talking(True)
            self._set("speaking")
            await self._play(wav, conf)
            sentence = following

    async def _tts(self, conf: dict, text: str) -> bytes:
        """wav from the robot's own Kokoro (Jetson GPU, mercury-speech)"""
        body = {"text": text, "voice": self.settings["voice"], "speed": self.settings["speed"]}
        base = conf.get("VOICE_SPEECH_URL", LOCAL_SPEECH).rstrip("/")
        t = time.time()
        try:
            async with self._session.post(base + "/tts", json=body, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    raise RuntimeError(f"speech: HTTP {r.status}")
                data = await r.read()
        except aiohttp.ClientConnectionError as e:
            raise RuntimeError("speech service is not running (sudo systemctl start mercury-speech)") from e
        self._last_tts_ms = round((time.time() - t) * 1000)
        self._last_tts_on = "Jetson" if urlparse(base).hostname in ("127.0.0.1", "localhost") else urlparse(base).hostname
        return data

    _last_tts_ms, _last_tts_on = 0, ""

    async def _play(self, wav: bytes, conf: dict) -> None:
        self._aplay = await asyncio.create_subprocess_exec(
            "aplay", "-q", "-D", conf.get("VOICE_SPEAKER", "default"), "-",
            stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            self._aplay.stdin.write(wav)
            await self._aplay.stdin.drain()
            self._aplay.stdin.close()
            await self._aplay.wait()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self._aplay = None

    @staticmethod
    async def _kill(proc) -> None:
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), 2.0)
            except asyncio.TimeoutError:
                pass
