"""Hands-free voice chat for the head display: mic -> speech service (11.51) -> Qwen -> speech -> speaker.

Runs on web_server's asyncio loop (all I/O, no threads):
  * `arecord` streams the XFM-DP mic array (16 kHz mono, beamformed on the board) to the speech service's
    /listen WebSocket, which finds the phrases and transcribes them with Whisper.
  * Each phrase goes to Qwen (the pennyroyal proxy, streamed, thinking off); the reply is cut into
    sentences, each one is synthesised by Kokoro (/tts) and played with `aplay` on the C-Media speaker
    while the next is being made. Kokoro runs on the robot's GPU (`mercury-tts`, 127.0.0.1:8097,
    VOICE_TTS_URL; 11.51 only does speech-to-text).
  * The mic is muted while the robot talks (no echo loop) and for a short tail after it.
Config: ~/.config/mercury_x1/voice.env (URLs, bearer token, ALSA devices; chmod 600).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from urllib.parse import urlparse

import aiohttp

CONF = os.path.expanduser("~/.config/mercury_x1/voice.env")
SETTINGS_FILE = os.path.expanduser("~/.config/mercury_x1/voice_settings.json")
VOICES = {"af_heart": "Heart (US)", "af_bella": "Bella (US)", "am_michael": "Michael (US)",
          "bf_emma": "Emma (UK)", "bm_george": "George (UK)", "am_puck": "Puck (US)"}
DEFAULT_SETTINGS = {"volume": 46, "voice": "af_heart", "speed": 1.0}
TEST_PHRASE = "Hello, I am Mercury. This is how loud I speak."
SYSTEM_PROMPT = (
    "You are Mercury, a friendly wheeled humanoid robot with two arms, made by Elephant Robotics and working in the "
    "HCLTech robotics lab. You are talking out loud with people standing in front of you, so answer in one to three "
    "short spoken sentences, in plain words: no lists, no markdown, no emojis, no code. If you did not understand, "
    "ask briefly. You cannot move or act on requests yet; you can only talk.")
HISTORY_TURNS = 8          # user + assistant pairs kept as context
IDLE_RESET_S = 180         # forget the conversation after this long without speech
UNMUTE_TAIL_S = 0.35       # keep the mic muted this long after the speaker stops
LOCAL_TTS = "http://127.0.0.1:8097"   # server/robot_tts on the Jetson GPU
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")
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


def speakable(text: str) -> str:
    """strip what TTS would read out literally (markdown, emoji, URLs)"""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_`#>|~\[\]]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class VoiceChat:
    def __init__(self, on_change=None) -> None:
        self.on_change = on_change or (lambda: None)
        self.enabled = False
        self.state = "off"         # off | connecting | listening | hearing | thinking | speaking | error
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
        conf = load_conf()
        if not self._vol_applied:
            await self.apply_volume()
        own = self._session is None
        session = self._session or aiohttp.ClientSession(headers={"Authorization": f"Bearer {conf.get('VOICE_API_KEY', '')}"})
        prev = self._session
        self._session = session
        try:
            await self._mute(True)
            wav = await self._tts(conf, TEST_PHRASE)
            await self._play(wav, conf)
        except Exception as e:  # noqa: BLE001 - shown in the conversation
            self._say("note", f"voice test failed: {str(e)[:120] or type(e).__name__}")
        finally:
            await self._mute(False)
            if own:
                await session.close()
                self._session = prev

    # ---- state for the page ---------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "message": self.message,
                "transcript": self.transcript[-14:], "stats": self.stats, "settings": self.settings, "voices": VOICES}

    def _set(self, state: str, message: str = "") -> None:
        if (state, message) != (self.state, self.message):
            self.state, self.message = state, message
            self.on_change()

    def _say(self, role: str, text: str) -> None:
        self.transcript = (self.transcript + [{"role": role, "text": text, "t": time.strftime("%H:%M")}])[-40:]
        self.on_change()

    # ---- control ---------------------------------------------------------------------------
    async def enable(self, on: bool) -> None:
        if on == self.enabled:
            return
        self.enabled = on
        if on:
            self._tasks = [asyncio.create_task(self._run())]
        else:
            await self.stop_speaking()
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks = []
            self._set("off")

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
        if not conf.get("VOICE_STT_TTS_URL"):
            raise RuntimeError(f"not configured ({CONF})")
        if not self._vol_applied:
            await self.apply_volume()
        self._session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {conf.get('VOICE_API_KEY', '')}"})
        url = conf["VOICE_STT_TTS_URL"].rstrip("/")
        self._ws = await self._session.ws_connect(url.replace("http", "ws", 1) + "/listen", heartbeat=15,
                                                  timeout=aiohttp.ClientWSTimeout(ws_close=5))
        self._arecord = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", conf.get("VOICE_MIC", "default"), "-f", "S16_LE", "-r", "16000", "-c", "1",
            "-t", "raw", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self._muted = False
        self._set("listening")
        pump = asyncio.create_task(self._pump_mic())
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                ev = json.loads(msg.data)
                kind = ev.get("type")
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
        now = time.time()
        if now - self.last_speech > IDLE_RESET_S:
            self.history = []
        self.last_speech = now
        self._say("user", text)
        await self._mute(True)
        self._reply_task = asyncio.create_task(self._reply(text, conf))

    async def _reply(self, text: str, conf: dict) -> None:
        self._set("thinking")
        t0 = time.time()
        answer, queue = "", asyncio.Queue()
        speaker = asyncio.create_task(self._speak_queue(queue, conf))
        try:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-2 * HISTORY_TURNS:] + \
                   [{"role": "user", "content": text}]
            body = {"model": conf.get("VOICE_LLM_MODEL", "pennyroyal"), "messages": msgs, "stream": True,
                    "max_tokens": 300, "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
            pending, first, spoken = "", None, False
            async with self._session.post(conf["VOICE_LLM_URL"], json=body,
                                          timeout=aiohttp.ClientTimeout(total=60)) as r:
                if r.status != 200:
                    raise RuntimeError(f"model: HTTP {r.status} {(await r.text())[:100]}")
                async for raw in r.content:
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    delta = json.loads(line[5:])["choices"][0].get("delta", {})
                    piece = delta.get("content") or ""
                    if not piece:
                        continue
                    first = first or time.time() - t0
                    answer += piece
                    pending += piece
                    parts = SENTENCE_END.split(pending)
                    clause = FIRST_CLAUSE.match(parts[0]) if not spoken and len(parts) == 1 else None
                    if clause and speakable(clause.group(1)):    # speak the opening clause now: earlier first word
                        await queue.put(speakable(clause.group(1)))
                        spoken, pending = True, pending[clause.end():]
                        continue
                    for sentence in parts[:-1]:      # complete sentences go to speech right away
                        if speakable(sentence):
                            await queue.put(speakable(sentence))
                            spoken = True
                    pending = parts[-1]
            if speakable(pending):
                await queue.put(speakable(pending))
            self.stats["llm_first_s"] = round(first or 0, 2)
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
            await asyncio.sleep(UNMUTE_TAIL_S)
            await self._mute(False)
            self.last_speech = time.time()
            if self.enabled and self.state in ("thinking", "speaking"):
                self._set("listening")

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
            self._set("speaking")
            await self._play(wav, conf)
            sentence = following

    async def _tts(self, conf: dict, text: str) -> bytes:
        """wav from the robot's own Kokoro (Jetson GPU, mercury-tts)"""
        body = {"text": text, "voice": self.settings["voice"], "speed": self.settings["speed"]}
        base = conf.get("VOICE_TTS_URL", LOCAL_TTS).rstrip("/")
        t = time.time()
        try:
            async with self._session.post(base + "/tts", json=body, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    raise RuntimeError(f"speech: HTTP {r.status}")
                data = await r.read()
        except aiohttp.ClientConnectionError as e:
            raise RuntimeError("speech: mercury-tts is not running (sudo systemctl start mercury-tts)") from e
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
