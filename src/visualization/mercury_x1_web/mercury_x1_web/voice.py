"""Hands-free voice chat for the head display: mic -> speech service (11.51) -> Qwen -> speech -> speaker.

Runs on web_server's asyncio loop (all I/O, no threads):
  * `arecord` streams the XFM-DP mic array (16 kHz mono, beamformed on the board) to the speech service's
    /listen WebSocket, which finds the phrases and transcribes them with Whisper.
  * Each phrase goes to Qwen (the pennyroyal proxy, streamed, thinking off); the reply is cut into
    sentences, each one is synthesised by Kokoro (/tts) and played with `aplay` on the C-Media speaker
    while the next is being made.
  * The mic is muted while the robot talks (no echo loop) and for a short tail after it.
Config: ~/.config/mercury_x1/voice.env (URLs, bearer token, ALSA devices; chmod 600).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import aiohttp

CONF = os.path.expanduser("~/.config/mercury_x1/voice.env")
SYSTEM_PROMPT = (
    "You are Mercury, a friendly wheeled humanoid robot with two arms, made by Elephant Robotics and working in the "
    "HCLTech robotics lab. You are talking out loud with people standing in front of you, so answer in one to three "
    "short spoken sentences, in plain words: no lists, no markdown, no emojis, no code. If you did not understand, "
    "ask briefly. You cannot move or act on requests yet; you can only talk.")
HISTORY_TURNS = 8          # user + assistant pairs kept as context
IDLE_RESET_S = 180         # forget the conversation after this long without speech
UNMUTE_TAIL_S = 0.35       # keep the mic muted this long after the speaker stops
SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")


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

    # ---- state for the page ---------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"enabled": self.enabled, "state": self.state, "message": self.message,
                "transcript": self.transcript[-14:], "stats": self.stats}

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
            pending, first = "", None
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
                    for sentence in parts[:-1]:      # complete sentences go to speech right away
                        if speakable(sentence):
                            await queue.put(speakable(sentence))
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
        url = conf["VOICE_STT_TTS_URL"].rstrip("/") + "/tts"
        nxt: asyncio.Task | None = None
        sentence = await queue.get()
        while sentence is not None:
            wav_task = nxt or asyncio.create_task(self._tts(url, sentence))
            following = await queue.get()
            nxt = asyncio.create_task(self._tts(url, following)) if following is not None else None
            wav = await wav_task
            self.stats["tts_ms"] = self._last_tts_ms
            self._set("speaking")
            await self._play(wav, conf)
            sentence = following

    async def _tts(self, url: str, text: str) -> bytes:
        t = time.time()
        async with self._session.post(url, json={"text": text, "voice": "af_heart"},
                                      timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status != 200:
                raise RuntimeError(f"speech: HTTP {r.status}")
            data = await r.read()
        self._last_tts_ms = round((time.time() - t) * 1000)
        return data

    _last_tts_ms = 0

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
