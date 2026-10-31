#!/usr/bin/env python3
"""Speech service for the Mercury X1 voice chat, next to the Qwen model on 192.168.11.51.

  WS   /listen    robot streams its mic (PCM16 LE, 16 kHz, mono, binary frames); the service finds the
                  phrases (Silero VAD, streaming), transcribes each with Whisper on the GPU and sends JSON events:
                  {"type": "speech_start"} / {"type": "speech_end"} / {"type": "final", "text", ...} /
                  {"type": "dropped", "text", "why"}. Client messages: {"type": "mute", "on": bool} (stop
                  segmenting, e.g. while the robot talks), {"type": "config", "language": "en"}.
  POST /tts       {"text", "voice"?, "speed"?} -> audio/wav (Kokoro, CPU, 24 kHz mono)
  GET  /health    no auth

Bearer token on /listen and /tts: the model's PENNY_API_KEY (systemd EnvironmentFile).
Whisper large-v3-turbo fp16 holds ~2.5 GB of GPU memory and runs ~0.1 s per phrase; Kokoro runs on
the CPU (~0.2 s per sentence) so it never competes with the model for the GPU.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
import wave

import numpy as np
import onnxruntime as ort
import faster_whisper
from aiohttp import web
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("VOICE_PORT", "8096"))
API_KEY = os.environ.get("PENNY_API_KEY", "")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
TTS_THREADS = int(os.environ.get("TTS_THREADS", "8"))
RATE = 16000
# Voice activity: Silero (neural). webrtcvad called 95 % of the robot's room "speech": its noise is a
# 150-300 Hz hum (fans / servos), right in the voice pitch range; Silero scores it <= 0.3.
FRAME = 512                          # 32 ms Silero chunk
FRAME_BYTES = FRAME * 2
P_START, P_KEEP = 0.5, 0.35          # speech probability to start / to stay in a phrase
START_FRAMES, START_NEED = 6, 3      # start: 3 of the last 6 chunks (~200 ms) above P_START
END_SILENCE = 25                     # end: 25 chunks (0.8 s) below P_KEEP
PREROLL = 12                         # chunks kept before the start (~380 ms)
MAX_FRAMES = int(15 / 0.032)         # cut a phrase at 15 s
MIN_FRAMES = int(0.4 / 0.032)        # ignore blips shorter than 0.4 s
# what Whisper "hears" in noise or silence
HALLUCINATIONS = {"you", "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "bye.", ".", "uh",
                  "thank you for watching.", "so", "okay.", "you."}

log = logging.getLogger("voice")


def authorized(req: web.Request) -> bool:
    return not API_KEY or req.headers.get("Authorization", "") == f"Bearer {API_KEY}"


class Engines:
    def __init__(self) -> None:
        t = time.time()
        self.whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        # warm-up: the first transcription on a fresh process takes ~5 s
        list(self.whisper.transcribe(np.zeros(RATE, np.float32), language="en", beam_size=1)[0])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = TTS_THREADS
        sess = ort.InferenceSession(os.path.join(HERE, "models", "kokoro-v1.0.onnx"), opts,
                                    providers=["CPUExecutionProvider"])
        self.kokoro = Kokoro.from_session(sess, os.path.join(HERE, "models", "voices-v1.0.bin"))
        self.kokoro.create("Ready.", voice="af_heart", lang="en-us")
        self.stt_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        log.info("engines ready in %.1f s (whisper %s, kokoro %d threads)", time.time() - t, WHISPER_MODEL, TTS_THREADS)

    def transcribe(self, pcm: bytes, language: str) -> dict:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t = time.time()
        with self.stt_lock:
            segs, _ = self.whisper.transcribe(audio, language=language or None, beam_size=1,
                                              condition_on_previous_text=False)
            segs = list(segs)
        text = " ".join(s.text.strip() for s in segs).strip()
        no_speech = max((s.no_speech_prob for s in segs), default=1.0)
        logprob = min((s.avg_logprob for s in segs), default=-9.0)
        return {"text": text, "stt_ms": round((time.time() - t) * 1000), "duration": round(len(audio) / RATE, 2),
                "no_speech_prob": round(no_speech, 2), "avg_logprob": round(logprob, 2)}

    def speak(self, text: str, voice: str, speed: float) -> bytes:
        with self.tts_lock:
            # voice prefix a* = American, b* = British English (Kokoro voice naming)
            audio, sr = self.kokoro.create(text, voice=voice, speed=speed, lang="en-gb" if voice[:1] == "b" else "en-us")
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        return buf.getvalue()


class Silero:
    """streaming Silero VAD v6 (the copy bundled with faster-whisper): 512-sample chunks, state kept"""

    _session = None

    def __init__(self) -> None:
        if Silero._session is None:
            o = ort.SessionOptions()
            o.intra_op_num_threads = o.inter_op_num_threads = 1
            o.log_severity_level = 4
            path = os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")
            Silero._session = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)

    def __call__(self, frame: bytes) -> float:
        x = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
        inp = np.concatenate([self.ctx, x])[None]
        self.ctx = x[-64:]
        p, self.h, self.c = self._session.run(None, {"input": inp, "h": self.h, "c": self.c})
        return float(np.ravel(p)[0])


def rejected(r: dict) -> str | None:
    """why a transcription is not a real phrase (or None)"""
    t = r["text"].strip().lower()
    if not t:
        return "empty"
    words = [w.strip(".,!?") for w in t.split()]
    if len(words) >= 5 and max(words.count(w) for w in set(words)) > 0.6 * len(words):
        return "repeated word (noise)"
    if t in HALLUCINATIONS and r["duration"] < 2.5:
        return "typical noise transcription"
    if r["no_speech_prob"] > 0.6 and r["avg_logprob"] < -0.7:
        return "no speech"
    return None


class Listener:
    """voice activity segmentation for one /listen connection"""

    def __init__(self) -> None:
        self.vad = Silero()
        self.buf = b""
        self.recent: list[bool] = []
        self.pre: list[bytes] = []
        self.phrase: list[bytes] | None = None
        self.silence = 0
        self.muted = False

    def feed(self, data: bytes):
        """-> list of events: ("start",) / ("end", pcm)"""
        out = []
        self.buf += data
        while len(self.buf) >= FRAME_BYTES:
            f, self.buf = self.buf[:FRAME_BYTES], self.buf[FRAME_BYTES:]
            if self.muted:
                continue
            p = self.vad(f)
            if self.phrase is None:
                self.pre = (self.pre + [f])[-PREROLL:]
                self.recent = (self.recent + [p >= P_START])[-START_FRAMES:]
                if sum(self.recent) >= START_NEED:
                    self.phrase, self.silence = list(self.pre), 0
                    out.append(("start",))
            else:
                self.phrase.append(f)
                self.silence = 0 if p >= P_KEEP else self.silence + 1
                if self.silence >= END_SILENCE or len(self.phrase) >= MAX_FRAMES:
                    pcm = b"".join(self.phrase[:len(self.phrase) - self.silence + 5])
                    long_enough = len(self.phrase) - self.silence >= MIN_FRAMES
                    self.phrase, self.recent, self.pre = None, [], []
                    out.append(("end", pcm if long_enough else None))
        return out

    def mute(self, on: bool) -> None:
        self.muted = on
        self.phrase, self.recent, self.pre, self.buf = None, [], [], b""
        self.vad.reset()


async def listen(req: web.Request) -> web.StreamResponse:
    if not authorized(req):
        raise web.HTTPUnauthorized()
    eng: Engines = req.app["engines"]
    ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
    await ws.prepare(req)
    lst, language = Listener(), "en"
    loop = asyncio.get_running_loop()
    log.info("listener connected from %s", req.remote)
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            for ev in lst.feed(msg.data):
                if ev[0] == "start":
                    await ws.send_json({"type": "speech_start"})
                    continue
                await ws.send_json({"type": "speech_end"})
                if ev[1] is None:
                    continue
                r = await loop.run_in_executor(None, eng.transcribe, ev[1], language)
                why = rejected(r)
                log.info("%s %r (%.1f s audio, %d ms)", "dropped: " + why if why else "heard", r["text"], r["duration"], r["stt_ms"])
                await ws.send_json({**r, "type": "dropped" if why else "final", **({"why": why} if why else {})})
        elif msg.type == web.WSMsgType.TEXT:
            try:
                m = json.loads(msg.data)
            except ValueError:
                continue
            if m.get("type") == "mute":
                lst.mute(bool(m.get("on")))
            elif m.get("type") == "config":
                language = m.get("language", language)
    log.info("listener from %s closed", req.remote)
    return ws


async def tts(req: web.Request) -> web.Response:
    if not authorized(req):
        raise web.HTTPUnauthorized()
    body = await req.json()
    text = str(body.get("text", "")).strip()[:1000]
    if not text:
        raise web.HTTPBadRequest(text="no text")
    eng: Engines = req.app["engines"]
    t = time.time()
    wav = await asyncio.get_running_loop().run_in_executor(
        None, eng.speak, text, str(body.get("voice", "af_heart")), min(max(float(body.get("speed", 1.0)), 0.5), 2.0))
    return web.Response(body=wav, content_type="audio/wav", headers={"X-TTS-ms": str(round((time.time() - t) * 1000))})


async def health(req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "whisper": WHISPER_MODEL, "tts": "kokoro-v1.0"})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("faster_whisper", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    app = web.Application()
    app["engines"] = Engines()
    app.router.add_get("/health", health)
    app.router.add_get("/listen", listen)
    app.router.add_post("/tts", tts)
    web.run_app(app, port=PORT, access_log=None)


if __name__ == "__main__":
    main()
