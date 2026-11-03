#!/usr/bin/env python3
"""Speech on the robot itself (Jetson Orin) for the head display's voice chat: hearing and speaking, no cloud.

  WS   /listen    voice.py streams the mic (PCM16 LE, 16 kHz, mono, binary frames); Silero VAD finds the
                  phrases, Parakeet transcribes each one. Events (JSON): {"type": "speech_start"} /
                  {"type": "speech_end"} / {"type": "final", "text", "stt_ms", "duration"} /
                  {"type": "dropped", "text", "why"}. Client messages: {"type": "mute", "on": bool}.
  POST /tts       {"text", "voice"?, "speed"?} -> audio/wav (24 kHz mono)
  GET  /health

Listens on 127.0.0.1 only (no token). Own venv (~/mercury-speech, `setup.sh`), not ROS' Python: the Jetson
onnxruntime-gpu wheel needs NumPy 1.x. Measured with the robot stack running:
  * STT: NVIDIA Parakeet TDT 0.6B v2, fp16 on the GPU (make_parakeet_fp16.py; LayerNorm stays fp32): ~0.4 s
    for a short phrase, ~0.5 s for 7.7 s of speech, ~1.5 GB, no CPU. Falls back to the published int8 model on
    the CPU (2 threads: ~0.38 / ~1.44 s, ~1 GB) when the fp16 files are missing or the GPU load fails; int8 itself
    does not fit on CUDA (ORT's int8 kernels are CPU kernels; on CUDA it ran out of memory).
  * TTS: Kokoro fp32 on CUDA (GPU arena capped): ~0.35 s for a short sentence, ~0.75 s for 5 s of speech,
    ~0.9 GB; GPU-bound (shared with nvblox). The fp16 model is slower here (casts).
  * VAD: Silero v6, streaming, on the CPU (~1 % of a core).
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
import onnx_asr
from aiohttp import web
from kokoro_onnx import Kokoro

HOME = os.path.expanduser("~/mercury-speech")
MODELS = os.environ.get("SPEECH_MODELS", os.path.join(HOME, "models"))
PORT = int(os.environ.get("SPEECH_PORT", "8097"))
TTS_MODEL = os.environ.get("TTS_MODEL", "kokoro-v1.0.onnx")
TTS_GPU_MEM_MB = int(os.environ.get("TTS_GPU_MEM_MB", "768"))
STT_MODEL = "nemo-parakeet-tdt-0.6b-v2"
STT_DIR = os.path.join(MODELS, "parakeet-tdt-0.6b-v2")
STT_THREADS = int(os.environ.get("STT_THREADS", "2"))     # 4 is no faster: the CPU is busy with ROS
STT_DEVICE = os.environ.get("STT_DEVICE", "cuda")          # cuda: fp16 model on the GPU; cpu: int8 model
STT_GPU_MEM_MB = int(os.environ.get("STT_GPU_MEM_MB", "2048"))
RATE = 16000
WARMUP = ["Ready.", "Hello, I am Mercury, the robot in the HCLTech robotics lab.",
          "I can talk with you, but I cannot move yet, so please ask me something about the lab or the robots here."]

# voice activity: Silero (neural). webrtcvad called 95 % of the robot's room "speech": its noise is a
# 150-300 Hz hum (fans / servos), right in the voice pitch range; Silero scores it <= 0.3.
FRAME = 512                          # 32 ms Silero chunk
FRAME_BYTES = FRAME * 2
P_START, P_KEEP = 0.5, 0.35          # speech probability to start / to stay in a phrase
START_FRAMES, START_NEED = 6, 3      # start: 3 of the last 6 chunks (~200 ms) above P_START
END_SILENCE = 25                     # end: 25 chunks (0.8 s) below P_KEEP
PREROLL = 12                         # chunks kept before the start (~380 ms)
MAX_FRAMES = int(15 / 0.032)         # cut a phrase at 15 s
MIN_FRAMES = int(0.4 / 0.032)        # ignore blips shorter than 0.4 s
NOISE_WORDS = {"", "mm", "hmm", "uh", "um", "ah", "oh"}   # what a noise burst can come out as

log = logging.getLogger("speech")


def _quiet_session_options(threads: int) -> ort.SessionOptions:
    o = ort.SessionOptions()
    o.log_severity_level = 3
    o.intra_op_num_threads = threads
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")   # no busy-waiting: ROS keeps its cores
    return o


class Engines:
    def __init__(self) -> None:
        t = time.time()
        # TTS: Kokoro on the GPU
        cuda = {"cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
                "arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(TTS_GPU_MEM_MB << 20)}
        sess = ort.InferenceSession(os.path.join(MODELS, TTS_MODEL), _quiet_session_options(1),
                                    providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])
        self.tts_provider = sess.get_providers()[0]
        self.kokoro = Kokoro.from_session(sess, os.path.join(MODELS, "voices-v1.0.bin"))
        # warm Kokoro up before Parakeet loads: the first runs per input length pick their kernels (~1 s each),
        # and its GPU arena is claimed right after the memory compaction (systemd ExecStartPre)
        for _ in range(2):
            for s in WARMUP:
                self.kokoro.create(s, voice="af_heart", lang="en-us")
        # STT: Parakeet fp16 on the GPU, else int8 on the CPU
        self.parakeet, self.stt_where = None, ""
        if STT_DEVICE == "cuda" and os.path.exists(os.path.join(STT_DIR, "encoder-model.fp16.onnx")):
            try:
                self.parakeet = self._load_stt(cuda)
                self.stt_where = "fp16, GPU"
            except Exception as e:  # noqa: BLE001 - e.g. nvmap out of memory: hear on the CPU instead
                log.warning("Parakeet on the GPU failed (%s); using the int8 model on the CPU", str(e)[-160:])
                self.parakeet = None
        if self.parakeet is None:
            self.parakeet = self._load_stt(None)
            self.stt_where = f"int8, CPU {STT_THREADS} threads"
        self.tts_lock = threading.Lock()
        self.stt_lock = threading.Lock()
        log.info("ready in %.1f s (TTS %s on %s, GPU arena <= %d MB; STT %s %s)", time.time() - t,
                 TTS_MODEL, self.tts_provider, TTS_GPU_MEM_MB, STT_MODEL, self.stt_where)

    @staticmethod
    def _load_stt(cuda: dict | None):
        so = _quiet_session_options(STT_THREADS)
        so.enable_cpu_mem_arena = False                       # the CPU arena kept ~150 MB more for nothing
        if cuda is not None:
            providers = [("CUDAExecutionProvider", {**cuda, "gpu_mem_limit": str(STT_GPU_MEM_MB << 20)}),
                         "CPUExecutionProvider"]
            m = onnx_asr.load_model(STT_MODEL, STT_DIR, quantization="fp16", sess_options=so, providers=providers)
        else:
            m = onnx_asr.load_model(STT_MODEL, STT_DIR, quantization="int8", sess_options=so,
                                    providers=["CPUExecutionProvider"])
        # first runs per length pick the GPU kernels; the longest phrase (MAX_FRAMES, 15 s) also grows the GPU
        # arena to its full size now, right after the compaction, not later mid-conversation
        for sec in (1, 3, 6, 15):
            m.recognize(np.zeros(RATE * sec, np.float32), sample_rate=RATE)
        return m

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

    def transcribe(self, pcm: bytes) -> dict:
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
        t = time.time()
        with self.stt_lock:
            text = str(self.parakeet.recognize(audio, sample_rate=RATE)).strip()
        return {"text": text, "stt_ms": round((time.time() - t) * 1000), "duration": round(len(audio) / RATE, 2)}


class Silero:
    """streaming Silero VAD v6: 512-sample chunks, state kept"""

    _session = None

    def __init__(self) -> None:
        if Silero._session is None:
            Silero._session = ort.InferenceSession(os.path.join(MODELS, "silero_vad_v6.onnx"), _quiet_session_options(1),
                                                   providers=["CPUExecutionProvider"])
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
    words = [w.strip(".,!?") for w in t.split()]
    if not words or all(w in NOISE_WORDS for w in words):
        return "empty"
    if len(words) >= 5 and max(words.count(w) for w in set(words)) > 0.6 * len(words):
        return "repeated word (noise)"
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
        """-> list of events: ("start",) / ("end", pcm or None)"""
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
    eng: Engines = req.app["engines"]
    ws = web.WebSocketResponse(heartbeat=15, max_msg_size=1 << 20)
    await ws.prepare(req)
    lst = Listener()
    loop = asyncio.get_running_loop()
    log.info("listener connected")
    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            # VAD is ~0.3 ms per chunk: fine on the loop; transcription goes to a thread
            for ev in lst.feed(msg.data):
                if ev[0] == "start":
                    await ws.send_json({"type": "speech_start"})
                    continue
                await ws.send_json({"type": "speech_end"})
                if ev[1] is None:
                    continue
                r = await loop.run_in_executor(None, eng.transcribe, ev[1])
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
    log.info("listener closed")
    return ws


async def tts(req: web.Request) -> web.Response:
    body = await req.json()
    text = str(body.get("text", "")).strip()[:1000]
    if not text:
        raise web.HTTPBadRequest(text="no text")
    eng: Engines = req.app["engines"]
    t = time.time()
    wav = await asyncio.get_running_loop().run_in_executor(
        None, eng.speak, text, str(body.get("voice", "af_heart")), min(max(float(body.get("speed", 1.0)), 0.5), 2.0))
    ms = round((time.time() - t) * 1000)
    log.info("said %r in %d ms", text[:60], ms)
    return web.Response(body=wav, content_type="audio/wav", headers={"X-TTS-ms": str(ms)})


async def health(req: web.Request) -> web.Response:
    eng: Engines = req.app["engines"]
    return web.json_response({"ok": True, "stt": f"{STT_MODEL} ({eng.stt_where})", "tts": f"{TTS_MODEL} ({eng.tts_provider})"})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("phonemizer").setLevel(logging.ERROR)   # "words count mismatch" on every sentence
    app = web.Application()
    app["engines"] = Engines()
    app.router.add_get("/health", health)
    app.router.add_get("/listen", listen)
    app.router.add_post("/tts", tts)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
