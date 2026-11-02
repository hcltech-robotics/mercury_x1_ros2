#!/usr/bin/env python3
"""Kokoro speech on the robot's own GPU (Jetson Orin, onnxruntime CUDA), for the head display's voice chat.

  POST /tts       {"text", "voice"?, "speed"?} -> audio/wav (24 kHz mono); same API as server/voice_service
  GET  /health

Listens on 127.0.0.1 only (no token). Runs in its own venv (~/mercury-tts, `setup.sh`), not in ROS' Python:
the Jetson onnxruntime-gpu wheel needs NumPy 1.x. Kokoro fp32 on CUDA, with the robot stack running: ~0.35 s
for a short sentence, ~0.75 s for 5 s of speech (RTF ~0.15; 11.51's CPU does ~0.2 s), ~850 MB of the shared
memory with the GPU arena capped. It is GPU-bound (the GPU is shared with nvblox): CPU threads and nice level
make no difference. The fp16 model is slower here (casts); TensorRT rebuilds engines for every new input length.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
import wave

import numpy as np
import onnxruntime as ort
from aiohttp import web
from kokoro_onnx import Kokoro

HOME = os.path.expanduser("~/mercury-tts")
MODELS = os.environ.get("TTS_MODELS", os.path.join(HOME, "models"))
MODEL = os.environ.get("TTS_MODEL", "kokoro-v1.0.onnx")
PORT = int(os.environ.get("TTS_PORT", "8097"))
GPU_MEM_MB = int(os.environ.get("TTS_GPU_MEM_MB", "768"))
WARMUP = ["Ready.", "Hello, I am Mercury, the robot in the HCLTech robotics lab.",
          "I can talk with you, but I cannot move yet, so please ask me something about the lab or the robots here."]

log = logging.getLogger("tts")


class Engine:
    def __init__(self) -> None:
        t = time.time()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # the few CPU ops (STFT, Resize) are small: 1 thread, no busy-waiting, so ROS keeps its cores
        opts.intra_op_num_threads = int(os.environ.get("TTS_THREADS", "1"))
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        cuda = {"cudnn_conv_algo_search": "HEURISTIC", "cudnn_conv_use_max_workspace": "0",
                "arena_extend_strategy": "kSameAsRequested", "gpu_mem_limit": str(GPU_MEM_MB << 20)}
        sess = ort.InferenceSession(os.path.join(MODELS, MODEL), opts,
                                    providers=[("CUDAExecutionProvider", cuda), "CPUExecutionProvider"])
        self.provider = sess.get_providers()[0]
        self.kokoro = Kokoro.from_session(sess, os.path.join(MODELS, "voices-v1.0.bin"))
        self.lock = threading.Lock()
        # the first runs per input length pick their kernels (~1 s each); do them now, not on the first reply
        for _ in range(2):
            for s in WARMUP:
                self.kokoro.create(s, voice="af_heart", lang="en-us")
        log.info("ready in %.1f s (%s, %s, GPU arena <= %d MB)", time.time() - t, MODEL, self.provider, GPU_MEM_MB)

    def speak(self, text: str, voice: str, speed: float) -> bytes:
        with self.lock:
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


async def tts(req: web.Request) -> web.Response:
    body = await req.json()
    text = str(body.get("text", "")).strip()[:1000]
    if not text:
        raise web.HTTPBadRequest(text="no text")
    eng: Engine = req.app["engine"]
    t = time.time()
    wav = await asyncio.get_running_loop().run_in_executor(
        None, eng.speak, text, str(body.get("voice", "af_heart")), min(max(float(body.get("speed", 1.0)), 0.5), 2.0))
    ms = round((time.time() - t) * 1000)
    log.info("%d ms for %d chars", ms, len(text))
    return web.Response(body=wav, content_type="audio/wav", headers={"X-TTS-ms": str(ms)})


async def health(req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "tts": MODEL, "provider": req.app["engine"].provider})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("phonemizer").setLevel(logging.ERROR)   # "words count mismatch" on every sentence
    app = web.Application()
    app["engine"] = Engine()
    app.router.add_get("/health", health)
    app.router.add_post("/tts", tts)
    web.run_app(app, host="127.0.0.1", port=PORT, access_log=None, print=None)


if __name__ == "__main__":
    main()
