"""Second look at a wake word. The voice board's small wake model fires easily (low threshold, good recall);
before the robot reacts, the Orin's speech recognizer (mercury-speech, Parakeet) transcribes the last seconds
of a raw mic and the wake counts only if it hears something like "mercury". The mic is read all the time into
a short ring buffer (ALSA dsnoop: the voice chat's own capture can run alongside). Raw mic, not the board's
beam: measured 2026-11-05, the mics reach the Orin ~0.6 s after the sound and the wake event ~0.05 s after the
word ends, but the beam (board aplay -> USB gadget) 1.5 s later still, and that grows across board restarts."""
import array
import asyncio
import difflib
import math
import os
import re
import time
from collections import deque

import aiohttp

RING_S = 4.0        # audio kept
TAIL_S = 1.0        # the wake event arrives as the word ends; the mic audio of that end ~0.6 s later
TAKE_S = 3.0        # audio sent to the recognizer: covers mic-path lags of 0.3 .. 1.3 s
FRAMES = 800        # 50 ms at 16 kHz per ring chunk
CHUNK = FRAMES * 2  # ... as mono PCM16 bytes
MIC_DEVICE, MIC_CHANNELS, MIC_CHANNEL, MIC_GAIN = "mx1_mics", 5, 0, 25.0   # raw mic 0 of the voice board, +28 dB (mic PGA 0 since 2026-11-05)
DEBUG_FILE = "/tmp/mercury_wake_last.raw"          # written while /tmp/mercury_wake_last.raw.on exists (PCM16 16 kHz)
# what Parakeet makes of "mercury" said from across the room
HEARD = re.compile(r"merc(?:ur|er|ury|y)|murcur|mark\s*jury|mercuri|merkur|mccur", re.I)
LIKE = 0.72         # or a word (or two words joined) this similar to "mercury"


# consonant skeleton of "mercury" (m-r-k-r): what the recognizer makes of it from across the room ("Marcura",
# "Marjorie", "markets", "margarine", "Ama Curie") keeps the m-r-k / m-r-j / m-k-r onset; "Mercedes" (m-r-s),
# "Murphy" (m-r-f), "merci" (m-r-s), "Mary" (m-r) do not
SKELETON = re.compile(r"^m(?:r[kj]|[kj]r)[^aeiou]+")   # 4+ consonants: "mark", "Marco" are too short


def skeleton(word: str) -> str:
    w = word.lower()
    w = re.sub(r"c(?=[eiy])", "s", w)                                 # Mercedes, Marcel: a soft c
    for a, b in (("ph", "f"), ("ck", "k"), ("ch", "k"), ("x", "ks"), ("qu", "k"), ("c", "k"), ("q", "k"), ("g", "j")):
        w = w.replace(a, b)
    w = re.sub(r"[aeiouhwy']", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def _biquad(x: list[float], b0: float, b1: float, b2: float, a1: float, a2: float) -> list[float]:
    z1 = z2 = 0.0
    out = []
    for v in x:
        y = b0 * v + z1
        z1 = b1 * v - a1 * y + z2
        z2 = b2 * v - a2 * y
        out.append(y)
    return out


def dehum(pcm: bytes, fs: int = 16000, f0: float = 250.5) -> bytes:
    """The robot's ~250 Hz hum out of a mono PCM16 clip: the same 100 Hz high-pass + notches at f0 -+ 3 Hz (12 Hz
    wide) that voicefront's --hum applies to the mics (hum.h); it is 55..65 % of the noise power the recognizer sees"""
    x = [float(v) for v in array.array("h", pcm)]
    k = math.tan(math.pi * 100.0 / fs); k2 = k * k; n = 1.0 / (1.0 + math.sqrt(2.0) * k + k2)
    x = _biquad(x, n, -2.0 * n, n, 2.0 * (k2 - 1.0) * n, (1.0 - math.sqrt(2.0) * k + k2) * n)
    r = math.exp(-math.pi * 12.0 / fs)
    for f in (f0 - 3.0, f0 + 3.0):
        c = math.cos(2.0 * math.pi * f / fs); g = (1.0 - 2.0 * r * c + r * r) / (2.0 - 2.0 * c)
        x = _biquad(x, g, -2.0 * c * g, g, -2.0 * r * c, r * r)
    return array.array("h", (max(-32768, min(32767, int(round(v)))) for v in x)).tobytes()


def heard_mercury(text: str) -> bool:
    if HEARD.search(text):
        return True
    words = re.findall(r"[a-z']+", text.lower())
    cands = words + [a + b for a, b in zip(words, words[1:])]
    cands += [w[1:] for w in cands if w[:1] == "a" and len(w) > 4]      # "Amarty", "Ama Curie": a leading vowel
    if any(w[:1] == "m" and difflib.SequenceMatcher(None, w, "mercury").ratio() >= LIKE for w in cands):
        return True
    # the phonetic rule: single words, or a word plus a vowel-led tail ("mark jury", "marks are", "mark your")
    tails = [a + b for a, b in zip(words, words[1:]) if b[:1] in "aeiouyj"]
    return any(w[:1] == "m" and SKELETON.match(skeleton(w)) for w in words + tails)


class WakeVerifier:
    def __init__(self, speech_url: str, mic: str = MIC_DEVICE, channels: int = MIC_CHANNELS, channel: int = MIC_CHANNEL,
                 gain: float = MIC_GAIN) -> None:
        self.mic, self.channels, self.channel, self.gain = mic, channels, channel, gain
        self.speech_url = speech_url.rstrip("/")
        self._ring: deque[bytes] = deque(maxlen=int(RING_S * 32000 / CHUNK))
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self.error = ""
        self.last: dict = {}         # the last check, for the page

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                # stderr is dropped: an unread pipe fills with overrun warnings and then blocks arecord (a frozen
                # ring); the 1 s ALSA buffer rides out event-loop hiccups
                self._proc = await asyncio.create_subprocess_exec(
                    "arecord", "-q", "-D", self.mic, "-f", "S16_LE", "-r", "16000", "-c", str(self.channels), "-t", "raw",
                    "--buffer-size=16000", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                assert self._proc.stdout
                self.error = ""
                while True:
                    try:
                        raw = await self._proc.stdout.readexactly(CHUNK * self.channels)   # read() may return less
                    except asyncio.IncompleteReadError:
                        break
                    a = array.array("h", raw)[self.channel::self.channels]
                    if self.gain != 1.0:
                        a = array.array("h", (max(-32768, min(32767, int(v * self.gain))) for v in a))
                    self._ring.append(a.tobytes())
                self.error = f"mic: arecord exited ({await self._proc.wait()})"
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.error = f"mic: {str(e)[:100]}"
            self._ring.clear()
            await asyncio.sleep(3.0)

    async def verify(self) -> tuple[bool, str]:
        """(accepted, what was heard) for a wake candidate that just fired"""
        await asyncio.sleep(TAIL_S)
        pcm = dehum(b"".join(self._ring)[-int(TAKE_S * 32000):])
        t = time.time()
        if os.path.exists(DEBUG_FILE + ".on"):
            try:
                with open(DEBUG_FILE, "wb") as f:
                    f.write(pcm)
            except OSError:
                pass
        if len(pcm) < 16000:
            self.last = {"t": t, "ok": True, "heard": "", "why": self.error or "no audio yet"}
            return True, ""            # cannot check: let it through rather than go deaf
        try:
            async with aiohttp.ClientSession() as s, s.post(self.speech_url + "/stt", data=pcm,
                                                            timeout=aiohttp.ClientTimeout(total=4)) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                text = str((await r.json()).get("text", "")).strip()
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            self.last = {"t": t, "ok": True, "heard": "", "why": f"speech: {str(e)[:60] or type(e).__name__}"}
            return True, ""
        ok = heard_mercury(text)
        self.last = {"t": t, "ok": ok, "heard": text, "ms": round((time.time() - t) * 1000)}
        return ok, text
