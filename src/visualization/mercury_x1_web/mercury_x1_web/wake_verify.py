"""Second look at a wake word. The voice board's small wake model fires easily (low threshold, good recall);
before the robot reacts, the Orin's speech recognizer (mercury-speech, Parakeet) transcribes the last seconds
of the board's beam and the wake counts only if it hears something like "mercury". The beam is read all the
time into a short ring buffer (ALSA dsnoop: the voice chat's own capture can run alongside)."""
import asyncio
import difflib
import re
import time
from collections import deque

import aiohttp

RING_S = 3.0        # audio kept
TAIL_S = 0.3        # the wake event arrives as the word ends: wait for its tail (board -> USB audio latency)
TAKE_S = 2.4        # audio sent to the recognizer
CHUNK = 1600        # 50 ms of PCM16 at 16 kHz
# what Parakeet makes of "mercury" said from across the room
HEARD = re.compile(r"merc(?:ur|er|ury|y)|murcur|mark\s*jury|mercuri|merkur|mccur", re.I)
LIKE = 0.72         # or a word (or two words joined) this similar to "mercury"


def heard_mercury(text: str) -> bool:
    if HEARD.search(text):
        return True
    words = re.findall(r"[a-z']+", text.lower())
    cands = words + [a + b for a, b in zip(words, words[1:])]
    return any(w[:1] == "m" and difflib.SequenceMatcher(None, w, "mercury").ratio() >= LIKE for w in cands)


class WakeVerifier:
    def __init__(self, mic: str, speech_url: str) -> None:
        self.mic, self.speech_url = mic, speech_url.rstrip("/")
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
                self._proc = await asyncio.create_subprocess_exec(
                    "arecord", "-q", "-D", self.mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                assert self._proc.stdout
                self.error = ""
                while True:
                    chunk = await self._proc.stdout.read(CHUNK)
                    if not chunk:
                        break
                    self._ring.append(chunk)
                err = (await self._proc.stderr.read()).decode(errors="replace").strip() if self._proc.stderr else ""
                self.error = f"mic: {err[:100] or 'arecord exited'}"
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.error = f"mic: {str(e)[:100]}"
            self._ring.clear()
            await asyncio.sleep(3.0)

    async def verify(self) -> tuple[bool, str]:
        """(accepted, what was heard) for a wake candidate that just fired"""
        await asyncio.sleep(TAIL_S)
        pcm = b"".join(self._ring)[-int(TAKE_S * 32000):]
        t = time.time()
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
