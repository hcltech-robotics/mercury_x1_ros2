"""iFlytek AIUI UART protocol spoken by the head voice board (XFM-DP firmware on an Allwinner R818).

Frame (multi-byte fields little-endian):
    A5 | 01 | type | len (2) | msg id (2) | payload (len) | checksum
    checksum = (~sum(all previous bytes) + 1) & 0xFF

Types: 0x01 handshake (board -> host), 0x04 event JSON (board -> host), 0x05 control JSON (host -> board),
0xFF confirm (both ways; same msg id as the frame it confirms, payload A5 00 00 00).
The board answers every control message with a confirm and then an event
{"code": 0, "content": "<text>", "type": "<request type>"}.
See ~/Documents/reverse_engineer/mercury_x1/voice_board/HOW_IT_WORKS.md for the full analysis.
"""
import json
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

SYNC = 0xA5
USER_ID = 0x01
T_HANDSHAKE = 0x01
T_EVENT = 0x04
T_CONTROL = 0x05
T_CONFIRM = 0xFF
CONFIRM_PAYLOAD = bytes([0xA5, 0x00, 0x00, 0x00])
HEADER_LEN = 7
MAX_PAYLOAD = 8192
EVENT_WAKEUP = 4


def checksum(data: bytes) -> int:
    return (~sum(data) + 1) & 0xFF


def build_frame(msg_type: int, msg_id: int, payload: bytes) -> bytes:
    head = bytes([SYNC, USER_ID, msg_type]) + len(payload).to_bytes(2, "little") \
        + (msg_id & 0xFFFF).to_bytes(2, "little")
    body = head + payload
    return body + bytes([checksum(body)])


def build_confirm(msg_id: int) -> bytes:
    return build_frame(T_CONFIRM, msg_id, CONFIRM_PAYLOAD)


def build_control(msg_id: int, message: dict) -> bytes:
    return build_frame(T_CONTROL, msg_id, json.dumps(message, ensure_ascii=False,
                                                     separators=(",", ":")).encode("utf-8"))


class FrameParser:
    """Incremental parser: feed() raw bytes, get (type, msg_id, payload) for every valid frame.

    Garbage and frames with a bad checksum are skipped by resynchronising on the next A5 01."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.dropped_bytes = 0

    def feed(self, data: bytes) -> Iterator[Tuple[int, int, bytes]]:
        self._buf += data
        while True:
            start = self._buf.find(bytes([SYNC, USER_ID]))
            if start < 0:
                keep = 1 if self._buf[-1:] == bytes([SYNC]) else 0
                self.dropped_bytes += len(self._buf) - keep
                del self._buf[:len(self._buf) - keep]
                return
            if start:
                self.dropped_bytes += start
                del self._buf[:start]
            if len(self._buf) < HEADER_LEN + 1:
                return
            length = int.from_bytes(self._buf[3:5], "little")
            if length > MAX_PAYLOAD:
                self.dropped_bytes += 1
                del self._buf[:1]
                continue
            total = HEADER_LEN + length + 1
            if len(self._buf) < total:
                return
            frame = bytes(self._buf[:total])
            if checksum(frame[:-1]) != frame[-1]:
                self.dropped_bytes += 1
                del self._buf[:1]
                continue
            del self._buf[:total]
            yield frame[2], int.from_bytes(frame[5:7], "little"), frame[HEADER_LEN:-1]


def decode_json(payload: bytes) -> Optional[dict]:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class WakeInfo:
    keyword: str
    angle_deg: float
    beam: int
    score: float
    start_ms: int
    raw: str


def _maybe_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _num(value, default, cast):
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def parse_wakeup(event: dict) -> Optional[WakeInfo]:
    """Wake info from an aiui_event with eventType 4, else None.

    Field placement differs between firmware builds, so look in content.info.ivw, content.info and
    content (arg1 = beam): e.g.
    {"type":"aiui_event","content":{"eventType":4,"arg1":1,"info":"{\\"ivw\\":{\\"angle\\":95,...}}"}}"""
    if event.get("type") != "aiui_event":
        return None
    content = event.get("content")
    if not isinstance(content, dict) or _num(content.get("eventType"), -1, int) != EVENT_WAKEUP:
        return None
    info = _maybe_json(content.get("info"))
    info = info if isinstance(info, dict) else {}
    ivw = info.get("ivw") if isinstance(info.get("ivw"), dict) else {}
    result = _maybe_json(content.get("result"))
    result = result if isinstance(result, dict) else {}

    def pick(key):
        for source in (ivw, info, result, content):
            if key in source:
                return source[key]
        return None

    beam = pick("beam")
    if beam is None:
        beam = content.get("arg1")
    return WakeInfo(
        keyword=str(pick("keyword") or ""),
        angle_deg=_num(pick("angle"), float("nan"), float),
        beam=_num(beam, -1, int),
        score=_num(pick("score"), float("nan"), float),
        start_ms=_num(pick("start_ms"), -1, int),
        raw=json.dumps(event, ensure_ascii=False, separators=(",", ":")),
    )
