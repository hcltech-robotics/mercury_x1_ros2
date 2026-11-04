"""Serial transport for the voice board: raw termios (no pyserial), a reader thread and reconnects on errors.

Two board firmwares, told apart by what arrives:
  "json": voicefront (ours): one JSON object per line; commands {"cmd": ..., "id": n} get {"type": "reply", "id": n}
  "aiui": the stock iFlytek demo: binary AIUI frames; handshakes/events are confirmed immediately (the board
          expects a confirm within 50 ms), requests are answered by an event of the same "type"
"""
import errno
import json
import os
import select
import termios
import threading
import time
from typing import Callable, Dict, List, Optional

from . import aiui_protocol as proto


def open_raw(path: str) -> int:
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0  # iflag
        attrs[1] = 0  # oflag
        attrs[2] = (attrs[2] & ~(termios.CSIZE | termios.PARENB | termios.CSTOPB | termios.CRTSCTS)) \
            | termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0  # lflag
        attrs[4] = attrs[5] = termios.B115200
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
    except termios.error:
        os.close(fd)
        raise
    return fd


class _Pending:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.reply: Optional[dict] = None


class VoiceBoardLink:
    """Owns the port. Callbacks run on the reader thread:
    on_event(dict) for every board event, on_state(str) for open/close/error changes."""

    def __init__(self, port: str, on_event: Callable[[dict], None],
                 on_state: Callable[[str], None], reopen_delay: float = 1.0) -> None:
        self.port = port
        self._on_event = on_event
        self._on_state = on_state
        self._reopen_delay = reopen_delay
        self._fd: Optional[int] = None
        self._write_lock = threading.Lock()
        self._pending: Dict[str, List[_Pending]] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 1
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="voice_board_rx", daemon=True)
        self.parser = proto.FrameParser()
        self.last_rx = 0.0
        self.handshakes = 0
        self.mode = "unknown"  # "json" (voicefront) | "aiui" (stock demo), from the received data
        self._line = b""
        self.is_open = False
        self._opened = threading.Event()

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._close()

    # -- requests ----------------------------------------------------------------------------
    def request(self, message: dict, timeout: float) -> Optional[dict]:
        """Send a control message and wait for its reply: in json mode the reply with the same "id", in aiui
        mode the board's event with the same "type". Waits for the port to open and the firmware to be seen
        within the same timeout; returns None if there is no reply."""
        deadline = time.monotonic() + timeout
        if not self._opened.wait(timeout):
            return None
        while self.mode == "unknown" and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.mode == "json":
            message = dict(message)
            with self._write_lock:
                message["id"] = self._next_id
                self._next_id = self._next_id % 0xFFFF + 1
            kind = f"id:{message['id']}"
        else:
            kind = str(message.get("type", ""))
        pending = _Pending()
        with self._pending_lock:
            self._pending.setdefault(kind, []).append(pending)
        try:
            if not self.send(message):
                return None
            pending.done.wait(max(0.0, deadline - time.monotonic()))
            return pending.reply
        finally:
            with self._pending_lock:
                waiters = self._pending.get(kind, [])
                if pending in waiters:
                    waiters.remove(pending)
                if not waiters:
                    self._pending.pop(kind, None)

    def send(self, message: dict) -> bool:
        if self.mode == "json":
            line = json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n"
            return self._write(line.encode())
        with self._write_lock:
            msg_id = self._next_id
            self._next_id = self._next_id % 0xFFFF + 1
        return self._write(proto.build_control(msg_id, message))

    # -- internals ---------------------------------------------------------------------------
    def _write(self, data: bytes) -> bool:
        with self._write_lock:
            fd = self._fd
            if fd is None:
                return False
            try:
                view = memoryview(data)
                deadline = time.monotonic() + 0.5
                while view:
                    try:
                        n = os.write(fd, view)
                        view = view[n:]
                    except BlockingIOError:
                        if time.monotonic() > deadline:
                            return False
                        select.select([], [fd], [], 0.05)
                return True
            except OSError:
                return False

    def _open(self) -> bool:
        try:
            self._fd = open_raw(self.port)
        except OSError as exc:
            self._on_state(f"cannot open {self.port}: {exc.strerror}")
            return False
        self.parser = proto.FrameParser()
        self._line = b""
        self.is_open = True
        self._opened.set()
        self._on_state("open")
        return True

    def _close(self) -> None:
        with self._write_lock:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None
        self._opened.clear()
        if self.is_open:
            self.is_open = False
            self._on_state("closed")

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._fd is None and not self._open():
                self._stop.wait(self._reopen_delay)
                continue
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.2)
                if not ready:
                    continue
                data = os.read(self._fd, 4096)
                if not data:  # hang-up (USB unplugged)
                    raise OSError(errno.EIO, "device disconnected")
            except OSError as exc:
                self._on_state(f"read error: {exc.strerror or exc}")
                self._close()
                self._stop.wait(self._reopen_delay)
                continue
            self.last_rx = time.monotonic()
            if b"\xa5\x01" in data and self.mode != "aiui":
                self.mode, self.parser = "aiui", proto.FrameParser()
                self._on_state("firmware: stock demo (AIUI frames)")
            elif b'{"type":' in data and self.mode != "json":
                self.mode, self._line = "json", b""
                self._on_state("firmware: voicefront (JSON lines)")
            if self.mode == "json":
                self._feed_json(data)
            else:
                for msg_type, msg_id, payload in self.parser.feed(data):
                    self._handle(msg_type, msg_id, payload)

    def _feed_json(self, data: bytes) -> None:
        self._line += data
        *lines, self._line = self._line.split(b"\n")
        if len(self._line) > 8192:  # no newline in sight: garbage
            self._line = b""
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                self.parser.dropped_bytes += len(raw)
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "hello":
                self.handshakes += 1
            if event.get("type") == "reply" and "id" in event:
                with self._pending_lock:
                    waiters = list(self._pending.get(f"id:{event['id']}", []))
                for pending in waiters:
                    pending.reply = event
                    pending.done.set()
            self._on_event(event)

    def _handle(self, msg_type: int, msg_id: int, payload: bytes) -> None:
        if msg_type == proto.T_CONFIRM:
            return
        self._write(proto.build_confirm(msg_id))
        if msg_type == proto.T_HANDSHAKE:
            self.handshakes += 1
            return
        if msg_type != proto.T_EVENT:
            return
        event = proto.decode_json(payload)
        if event is None:
            event = {"type": "undecodable", "content": payload.hex(" ")}
        kind = str(event.get("type", ""))
        with self._pending_lock:
            waiters = list(self._pending.get(kind, []))
        for pending in waiters:  # every waiter for this type gets the reply
            pending.reply = event
            pending.done.set()
        self._on_event(event)


def event_text(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))
