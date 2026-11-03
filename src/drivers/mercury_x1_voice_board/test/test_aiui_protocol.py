import json
import math
import os
import threading
import time

from mercury_x1_voice_board import aiui_protocol as proto
from mercury_x1_voice_board.serial_link import VoiceBoardLink

# captured from the real board on /dev/ttyACM3
HANDSHAKE = bytes.fromhex("a5 01 01 04 00 00 00 a5 00 00 00 b0")


def test_checksum_matches_captured_handshake():
    assert proto.checksum(HANDSHAKE[:-1]) == HANDSHAKE[-1]
    assert proto.build_frame(proto.T_HANDSHAKE, 0, proto.CONFIRM_PAYLOAD) == HANDSHAKE


def test_confirm_frame():
    assert proto.build_confirm(0) == bytes.fromhex("a5 01 ff 04 00 00 00 a5 00 00 00 b2")
    frame = proto.build_confirm(0x9527)
    assert frame[5:7] == bytes([0x27, 0x95])
    assert proto.checksum(frame[:-1]) == frame[-1]


def test_control_frame_round_trip():
    frame = proto.build_control(7, {"type": "manual_wakeup", "content": {"beam": 1}})
    (msg_type, msg_id, payload), = proto.FrameParser().feed(frame)
    assert (msg_type, msg_id) == (proto.T_CONTROL, 7)
    assert json.loads(payload) == {"type": "manual_wakeup", "content": {"beam": 1}}


def test_parser_handles_split_noise_and_bad_checksum():
    event = proto.build_frame(proto.T_EVENT, 3, b'{"type":"version","code":0}')
    bad = bytearray(event)
    bad[-1] ^= 0xFF
    stream = b"\x00\xff garbage" + HANDSHAKE + bytes(bad) + b"\xa5" + event
    parser = proto.FrameParser()
    frames = []
    for i in range(0, len(stream), 5):  # arrives in small pieces
        frames += list(parser.feed(stream[i:i + 5]))
    assert [(t, i) for t, i, _ in frames] == [(proto.T_HANDSHAKE, 0), (proto.T_EVENT, 3)]
    assert parser.dropped_bytes > 0


def test_parse_wakeup_info_as_json_string():
    info = {"ivw": {"angle": 130, "beam": 2, "score": 1234.5, "start_ms": 98765,
                    "keyword": "xiao3 wei1 xiao3 wei1"}}
    event = {"type": "aiui_event", "content": {"eventType": 4, "arg1": 2, "info": json.dumps(info)}}
    wake = proto.parse_wakeup(event)
    assert wake.keyword == "xiao3 wei1 xiao3 wei1"
    assert (wake.angle_deg, wake.beam, wake.score, wake.start_ms) == (130.0, 2, 1234.5, 98765)


def test_parse_wakeup_flat_info_and_arg1_beam():
    event = {"type": "aiui_event", "content": {"eventType": 4, "arg1": 1, "arg2": 0,
                                               "info": {"keyword": "xiao3 fei1 xiao3 fei1", "angle": "45"},
                                               "result": {}}}
    wake = proto.parse_wakeup(event)
    assert wake.keyword == "xiao3 fei1 xiao3 fei1"
    assert wake.angle_deg == 45.0 and wake.beam == 1 and wake.start_ms == -1
    assert math.isnan(wake.score)


def test_parse_wakeup_ignores_other_events():
    assert proto.parse_wakeup({"type": "version", "code": 0, "content": "x"}) is None
    assert proto.parse_wakeup({"type": "aiui_event", "content": {"eventType": 1}}) is None
    assert proto.parse_wakeup({"type": "aiui_event", "content": "oops"}) is None


class FakeBoard(threading.Thread):
    """Board side on a pty: sends 'started' + handshakes, answers control messages like the firmware."""

    def __init__(self, fd):
        super().__init__(daemon=True)
        self.fd = fd
        self.confirmed = set()
        self.controls = []
        self.stop = threading.Event()

    def run(self):
        parser = proto.FrameParser()
        time.sleep(0.3)  # the host flushes input when it opens the port
        os.write(self.fd, proto.build_frame(proto.T_EVENT, 0, b'{"code":0,"content":"version: test","type":"started"}'))
        last_shake = 0.0
        while not self.stop.is_set():
            if 0 not in self.confirmed and time.monotonic() - last_shake > 0.2:
                os.write(self.fd, HANDSHAKE)
                last_shake = time.monotonic()
            try:
                data = os.read(self.fd, 4096)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            for msg_type, msg_id, payload in parser.feed(data):
                if msg_type == proto.T_CONFIRM:
                    self.confirmed.add(msg_id)
                elif msg_type == proto.T_CONTROL:
                    request = json.loads(payload)
                    self.controls.append(request)
                    os.write(self.fd, proto.build_confirm(msg_id))
                    reply = {"code": 0, "content": "beam changed", "type": request["type"]}
                    os.write(self.fd, proto.build_frame(proto.T_EVENT, 50 + msg_id, json.dumps(reply).encode()))


def test_link_confirms_handshake_and_gets_reply():
    board_fd, host_fd = os.openpty()
    os.set_blocking(board_fd, False)
    board = FakeBoard(board_fd)
    board.start()
    events, states = [], []
    link = VoiceBoardLink(os.ttyname(host_fd), events.append, states.append)
    link.start()
    try:
        reply = link.request({"type": "manual_wakeup", "content": {"beam": 1}}, timeout=3.0)
        assert reply == {"code": 0, "content": "beam changed", "type": "manual_wakeup"}
        deadline = time.monotonic() + 2.0
        while 0 not in board.confirmed and time.monotonic() < deadline:
            time.sleep(0.02)
        assert 0 in board.confirmed  # handshake (and the started event, id 0) confirmed
        assert board.controls == [{"type": "manual_wakeup", "content": {"beam": 1}}]
        assert any(e.get("type") == "started" for e in events)
        assert states[0] == "open"
    finally:
        board.stop.set()
        link.stop()
        os.close(host_fd)
        os.close(board_fd)
