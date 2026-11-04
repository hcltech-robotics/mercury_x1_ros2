"""voicefront (JSON lines) side of the serial link: firmware detection, event parsing, reply matching."""
import threading

from mercury_x1_voice_board.serial_link import VoiceBoardLink


def make_link():
    events, states = [], []
    link = VoiceBoardLink("/dev/null", events.append, states.append)
    return link, events, states


def test_json_lines_parse_and_hello_counts():
    link, events, _ = make_link()
    link.mode = "json"
    link._feed_json(b'{"type":"hello","version":"0.1.0"}\n{"type":"wa')
    link._feed_json(b'ke","keyword":"hey mercury","angle":87.5}\nnot json\n')
    assert [e["type"] for e in events] == ["hello", "wake"]
    assert events[1]["angle"] == 87.5
    assert link.handshakes == 1
    assert link.parser.dropped_bytes == len(b"not json")


def test_reply_matches_request_id():
    link, events, _ = make_link()
    link.mode = "json"
    link._opened.set()
    sent = []
    link._write = lambda data: sent.append(data) or True  # no port

    def answer():
        while not sent:
            pass
        rid = int(sent[0].split(b'"id":')[1].split(b"}")[0])
        link._feed_json(b'{"type":"reply","cmd":"other","ok":true,"id":%d}\n' % (rid + 7))
        link._feed_json(b'{"type":"reply","cmd":"ping","ok":true,"id":%d}\n' % rid)

    threading.Thread(target=answer, daemon=True).start()
    reply = link.request({"cmd": "ping"}, timeout=2.0)
    assert reply is not None and reply["cmd"] == "ping" and reply["ok"] is True
    assert sent[0].endswith(b"\n") and b'"cmd":"ping"' in sent[0]
