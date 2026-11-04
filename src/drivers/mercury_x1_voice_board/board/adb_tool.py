#!/usr/bin/env python3
"""Tiny ADB-over-usbfs client (no adb binary, no pyusb).

Claims only interface 0 (vendor ADB) of the device, so the UAC audio interfaces stay untouched.
  adb_tool.py DEV shell "cmd"      -> command output on stdout
  adb_tool.py DEV pull /remote     -> file bytes on stdout (sync RECV, binary-safe)
  adb_tool.py DEV push /remote[,mode] < file   (sync SEND, binary-safe; mode octal, default 0644)
DEV is /dev/bus/usb/BBB/DDD.
"""
import ctypes
import fcntl
import os
import struct
import sys

USBDEVFS_CLAIMINTERFACE = 0x8004550F
USBDEVFS_RELEASEINTERFACE = 0x80045510
USBDEVFS_BULK = 0xC0185502
EP_OUT, EP_IN, IFACE = 0x01, 0x81, 0
TIMEOUT_MS = 60000


def tag(s):
    return struct.unpack("<I", s)[0]


A_CNXN, A_AUTH, A_OPEN, A_OKAY, A_CLSE, A_WRTE = map(tag, (b"CNXN", b"AUTH", b"OPEN", b"OKAY", b"CLSE", b"WRTE"))


class Bulk(ctypes.Structure):
    _fields_ = [("ep", ctypes.c_uint), ("len", ctypes.c_uint), ("timeout", ctypes.c_uint),
                ("data", ctypes.c_void_p)]


class Adb:
    def __init__(self, dev):
        self.fd = os.open(dev, os.O_RDWR)
        fcntl.ioctl(self.fd, USBDEVFS_CLAIMINTERFACE, struct.pack("I", IFACE))
        self.inbuf = ctypes.create_string_buffer(1 << 20)
        self.send(A_CNXN, 0x01000000, 256 * 1024, b"host::\0")
        for _ in range(16):  # skip stale packets left over from a previous session
            cmd, _, maxdata, data = self.recv()
            if cmd in (A_CNXN, A_AUTH):
                break
        if cmd != A_CNXN:
            raise SystemExit(f"no CNXN (got {cmd:#x}); auth required?")
        self.maxdata = min(maxdata, 256 * 1024)
        self.local_id = 1
        self.drain()

    def drain(self):
        """Discard anything still queued (e.g. a duplicate CNXN) before opening services."""
        while True:
            try:
                self._bulk_in(1 << 20, timeout=300)
            except OSError:
                return

    def close(self):
        fcntl.ioctl(self.fd, USBDEVFS_RELEASEINTERFACE, struct.pack("I", IFACE))
        os.close(self.fd)

    def _bulk_out(self, data):
        buf = ctypes.create_string_buffer(data, len(data))
        fcntl.ioctl(self.fd, USBDEVFS_BULK, Bulk(EP_OUT, len(data), TIMEOUT_MS, ctypes.cast(buf, ctypes.c_void_p)))

    def _bulk_in(self, n, timeout=TIMEOUT_MS):
        got = fcntl.ioctl(self.fd, USBDEVFS_BULK, Bulk(EP_IN, n, timeout, ctypes.cast(self.inbuf, ctypes.c_void_p)))
        return self.inbuf.raw[:got]

    def send(self, cmd, a0, a1, payload=b""):
        self._bulk_out(struct.pack("<6I", cmd, a0, a1, len(payload), sum(payload) & 0xFFFFFFFF, cmd ^ 0xFFFFFFFF))
        if payload:
            self._bulk_out(payload)

    def recv(self):
        cmd, a0, a1, ln, _crc, _magic = struct.unpack("<6I", self._bulk_in(24))
        data = b""
        while len(data) < ln:
            data += self._bulk_in(ln - len(data))
        return cmd, a0, a1, data

    def open(self, service):
        self.send(A_OPEN, self.local_id, 0, service.encode() + b"\0")
        cmd, remote, _, _ = self.recv()
        if cmd != A_OKAY:
            raise SystemExit(f"OPEN {service!r} refused ({cmd:#x})")
        return remote

    def stream(self, remote):
        """Yield WRTE payloads, acking each, until CLSE."""
        while True:
            cmd, a0, _, data = self.recv()
            if cmd == A_WRTE:
                self.send(A_OKAY, self.local_id, a0)
                yield data
            elif cmd == A_CLSE:
                return

    def shell(self, command, out):
        remote = self.open("shell:" + command)
        for chunk in self.stream(remote):
            out.write(chunk)

    def shell_raw(self, command, out):
        """Shell output with the PTY's ONLCR (\n -> \r\n) undone, so binary output survives."""
        remote = self.open("shell:" + command)
        carry = b""
        total = 0
        for chunk in self.stream(remote):
            data = carry + chunk
            carry = b"\r" if data.endswith(b"\r") else b""
            if carry:
                data = data[:-1]
            data = data.replace(b"\r\n", b"\n")
            out.write(data)
            total += len(data)
        out.write(carry)
        return total + len(carry)

    def pull(self, path, out):
        remote = self.open("sync:")
        p = path.encode()
        self.send(A_WRTE, self.local_id, remote, b"RECV" + struct.pack("<I", len(p)) + p)
        cmd, _, _, _ = self.recv()  # OKAY for our WRTE
        buf = b""
        total = 0
        while True:
            cmd, a0, _, data = self.recv()
            if cmd == A_OKAY:
                continue
            if cmd == A_CLSE:
                raise SystemExit("stream closed early")
            self.send(A_OKAY, self.local_id, a0)
            buf += data
            while len(buf) >= 8:
                kind, ln = buf[:4], struct.unpack("<I", buf[4:8])[0]
                if kind == b"DATA":
                    if len(buf) < 8 + ln:
                        break
                    out.write(buf[8:8 + ln])
                    total += ln
                    buf = buf[8 + ln:]
                elif kind == b"DONE":
                    self.send(A_WRTE, self.local_id, remote, b"QUIT" + struct.pack("<I", 0))
                    while self.recv()[0] != A_CLSE:  # drain OKAY + CLSE
                        pass
                    return total
                elif kind == b"FAIL":
                    if len(buf) < 8 + ln:
                        break
                    raise SystemExit("FAIL: " + buf[8:8 + ln].decode(errors="replace"))
                else:
                    raise SystemExit(f"bad sync id {kind!r}")

    def _wrte(self, remote, payload):
        """WRTE and wait for the peer's OKAY (ADB flow control)."""
        self.send(A_WRTE, self.local_id, remote, payload)
        while True:
            cmd, a0, _, data = self.recv()
            if cmd == A_OKAY:
                return
            if cmd == A_WRTE:  # an early sync reply (FAIL) - ack and surface it
                self.send(A_OKAY, self.local_id, a0)
                raise SystemExit("push refused: " + data[8:].decode(errors="replace"))
            if cmd == A_CLSE:
                raise SystemExit("stream closed early")

    def push(self, spec, data):
        path, _, mode = spec.partition(",")
        mode = int(mode or "644", 8)
        remote = self.open("sync:")
        hdr = f"{path},{0o100000 | mode}".encode()
        self._wrte(remote, b"SEND" + struct.pack("<I", len(hdr)) + hdr)
        step = max(256, self.maxdata - 8)
        for i in range(0, len(data), step):
            chunk = data[i:i + step]
            self._wrte(remote, b"DATA" + struct.pack("<I", len(chunk)) + chunk)
        self._wrte(remote, b"DONE" + struct.pack("<I", 0))
        while True:  # sync status: OKAY or FAIL
            cmd, a0, _, reply = self.recv()
            if cmd == A_WRTE:
                self.send(A_OKAY, self.local_id, a0)
                if reply[:4] != b"OKAY":
                    raise SystemExit("push failed: " + reply[8:].decode(errors="replace"))
                break
            if cmd == A_CLSE:
                raise SystemExit("stream closed before status")
        self.send(A_WRTE, self.local_id, remote, b"QUIT" + struct.pack("<I", 0))
        while self.recv()[0] != A_CLSE:
            pass
        return len(data)


def main():
    dev, op, arg = sys.argv[1], sys.argv[2], sys.argv[3]
    adb = Adb(dev)
    out = sys.stdout.buffer
    try:
        if op == "shell":
            adb.shell(arg, out)
        elif op == "shellraw":
            n = adb.shell_raw(arg, out)
            sys.stderr.write(f"received {n} bytes\n")
        elif op == "pull":
            n = adb.pull(arg, out)
            sys.stderr.write(f"pulled {n} bytes from {arg}\n")
        elif op == "push":
            n = adb.push(arg, sys.stdin.buffer.read())
            sys.stderr.write(f"pushed {n} bytes to {arg}\n")
        else:
            raise SystemExit("op must be shell, shellraw, pull or push")
        out.flush()
    finally:
        adb.close()


if __name__ == "__main__":
    main()
