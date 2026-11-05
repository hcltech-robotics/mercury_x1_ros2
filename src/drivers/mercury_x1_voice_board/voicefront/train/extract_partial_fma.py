"""Extract mp3 entries from a partially downloaded fma_small.zip (bzip2-compressed entries, sizes in local
headers) and convert to 16 kHz mono wav with ffmpeg. usage: extract_partial_fma.py <zip> <outdir> <hours>"""
import struct, sys, subprocess, random, bz2, zlib
from pathlib import Path
zp, out, hours = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
out.mkdir(exist_ok=True)
want = int(hours * 3600 // 30)
entries = []
size = zp.stat().st_size
with open(zp, "rb") as f:
    pos = 0
    while pos + 30 < size:
        f.seek(pos); h = f.read(30)
        if h[:4] != b"PK\x03\x04": break
        flags, method = struct.unpack("<HH", h[6:10]); csize, usize = struct.unpack("<II", h[18:26]); nlen, xlen = struct.unpack("<HH", h[26:30])
        name = f.read(nlen).decode("utf-8", "replace")
        d0 = pos + 30 + nlen + xlen
        if flags & 8 or d0 + csize > size: break
        if name.endswith(".mp3") and method in (0, 8, 12): entries.append((name, d0, csize, method))
        pos = d0 + csize
    print("complete mp3 entries:", len(entries), flush=True)
    random.seed(0); random.shuffle(entries)
    k = 0
    for name, d0, n, method in entries:
        if k >= want: break
        f.seek(d0); raw = f.read(n)
        try:
            data = bz2.decompress(raw) if method == 12 else zlib.decompress(raw, -15) if method == 8 else raw
        except Exception as e:
            print("decompress fail", name, e); continue
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"], input=data, capture_output=True)
        if r.returncode != 0 or len(r.stdout) < 100000: continue
        (out / (Path(name).stem + ".wav")).write_bytes(r.stdout); k += 1
print("fma wavs written:", k)
