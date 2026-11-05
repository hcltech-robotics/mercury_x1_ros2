"""Convert AudioSet parquet shards (flac bytes) and FMA small mp3s into 16 kHz mono int16 wavs."""
import io, sys, zipfile, subprocess, random
from pathlib import Path
import numpy as np, scipy.io.wavfile, soundfile as sf, scipy.signal
import pyarrow.parquet as pq

root = Path.home() / "oww_hey_mercury"

def to16k(y, sr):
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != 16000: y = scipy.signal.resample_poly(y, 16000, sr)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16)

out = root / "audioset_16k"; out.mkdir(exist_ok=True)
n = 0
for p in sorted((root / "audioset").glob("*.parquet")):
    t = pq.read_table(p)
    print(p.name, t.num_rows, t.column_names, flush=True)
    for row in t.to_pylist():
        a = row["audio"]
        try:
            y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        except Exception as e:
            print("skip", e); continue
        name = (a.get("path") or row.get("video_id") or f"a{n}").split("/")[-1].rsplit(".", 1)[0] + ".wav"
        scipy.io.wavfile.write(out / name, 16000, to16k(y, sr)); n += 1
print("audioset clips:", n, flush=True)

out = root / "fma"; out.mkdir(exist_ok=True)
z = zipfile.ZipFile(root / "fma_zip" / "fma_small.zip")
mp3s = [i for i in z.namelist() if i.endswith(".mp3")]
random.seed(0); random.shuffle(mp3s)
n_hours = int(sys.argv[1]) if len(sys.argv) > 1 else 2
k = 0
for name in mp3s:
    if k >= n_hours * 3600 // 30: break
    data = z.read(name)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                       input=data, capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 1000: continue
    (out / (Path(name).stem + ".wav")).write_bytes(r.stdout); k += 1
print("fma clips:", k)
