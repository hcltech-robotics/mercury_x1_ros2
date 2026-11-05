"""Convert AudioSet parquet shards (flac bytes) into 16 kHz mono int16 wavs in ~/oww_hey_mercury/audioset_16k."""
import io
from pathlib import Path
import numpy as np, scipy.io.wavfile, soundfile as sf, scipy.signal
import pyarrow.parquet as pq
root = Path.home() / "oww_hey_mercury"
out = root / "audioset_16k"; out.mkdir(exist_ok=True)
n = 0
for p in sorted((root / "audioset").glob("*.parquet")):
    pf = pq.ParquetFile(p)
    print(p.name, pf.metadata.num_rows, pf.schema_arrow.names, flush=True)
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            a = row["audio"]
            try:
                y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            except Exception as e:
                print("skip", e); continue
            y = y.mean(axis=1)
            if sr != 16000: y = scipy.signal.resample_poly(y, 16000, sr)
            name = ((a.get("path") or row.get("video_id") or f"a{n}").split("/")[-1].rsplit(".", 1)[0]) + ".wav"
            scipy.io.wavfile.write(out / name, 16000, (np.clip(y, -1, 1) * 32767).astype(np.int16)); n += 1
print("audioset clips:", n, "total hours ~", n * 10 / 3600)
