"""Extra 'hey mercury' positives from multi-speaker / British / accented Piper ONNX voices (CPU), resampled to
16 kHz into the openWakeWord positive_train dir. usage: gen_piper_voices.py <voice_name> <n_samples>"""
import os, sys, uuid, wave
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, scipy.signal, scipy.io.wavfile
from piper_sample_generator.__main__ import generate_samples_onnx
root = Path.home() / "oww_hey_mercury"
name, n = sys.argv[1], int(sys.argv[2])
raw = root / "extra_pos_raw" / name
generate_samples_onnx(text=["hey mercury", "hey, mercury", "hey mercury!"], output_dir=raw,
                      model=str(root / "piper_voices" / f"{name}.onnx"), max_samples=n,
                      length_scales=(0.8, 1.0, 1.2), noise_scales=(0.667, 0.9), noise_scale_ws=(0.8, 1.0))
dst = root / "hey_mercury_model/hey_mercury/positive_train"
k = 0
for f in raw.glob("*.wav"):
    sr, y = scipy.io.wavfile.read(f)
    if len(y) < sr * 0.3: continue
    if sr != 16000:
        y = scipy.signal.resample_poly(y.astype(np.float32), 16000, sr)
        y = np.clip(y, -32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(dst / f"piper_{name}_{uuid.uuid4().hex}.wav", 16000, y); k += 1
print(name, "positives added:", k)
