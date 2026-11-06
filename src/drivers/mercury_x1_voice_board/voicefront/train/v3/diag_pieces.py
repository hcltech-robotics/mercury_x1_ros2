"""Which noise pieces (x10) trigger a model? prints pieces with max score >= 0.2"""
import sys, glob, numpy as np, soundfile as sf, scipy.signal as ss
from openwakeword.model import Model
def rd(f):
    y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0]
    return ss.resample_poly(y, 16000, sr).astype(np.float32) if sr != 16000 else y
P = {f.split("/")[-1]: rd(f) for f in sorted(glob.glob("r3/robot_noise/*_1.00.wav"))}
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    y = rd(f); n = f.split("/")[-1][:12]; P[n + "_head"] = y[:4*16000]; P[n + "_tail"] = y[36*16000:]
for n in ("fan_full", "fan_slow"):
    x = np.fromfile(f"r4/robot_extra/{n}.raw", np.int16).reshape(-1, 6).astype(np.float32) / 32768
    for k in range(4): P[f"{n}_m{k}"] = x[:, k].copy()
for name, path in [a.split("=") for a in sys.argv[1:]]:
    m = Model(wakeword_models=[path], inference_framework="onnx"); hits = []
    for k, y in P.items():
        for g in (10.0, 30.0):
            m.reset(); x = (np.clip(np.concatenate([np.zeros(8000, np.float32), y * g]), -1, 1) * 32767).astype(np.int16)
            p = max(list(m.predict(x[i:i+1280]).values())[0] for i in range(0, len(x)-1279, 1280))
            if p >= 0.2: hits.append(f"{k}x{g:.0f}={p:.2f}")
    print(name, len(P), "pieces; hits:", " ".join(hits) or "none", flush=True)
