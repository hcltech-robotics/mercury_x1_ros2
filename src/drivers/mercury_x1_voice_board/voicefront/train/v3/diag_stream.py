"""Where do robot-stream false wakes come from? noise-only stream (x10) vs robot speech pieces (x1 raw, and x10 clipped), per model."""
import sys, glob, numpy as np, soundfile as sf, scipy.signal as ss
from openwakeword.model import Model
rng = np.random.default_rng(11)
def rd(f):
    y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0]
    return ss.resample_poly(y, 16000, sr).astype(np.float32) if sr != 16000 else y
NOISE = [rd(f) for f in sorted(glob.glob("r3/robot_noise/*.wav"))]
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    y = rd(f); NOISE += [y[:4*16000], y[36*16000:]]
SP = []
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    SP.append(rd(f)[16*16000:36*16000])
noise = np.concatenate([NOISE[rng.integers(len(NOISE))] for _ in range(200)])[:20*60*16000] * 10
speech = np.concatenate(SP)   # 60 s raw robot speech at native (clipped) level
def events(p, t):
    ev, last = 0, -99
    for i in np.flatnonzero(p >= t):
        if i - last > 15: ev += 1; last = i
    return ev
for name, path in [a.split("=") for a in sys.argv[1:]]:
    m = Model(wakeword_models=[path], inference_framework="onnx")
    out = []
    for tag, y in (("noise x10 20min", noise), ("robot speech x1", speech), ("robot speech x10 clipped", np.clip(speech*10, -1, 1)), ("robot speech x0.1 (AEC-like)", speech*0.1)):
        m.reset(); x = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        p = np.array([list(m.predict(x[i:i+1280]).values())[0] for i in range(0, len(x)-1279, 1280)])
        out.append(f"{tag}: {events(p,.5)}@0.5 {events(p,.3)}@0.3 max {p.max():.2f}")
    print(name, " | ".join(out), flush=True)
