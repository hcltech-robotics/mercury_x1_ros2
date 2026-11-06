"""Fair robot false-wake stream: 30 min of robot noise with the robot talking, 100 ms cross-fades at every join (no splice clicks).
Writes r4/feat/robot_fp_stream_xf.wav once; scores the models given as name=path."""
import os, sys, glob, numpy as np, soundfile as sf, scipy.signal as ss
from openwakeword.model import Model
OUT = "r4/feat/robot_fp_stream_xf.wav"
def rd(f):
    y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0]
    return ss.resample_poly(y, 16000, sr).astype(np.float32) if sr != 16000 else y
if not os.path.exists(OUT):
    rng = np.random.default_rng(21)
    NOISE = [rd(f) for f in sorted(glob.glob("r3/robot_noise/*.wav"))]
    for f in sorted(glob.glob("r4/pc_noise/*.wav")):
        y = rd(f); NOISE += [y[:4*16000], y[36*16000:]]
    SP = []
    for f in sorted(glob.glob("r4/pc_noise/*.wav")):
        y = rd(f)[16*16000:36*16000]; SP += [y[i:i+2*16000] for i in range(0, len(y)-2*16000, 16000)]
    X = 1600; fade = np.linspace(0, 1, X, dtype=np.float32)
    out = NOISE[0][:5*16000].copy()
    while len(out) < 30*60*16000:
        n = NOISE[rng.integers(len(NOISE))]
        seg = n[:10*16000] if rng.random() < 0.7 else SP[rng.integers(len(SP))] * 0.3   # robot talk at a post-AEC-ish level
        out[-X:] = out[-X:] * (1 - fade) + seg[:X] * fade
        out = np.concatenate([out, seg[X:]])
    sf.write(OUT, np.clip(out * 10, -1, 1), 16000); print("wrote", OUT, len(out)/16000/60, "min")
y = rd(OUT); x = (np.clip(y, -1, 1) * 32767).astype(np.int16)
def events(p, t):
    ev, last = 0, -99
    for i in np.flatnonzero(p >= t):
        if i - last > 15: ev += 1; last = i
    return ev
for name, path in [a.split("=") for a in sys.argv[1:]]:
    m = Model(wakeword_models=[path], inference_framework="onnx")
    p = np.array([list(m.predict(x[i:i+1280]).values())[0] for i in range(0, len(x)-1279, 1280)])
    print(f"XF {name}: {events(p,.5)}@0.5 {events(p,.3)}@0.3 {events(p,.2)}@0.2 max {p.max():.2f} in 30 min", flush=True)
