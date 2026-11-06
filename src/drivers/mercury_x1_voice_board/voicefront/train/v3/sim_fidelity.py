"""Does clean -> robot RIR + robot noise reproduce the real far-field scores?
per clip: real / sim(rir+noise) / sim(rir) / sim(noise) / clean, for models r2 and D."""
import glob, numpy as np, soundfile as sf, scipy.signal as ss
from openwakeword.model import Model
ms = {k: Model(wakeword_models=[v], inference_framework="onnx")
      for k, v in {"r2": "export_r2/hey_mercury.onnx", "D": "sweep_D/hey_mercury.onnx"}.items()}


def score(m, y):
    m.reset()
    x = (np.clip(np.concatenate([np.zeros(16000, np.float32), y, np.zeros(8000, np.float32)]), -1, 1) * 32767).astype(np.int16)
    return max(list(m.predict(x[i:i + 1280]).values())[0] for i in range(0, len(x) - 1279, 1280))


def rd(f):
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    y = y[:, 0]
    return ss.resample_poly(y, 16000, sr) if sr != 16000 else y


def rms(y):
    return np.sqrt((y ** 2).mean() + 1e-12)


rir = rd("r3/robot_rirs/robot_mic0.wav")
noise = rd("r3/robot_noise/hum_m0_00_1.00.wav") * 10
G = 10.0
rows = []
for f in sorted(glob.glob("r3/ff_test/*_pos_*.wav")):
    n = f.split("/")[-1]
    real = rd(f) * G
    clean = rd("r3/clean/" + n)
    rv = ss.fftconvolve(clean, rir)[: len(clean) + 4000]
    # match the loudness of the real recording speech part (real has 0.5 s noise pads either side)
    tgt = rms(real[8000:-8000])
    rv *= tgt / rms(rv)
    cl = clean * tgt / rms(clean)
    nz = noise[: len(rv) + 16000]
    sims = {"real": real,
            "rir+noise": np.concatenate([nz[:8000], rv + nz[8000:8000 + len(rv)], nz[-8000:]]),
            "rir": rv,
            "noise": np.concatenate([nz[:8000], cl + nz[8000:8000 + len(cl)], nz[-8000:]]),
            "clean": cl}
    r = {k: {s: score(m, y) for s, y in sims.items()} for k, m in ms.items()}
    rows.append(r)
    print("%-22s " % n[:22] + "  ".join("%s[%s]" % (k, " ".join("%.2f" % r[k][s] for s in sims)) for k in ms), flush=True)
for k in ms:
    for t in (0.5, 0.3):
        print(k, t, {s: sum(r[k][s] >= t for r in rows) for s in rows[0][k]})
print("order: real rir+noise rir noise clean")
