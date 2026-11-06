"""Per-clip far-field scores for several models + speech-band SNR per clip."""
import sys, glob, numpy as np, soundfile as sf, scipy.signal as ss
from openwakeword.model import Model
models = {"r2": "export_r2/hey_mercury.onnx", "IJK": "ens/IJK.onnx",
          "D": "sweep_D/hey_mercury.onnx", "M": "sweep_v3_M/hey_mercury.onnx"}
ms = {k: Model(wakeword_models=[v], inference_framework="onnx") for k, v in models.items()}
def score(m, y):
    m.reset()
    x = (np.clip(np.concatenate([np.zeros(16000, np.float32), y, np.zeros(8000, np.float32)]), -1, 1) * 32767).astype(np.int16)
    return max(list(m.predict(x[i:i + 1280]).values())[0] for i in range(0, len(x) - 1279, 1280))
def snr(y):
    sos = ss.butter(4, [300, 3500], "bp", fs=16000, output="sos"); b = ss.sosfilt(sos, y)
    fr = b[: len(b) // 160 * 160].reshape(-1, 160); e = 10 * np.log10((fr ** 2).mean(1) + 1e-12)
    return np.percentile(e, 95) - np.percentile(e, 10)
sets = [("ff_test", sorted(glob.glob("r3/ff_test/*.wav")), 10.0), ("ff_train_m0", sorted(glob.glob("r3/ff_pos_train/*_m0.wav")), 1.0),
        ("ff_train_m2", sorted(glob.glob("r3/ff_pos_train/*_m2.wav")), 1.0)]
print("%-28s %6s " % ("clip", "snr") + " ".join("%5s" % k for k in models))
for name, files, g in sets:
    print("==", name)
    tot = {k: [0, 0] for k in models}
    for f in files:
        y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0] * g
        s = {k: score(m, y) for k, m in ms.items()}
        for k in models:
            if "_pos_" in f: tot[k][0] += s[k] >= 0.5; tot[k][1] += s[k] >= 0.3
        print("%-28s %6.1f " % (f.split("/")[-1][:28], snr(y)) + " ".join("%5.2f" % s[k] for k in models), flush=True)
    print("%-35s " % "pos >=0.5 / >=0.3" + " ".join("%2d/%2d" % tuple(tot[k]) for k in models))
