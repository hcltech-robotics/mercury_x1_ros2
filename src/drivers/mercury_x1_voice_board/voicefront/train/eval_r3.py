"""Round-3 evaluation of a hey_mercury model: held-out far-field robot recordings (14 voices the model never
heard, mic 0 at board level +20 dB), clean Kokoro clips, false wakes / hour on the 11 h validation features.
usage: python eval_r3.py MODEL.onnx"""
import sys, glob, numpy as np, soundfile as sf, scipy.signal, onnx, onnxruntime as ort
from openwakeword.model import Model
model = sys.argv[1]
TH = (0.5, 0.3)
def scores(y):
    m = Model(wakeword_models=[model], inference_framework="onnx")
    x = (np.clip(np.concatenate([np.zeros(16000, np.float32), y, np.zeros(8000, np.float32)]), -1, 1) * 32767).astype(np.int16)
    return max(list(m.predict(x[i:i + 1280]).values())[0] for i in range(0, len(x) - 1279, 1280))
def load(f, gain=1.0):
    y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0]
    return (scipy.signal.resample_poly(y, 16000, sr) if sr != 16000 else y) * gain
def report(name, files, gain):
    s = {f.split("/")[-1]: scores(load(f, gain)) for f in files}
    pos = [v for k, v in s.items() if "_pos_" in k]; neg = [v for k, v in s.items() if "_neg_" in k]
    for t in TH:
        print(f"{name} >= {t}: wake {sum(v >= t for v in pos)}/{len(pos)}  false {sum(v >= t for v in neg)}/{len(neg)}")
    return s
print("model", model)
ff = report("far-field held-out voices", sorted(glob.glob("r3/ff_test/*.wav")), 10.0)
report("far-field train voices (seen)", sorted(glob.glob("r3/ff_pos_train/*_m0.wav")), 1.0)
report("clean Kokoro (28 voices)", sorted(glob.glob("r3/clean/*.wav")), 1.0)
mo = onnx.load(model)
for v in list(mo.graph.input) + list(mo.graph.output): v.type.tensor_type.shape.dim[0].dim_param = "N"
sess = ort.InferenceSession(mo.SerializeToString(), providers=["CPUExecutionProvider"]); inp = sess.get_inputs()[0].name
f = np.load("validation_set_features.npy", mmap_mode="r"); T = f.shape[0]; win = 16
p = np.zeros(T - win + 1, np.float32)
for i in range(0, T - win + 1, 8192):
    idx = np.arange(i, min(i + 8192, T - win + 1))
    p[idx] = sess.run(None, {inp: np.stack([f[j:j + win] for j in idx]).astype(np.float32)})[0].ravel()
h = T * 0.08 / 3600
for t in TH:
    ev, last = 0, -99
    for i in np.flatnonzero(p >= t):
        if i - last > 15:
            ev += 1
            last = i
    print(f"validation {h:.1f} h >= {t}: {ev} false wakes = {ev / h:.2f} / h")
