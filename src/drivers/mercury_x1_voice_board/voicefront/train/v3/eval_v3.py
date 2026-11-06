"""v3 evaluation of a hey_mercury openWakeWord model (onnx), streaming 80 ms frames like voicefront.
  a) far-field through the robot mics: held-out voices (28 pos / 12 neg, mic 0 x10), seen voices mic 0 and mic 2
  b) clean Kokoro (56 pos / 12 neg), round-2 testclips (6 clean TTS, board mic 5 native + x10, hdmi beam 2, talk3)
  c) false wakes / h on the 10.7 h validation features; robot false-wake stream (30 min hum + robot talking);
     the robot recordings from this PC (person at 4..16 s, robot talking 16..36 s)
usage: python eval_v3.py MODEL.onnx [--brief]"""
import sys, glob, numpy as np, soundfile as sf, scipy.signal as ss, onnx, onnxruntime as ort
from openwakeword.model import Model
model = sys.argv[1]
brief = "--brief" in sys.argv
TH = (0.5, 0.3)
m = Model(wakeword_models=[model], inference_framework="onnx")


def stream(y, pad=True):
    m.reset()
    if pad:
        y = np.concatenate([np.zeros(16000, np.float32), y, np.zeros(8000, np.float32)])
    x = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    return np.array([list(m.predict(x[i:i + 1280]).values())[0] for i in range(0, len(x) - 1279, 1280)])


def load(f, gain=1.0):
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    y = y[:, 0]
    return (ss.resample_poly(y, 16000, sr) if sr != 16000 else y) * gain


def events(p, t, refractory=15):
    ev, last = 0, -99
    for i in np.flatnonzero(p >= t):
        if i - last > refractory:
            ev += 1
            last = i
    return ev


def top_peaks(s, k, min_sep=12):
    peaks = []
    for i in np.argsort(s)[::-1]:
        if all(abs(i - j) >= min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) == k:
            break
    return sorted(float(s[i]) for i in peaks)


res = {}


def clipset(name, files, gain):
    s = {f.split("/")[-1]: stream(load(f, gain)).max() for f in files}
    pos = [v for k, v in s.items() if "_pos_" in k]
    neg = [v for k, v in s.items() if "_neg_" in k]
    for t in TH:
        res[f"{name}@{t}"] = (sum(v >= t for v in pos), len(pos), sum(v >= t for v in neg), len(neg))
        print(f"{name:28s} >= {t}: wake {sum(v >= t for v in pos):2d}/{len(pos)}  false {sum(v >= t for v in neg)}/{len(neg)}", flush=True)
    return s


print("model", model)
clipset("far-field held-out (mic0)", sorted(glob.glob("r3/ff_test/*.wav")), 10.0)
clipset("far-field seen voices mic0", sorted(glob.glob("r3/ff_pos_train/*_m0.wav")), 1.0)
clipset("far-field seen voices mic2", sorted(glob.glob("r3/ff_pos_train/*_m2.wav")), 1.0)
clipset("clean Kokoro", sorted(glob.glob("r3/clean/*.wav")), 1.0)

# round-2 testclips
clean = [stream(load(f)).max() for f in sorted(glob.glob("testclips/hey_*.wav"))]
for t in TH:
    print(f"{'testclips clean TTS':28s} >= {t}: {sum(c >= t for c in clean)}/{len(clean)}")
for tag, fn, g in (("board mic native", "board_mic0_monitor_5clips.wav", 1.0), ("board mic x10", "board_mic0_monitor_5clips_x10.wav", 1.0)):
    pk = top_peaks(stream(load("testclips/" + fn), pad=False), 5)
    for t in TH:
        print(f"{tag:28s} >= {t}: {sum(p >= t for p in pk)}/5")
pk = top_peaks(stream(load("testclips/hdmi_beam.wav"), pad=False), 2)
for t in TH:
    print(f"{'hdmi beam':28s} >= {t}: {sum(p >= t for p in pk)}/2")
s = stream(load("testclips/talk3.wav"))
print(f"{'talk3 (negative)':28s} max {s.max():.3f}")

# robot false-wake stream + PC recordings of the robot mics
s = stream(load("r4/feat/robot_fp_stream.wav"), pad=False)
dur_h = len(s) * 0.08 / 3600
for t in TH:
    ev = events(s, t)
    res[f"robot_stream@{t}"] = ev
    print(f"{'robot noise+talk stream':28s} >= {t}: {ev} activations in {dur_h * 60:.0f} min (max {s.max():.3f})")
for fn, g in (("01_raw_mic0_as_recorded", 10.0), ("31_live_raw_mic0_same_run", 10.0), ("01_raw_mic0_stock_gain30", 1.0)):
    s = stream(load(f"r4/pc_noise/{fn}.wav", g), pad=False)
    i0, i1 = int(4 / 0.08), int(16 / 0.08)
    print(f"{fn:28s} person 4-16 s max {s[i0:i1].max():.2f} | robot 16-36 s max {s[i1:int(36 / 0.08)].max():.2f} | noise max {max(s[:i0].max(), s[int(36 / 0.08):].max()):.2f}")

# validation negatives (precomputed features, 10.7 h)
mo = onnx.load(model)
for v in list(mo.graph.input) + list(mo.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "N"
sess = ort.InferenceSession(mo.SerializeToString(), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
f = np.load("validation_set_features.npy", mmap_mode="r")
T, win = f.shape[0], 16
p = np.zeros(T - win + 1, np.float32)
for i in range(0, T - win + 1, 8192):
    idx = np.arange(i, min(i + 8192, T - win + 1))
    p[idx] = sess.run(None, {inp: np.stack([f[j:j + win] for j in idx]).astype(np.float32)})[0].ravel()
h = T * 0.08 / 3600
for t in TH + (0.2, 0.15, 0.1):
    ev = events(p, t)
    res[f"val_fph@{t}"] = ev / h
    print(f"{'validation 10.7 h':28s} >= {t}: {ev} false wakes = {ev / h:.2f} / h")
ff = res["far-field held-out (mic0)@0.5"], res["far-field held-out (mic0)@0.3"]
print(f"SUMMARY {model}: ff {ff[0][0]}/28@0.5 {ff[1][0]}/28@0.3 | ffneg {ff[0][2]}/12@0.5 {ff[1][2]}/12@0.3 | "
      f"val {res['val_fph@0.5']:.2f}/h@0.5 {res['val_fph@0.3']:.2f}/h@0.3 | robot stream {res['robot_stream@0.5']}@0.5 {res['robot_stream@0.3']}@0.3")
