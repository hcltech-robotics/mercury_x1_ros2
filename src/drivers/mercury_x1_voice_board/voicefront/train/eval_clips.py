"""Evaluate a trained openWakeWord model (onnx backend, 1280-sample frames) on the real test clips
and estimate false wakes/hour on the validation negative feature set.

usage: python eval_clips.py <model.onnx> [validation_set_features.npy]
"""
import sys, glob, os
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal
import onnxruntime as ort
from openwakeword.model import Model

model_path = sys.argv[1]
val_npy = sys.argv[2] if len(sys.argv) > 2 else None
root = Path.home() / "oww_hey_mercury"
clips_dir = root / "testclips"
THRESHOLDS = (0.5, 0.3)
FRAME = 1280


def load16k(path):
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != 16000:
        y = scipy.signal.resample_poly(y, 16000, sr)
    return (np.clip(y, -1, 1) * 32767).astype(np.int16), sr


def score_stream(wav_i16):
    """Return per-frame score array for one wav (model state reset per file)."""
    oww = Model(wakeword_models=[model_path], inference_framework="onnx")
    name = list(oww.models.keys())[0]
    scores = []
    n = len(wav_i16) // FRAME
    for i in range(n):
        p = oww.predict(wav_i16[i * FRAME:(i + 1) * FRAME])
        scores.append(float(p[name]))
    return np.array(scores)


def count_events(scores, thr, refractory_frames=15):
    """Count distinct activations: a crossing above thr followed by >= refractory frames before the next count."""
    events, last = 0, -10_000
    for i, s in enumerate(scores):
        if s >= thr and i - last > refractory_frames:
            events += 1
            last = i
    return events


def top_peaks(scores, k, min_sep=12):
    """k largest local maxima at least min_sep frames apart (for multi-clip recordings)."""
    order = np.argsort(scores)[::-1]
    peaks = []
    for i in order:
        if all(abs(i - j) >= min_sep for j in peaks):
            peaks.append(i)
        if len(peaks) == k:
            break
    return sorted(float(scores[i]) for i in peaks)


lines = []
def out(s=""):
    print(s, flush=True); lines.append(s)

out(f"model: {model_path}  size={os.path.getsize(model_path)} bytes")
import onnx
m = onnx.load(model_path)
out(f"onnx opset: {[o.version for o in m.opset_import]}  inputs: {[(i.name, i.type.tensor_type.elem_type, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}")
# batch dim is fixed to 1 in the export; make it dynamic in-memory for the batched validation sweep
for v in list(m.graph.input) + list(m.graph.output):
    v.type.tensor_type.shape.dim[0].dim_param = "N"
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

# 1) clean TTS clips
out("\n== clean TTS clips (hey_*.wav) ==")
clean = []
for f in sorted(clips_dir.glob("hey_*.wav")):
    w, sr = load16k(f)
    s = score_stream(w)
    clean.append(s.max())
    out(f"{f.name:22s} sr={sr} dur={len(w)/16000:.2f}s max={s.max():.3f}")
for t in THRESHOLDS:
    out(f"clean >= {t}: {sum(c >= t for c in clean)}/{len(clean)}")

# 2) board mic (5 clips), native and +20 dB
out("\n== board_mic0_monitor_5clips (5 clips, ~2 m, quiet) ==")
board = {}
for tag, fn in (("native", "board_mic0_monitor_5clips.wav"), ("x10 (+20 dB)", "board_mic0_monitor_5clips_x10.wav")):
    w, sr = load16k(clips_dir / fn)
    peak_dbfs = 20 * np.log10(max(np.abs(w).max(), 1) / 32767)
    s = score_stream(w)
    pk = top_peaks(s, 5)
    board[tag] = pk
    out(f"{fn} [{tag}] peak={peak_dbfs:.1f} dBFS  overall max={s.max():.3f}  top-5 clip peaks={['%.3f' % p for p in pk]}")
    for t in THRESHOLDS:
        out(f"   >= {t}: {sum(p >= t for p in pk)}/5   (distinct activations: {count_events(s, t)})")

# 3) hdmi beam (2 clips)
out("\n== hdmi_beam (2 clips, processed beam) ==")
w, sr = load16k(clips_dir / "hdmi_beam.wav")
s = score_stream(w)
pk = top_peaks(s, 2)
out(f"hdmi_beam.wav sr={sr} overall max={s.max():.3f} top-2 clip peaks={['%.3f' % p for p in pk]}")
for t in THRESHOLDS:
    out(f"   >= {t}: {sum(p >= t for p in pk)}/2   (distinct activations: {count_events(s, t)})")

# 4) talk3 unrelated speech
out("\n== talk3.wav (unrelated speech, must not trigger) ==")
w, sr = load16k(clips_dir / "talk3.wav")
s = score_stream(w)
out(f"talk3.wav sr={sr} dur={len(w)/16000:.1f}s max={s.max():.3f}")
for t in THRESHOLDS:
    out(f"   >= {t}: {'TRIGGERED' if s.max() >= t else 'ok'}")

# 5) false wakes/hour on the validation negative feature set (precomputed embeddings)
if val_npy and os.path.exists(val_npy):
    out("\n== validation negative set (precomputed features) ==")
    feats = np.load(val_npy, mmap_mode="r")
    out(f"features shape={feats.shape}")
    inp = sess.get_inputs()[0].name
    win = feats.shape[1] if feats.ndim == 3 else 16
    if feats.ndim == 3:   # already windowed [N, 16, 96]
        probs = []
        for i in range(0, feats.shape[0], 4096):
            probs.append(sess.run(None, {inp: np.asarray(feats[i:i + 4096], dtype=np.float32)})[0].ravel())
        probs = np.concatenate(probs)
        hours = feats.shape[0] * 0.08 / 3600
    else:                 # continuous [T, 96] -> sliding windows of 16 (stride 1 frame = 80 ms)
        T = feats.shape[0]
        probs = np.zeros(T - win + 1, dtype=np.float32)
        for i in range(0, T - win + 1, 4096):
            idx = np.arange(i, min(i + 4096, T - win + 1))
            batch = np.stack([feats[j:j + win] for j in idx]).astype(np.float32)
            probs[idx] = sess.run(None, {inp: batch})[0].ravel()
        hours = T * 0.08 / 3600
    out(f"duration ~{hours:.2f} h, max score={probs.max():.3f}")
    for t in THRESHOLDS:
        ev = count_events(probs, t)
        out(f"   >= {t}: {ev} activations -> {ev / hours:.2f} false wakes/hour")

Path(sys.argv[3] if len(sys.argv) > 3 else root / "eval_clips.txt").write_text("\n".join(lines) + "\n")
