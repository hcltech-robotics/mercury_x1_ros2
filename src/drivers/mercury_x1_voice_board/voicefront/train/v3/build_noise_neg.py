"""Pure robot-noise negatives (hum/hiss at board level and +20 dB, speed-shifted, some spliced) -> r4/feat/noise_neg.npy"""
import glob, numpy as np, soundfile as sf, scipy.signal as ss
from numpy.lib.format import open_memmap
L = 32000; rng = np.random.default_rng(7)
def rd(f):
    y, sr = sf.read(f, dtype="float32", always_2d=True); y = y[:, 0]
    return ss.resample_poly(y, 16000, sr).astype(np.float32) if sr != 16000 else y
NOISE = [rd(f) for f in sorted(glob.glob("r3/robot_noise/*.wav"))]
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    y = rd(f); NOISE += [y[:4*16000], y[36*16000:]]
for n in ("fan_full", "fan_slow"):
    x = np.fromfile(f"r4/robot_extra/{n}.raw", np.int16).reshape(-1, 6).astype(np.float32) / 32768
    NOISE += [x[:, k].copy() for k in range(4)]
def piece(n_len):
    n = NOISE[rng.integers(len(NOISE))]; o = rng.integers(0, len(n) - n_len + 1); return n[o:o + n_len].copy()
N = 8000
from openwakeword.utils import AudioFeatures
F = AudioFeatures(device="gpu")
fp = open_memmap("r4/feat/noise_neg.npy", mode="w+", dtype=np.float32, shape=(N, 16, 96))
for i0 in range(0, N, 500):
    clips = []
    for _ in range(500):
        if rng.random() < 0.3:
            k = rng.integers(4000, 28000); x = np.concatenate([piece(k), piece(L - k)])
        else:
            x = piece(L)
        if rng.random() < 0.3:
            sp = rng.uniform(0.93, 1.07); x = ss.resample_poly(x, 100, int(round(100 * sp))).astype(np.float32)
            x = np.pad(x, (0, max(0, L - len(x))))[:L]
        if rng.random() < 0.2:   # slow level ramp
            x = x * np.linspace(rng.uniform(0.3, 1), rng.uniform(0.3, 1), L)
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-42, -6) / 20)
        clips.append((np.clip(x, -1, 1) * 32767).astype(np.int16))
    fp[i0:i0 + 500] = F.embed_clips(np.stack(clips), batch_size=250)
fp.flush(); print("NOISE_NEG_DONE", N)
