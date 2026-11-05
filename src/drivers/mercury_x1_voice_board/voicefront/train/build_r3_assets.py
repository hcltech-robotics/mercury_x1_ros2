"""Round-3 assets from the robot recordings (val8.raw, played clips, play log):
  robot_rirs/   monitor -> mic impulse responses (Wiener deconvolution over all clips), one per mic
  robot_noise/  the robot's room noise (fan hum) in 10 s pieces, also speed-shifted (hum frequency drift)
  ff_pos_train/ real far-field "hey mercury" (14 voices x 2 phrasings x 4 mics), tight cut, board level (+20 dB)
  ff_test/      the other 14 voices + all negatives, mic 0 (held out)"""
import os, sys, numpy as np, soundfile as sf, scipy.signal as ss
S = sys.argv[1]; O = f"{S}/r3"; fs = 16000
x = np.fromfile(f"{S}/val8.raw", np.int16).reshape(-1, 8)[:, :4].astype(np.float32) / 32768
plays = [l.split() for l in open(f"{S}/live_play.log") if not l.startswith("T0")]
voices = sorted({p[1].split("_", 2)[2][:-4] for p in plays if "_pos_" in p[1]})
train_v = set(voices[0::2]); print("train voices", sorted(train_v)); print("test voices", sorted(set(voices) - train_v))
for d in ("robot_rirs", "robot_noise", "ff_pos_train", "ff_test"): os.makedirs(f"{O}/{d}", exist_ok=True)
def clip16(n):
    y, sr = sf.read(f"{S}/clips/{n}", dtype="float32"); return ss.resample_poly(y, fs, sr) if sr != fs else y
t_first = float(plays[0][0]); c0 = clip16(plays[0][1])
off = int(np.argmax(np.abs(ss.fftconvolve(x[:120 * fs, 0], c0[::-1], "valid"))))
L = 8192; NFFT = 1 << 15
Sxy = np.zeros((4, NFFT // 2 + 1), complex); Sxx = np.zeros(NFFT // 2 + 1)
for t, n in plays:
    c = clip16(n); g = off + int((float(t) - t_first) * fs)
    a = max(0, g - fs); r = ss.fftconvolve(x[a:g + fs + len(c), 0], c[::-1], "valid"); p = a + int(np.argmax(np.abs(r)))
    seg = x[p - 400: p - 400 + len(c) + L]                       # 25 ms before the direct sound
    X = np.fft.rfft(c, NFFT); Sxx += np.abs(X) ** 2
    for k in range(4): Sxy[k] += np.fft.rfft(seg[:, k], NFFT) * np.conj(X)
    v = n.split("_", 2)[2][:-4]
    if "_pos_" in n and v in train_v:
        for k in range(4):
            sf.write(f"{O}/ff_pos_train/{n[:-4]}_m{k}.wav", np.clip(x[p - 800: p + len(c) + 2400, k] * 10, -1, 1), fs)
    else:
        sf.write(f"{O}/ff_test/{n}", x[p - 8000: p + len(c) + 8000, 0], fs)
for k in range(4):
    H = Sxy[k] / (Sxx + 1e-3 * Sxx.max())
    h = np.fft.irfft(H, NFFT)[:L]
    h *= np.r_[np.ones(L - 1024), np.hanning(2048)[1024:]]          # fade the tail
    h /= np.abs(h).max()
    sf.write(f"{O}/robot_rirs/robot_mic{k}.wav", h.astype(np.float32), fs)
    e = np.cumsum(h[::-1] ** 2)[::-1]; edb = 10 * np.log10(e / e[0] + 1e-12)
    t20 = np.argmax(edb < -20) / fs
    print("mic%d IR: direct at %.1f ms, -20 dB decay after %.0f ms" % (k, np.argmax(np.abs(h)) / fs * 1000, t20 * 1000))
noise = x[: max(0, off - fs)]
for k in range(4):
    for i in range(0, len(noise) - 10 * fs, 10 * fs):
        seg = noise[i:i + 10 * fs, k]
        for sp in (0.94, 0.97, 1.0, 1.03, 1.06):               # fan speed: hum frequency +- 6 %
            y = ss.resample_poly(seg, 100, int(round(100 * sp))) if sp != 1.0 else seg
            sf.write(f"{O}/robot_noise/hum_m{k}_{i // fs:02d}_{sp:.2f}.wav", y.astype(np.float32), fs)
print({d: len(os.listdir(f"{O}/{d}")) for d in ("robot_rirs", "robot_noise", "ff_pos_train", "ff_test")})
print("noise rms %.1f dBFS (board level +20 dB: %.1f)" % (10 * np.log10((noise[:, 0] ** 2).mean()), 10 * np.log10((noise[:, 0] ** 2).mean()) + 20))
