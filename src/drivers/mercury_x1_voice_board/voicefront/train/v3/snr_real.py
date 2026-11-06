import glob, numpy as np, soundfile as sf
noise = sf.read("r3/robot_noise/hum_m0_00_1.00.wav", dtype="float32")[0]
nr = np.sqrt((noise**2).mean())
v = []
for f in sorted(glob.glob("r3/ff_test/*_pos_*.wav")) + sorted(glob.glob("r3/ff_pos_train/*_m0.wav")):
    y = sf.read(f, dtype="float32", always_2d=True)[0][:, 0]
    g = 0.1 if "ff_pos_train" in f else 1.0
    y = y * g
    sp = y[8000:-8000] if "ff_test" in f else y[800:-2400]
    fr = sp[: len(sp)//1600*1600].reshape(-1, 1600); e = np.sqrt((fr**2).mean(1))
    act = e[e > np.percentile(e, 40)]            # speech-active 100 ms frames
    snr = 20*np.log10(np.sqrt((act**2).mean()) / nr)
    v.append(snr)
v = np.array(v); print("real speech/noise SNR dB: min %.1f p10 %.1f median %.1f p90 %.1f max %.1f" % (v.min(), np.percentile(v,10), np.median(v), np.percentile(v,90), v.max()))
print("noise rms %.1f dBFS" % (20*np.log10(nr)))
