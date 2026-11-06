"""Round 4 ("v3" model) feature set: every clip goes through the ROBOT PATH jointly
(monitor->mic impulse response of the robot, then the robot's own hum/hiss at the SNR measured on the
far-field recordings), unlike round 3 where RIR and noise were drawn independently at random.

Outputs in r4/feat/:
  positive_robot.npy     piper (7757) + 32k LibriTTS-R positives, robot path         [N, 16, 96]
  negative_robot.npy     24k adversarial negatives via robot path + robot speech + pure robot noise
  positive_test_robot.npy / negative_test_robot.npy   the 2000-clip test sets via robot path (checkpoint selection)
  robot_fp_stream.wav    30 min of robot noise with the robot talking (false-wake check)
usage: python build_r4.py [n_pos_libri n_neg_adv]
"""
import glob, os, sys, time, numpy as np, soundfile as sf, scipy.signal as ss
from multiprocessing import Pool
from numpy.lib.format import open_memmap

L = 32000                      # openWakeWord total_length (2 s -> 16 embeddings)
M = "hey_mercury_model/hey_mercury"
OUT = os.environ.get("R4_OUT", "r4/feat")
os.makedirs(OUT, exist_ok=True)
N_POS_LIBRI = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
N_NEG_ADV = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
SNR_LO, SNR_HI = float(os.environ.get("R4_SNR_LO", -2.0)), float(os.environ.get("R4_SNR_HI", 14.0))
P_ROBOT_RIR = float(os.environ.get("R4_PROBOT", 0.7))   # rest: 2/3 of the remainder MIT RIR, 1/3 none    # measured real far-field: 2.6..11 dB (median 5.4), widened a bit


def rd(f, sr_out=16000):
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    y = y[:, 0]
    return ss.resample_poly(y, sr_out, sr).astype(np.float32) if sr != sr_out else y


# ---------- pools (module globals, inherited by forked workers) ----------
ROBOT_RIR = [rd(f) for f in sorted(glob.glob("r3/robot_rirs/robot_mic*.wav"))]
MIT_RIR = [rd(f) for f in sorted(glob.glob("mit_rirs/*.wav"))[:: max(1, len(glob.glob("mit_rirs/*.wav")) // 60)]]
MIT_RIR = [h[: 16000] / (np.abs(h).max() + 1e-9) for h in MIT_RIR]
NOISE = [rd(f) for f in sorted(glob.glob("r3/robot_noise/*.wav"))]
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    y = rd(f)
    NOISE += [y[: 4 * 16000], y[36 * 16000:]]
for n in ("fan_full", "fan_slow"):
    x = np.fromfile(f"r4/robot_extra/{n}.raw", np.int16).reshape(-1, 6).astype(np.float32) / 32768
    NOISE += [x[:, k].copy() for k in range(4)]
NOISE = [n for n in NOISE if len(n) >= L]
ROBOT_SPEECH = []              # the robot talking, already recorded through the robot mics (16..36 s of the pc files)
for f in sorted(glob.glob("r4/pc_noise/*.wav")):
    y = rd(f)[16 * 16000: 36 * 16000]
    ROBOT_SPEECH += [y[i: i + L] for i in range(0, len(y) - L, 16000)]
print(f"OUT={OUT} SNR {SNR_LO}..{SNR_HI} p_robot_rir {P_ROBOT_RIR}")
print(f"pools: robot RIR {len(ROBOT_RIR)}, MIT RIR {len(MIT_RIR)}, noise pieces {len(NOISE)}, robot speech pieces {len(ROBOT_SPEECH)}")


def rms(y):
    return float(np.sqrt((y ** 2).mean() + 1e-12))


def place(y, rng):
    """left-pad into a 2 s window, clip ends 0..200 ms before the end (openWakeWord convention)."""
    if len(y) > L - 3200:
        y = y[-(L - 3200):]
    x = np.zeros(L, np.float32)
    end = L - rng.integers(0, 3200)
    x[end - len(y): end] = y
    return x


def robot_path(args):
    """(path_or_array, seed, kind) -> int16 clip of L samples through the robot path."""
    src, seed, kind = args
    rng = np.random.default_rng(seed)
    if kind == "noise":                                            # pure robot noise negative
        n = NOISE[rng.integers(len(NOISE))]
        o = rng.integers(0, len(n) - L + 1)
        x = n[o: o + L].copy()
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-40, -8) / 20)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)
    if kind == "robot_speech":                                     # the robot talking, as recorded by its own mics
        x = ROBOT_SPEECH[src].copy()
        x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-24, -1) / 20)
        return (np.clip(x, -1, 1) * 32767).astype(np.int16)
    y = rd(src) if isinstance(src, str) else src
    if rng.random() < 0.3:                                         # speed perturbation
        sp = rng.choice([0.9, 0.95, 1.05, 1.1])
        y = ss.resample_poly(y, 100, int(round(100 * sp))).astype(np.float32)
    x = place(y, rng)
    r = rng.random()
    if r < P_ROBOT_RIR:
        h = ROBOT_RIR[rng.integers(len(ROBOT_RIR))]
    elif r < P_ROBOT_RIR + (1 - P_ROBOT_RIR) * 2 / 3:
        h = MIT_RIR[rng.integers(len(MIT_RIR))]
    else:
        h = None
    if h is not None:
        x = ss.fftconvolve(x, h)[:L].astype(np.float32)
    x /= (np.abs(x).max() + 1e-9)
    fr = x[: L // 1600 * 1600].reshape(-1, 1600)
    e = np.sqrt((fr ** 2).mean(1))
    act = e[e > 0.1 * e.max()]                                     # speech-active 100 ms frames
    s_rms = float(np.sqrt((act ** 2).mean())) if len(act) else rms(x)
    n = NOISE[rng.integers(len(NOISE))]
    o = rng.integers(0, len(n) - L + 1)
    n = n[o: o + L]
    snr = rng.uniform(SNR_LO, SNR_HI)
    x = x + n * (s_rms / (rms(n) * 10 ** (snr / 20)))
    x = x / (np.abs(x).max() + 1e-9) * 10 ** (rng.uniform(-30, -3) / 20)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def embed_jobs(jobs, out_file, F, pool, chunk=2048):
    fp = open_memmap(out_file, mode="w+", dtype=np.float32, shape=(len(jobs), 16, 96))
    t0 = time.time()
    for i in range(0, len(jobs), chunk):
        clips = np.stack(pool.map(robot_path, jobs[i: i + chunk], chunksize=32))
        fp[i: i + len(clips)] = F.embed_clips(clips, batch_size=256)
        if (i // chunk) % 4 == 0:
            print(f"  {out_file}: {i + len(clips)}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    fp.flush()
    del fp


if __name__ == "__main__":
    rng = np.random.default_rng(4)
    libri = sorted(f for f in glob.glob(f"{M}/positive_train/*.wav") if "/piper_" not in f and "/ff_" not in f)
    piper = sorted(glob.glob(f"{M}/positive_train/piper_*.wav"))
    adv = sorted(f for f in glob.glob(f"{M}/negative_train/*.wav") if "/hum_" not in f)
    pos_src = piper + list(rng.choice(libri, N_POS_LIBRI, replace=False))
    neg_src = list(rng.choice(adv, N_NEG_ADV, replace=False))
    extra_clean_neg = [rd(f) for f in ("r4/robot_extra/robot1.wav", "r4/robot_extra/robot2.wav", "r4/robot_extra/p2.wav")]
    pos_jobs = [(f, 1_000_000 + i, "pos") for i, f in enumerate(pos_src)]
    neg_jobs = [(f, 2_000_000 + i, "neg") for i, f in enumerate(neg_src)]
    neg_jobs += [(k, 3_000_000 + i, "robot_speech") for i, k in enumerate(list(range(len(ROBOT_SPEECH))) * 12)]
    neg_jobs += [(extra_clean_neg[i % 3], 4_000_000 + i, "neg") for i in range(240)]
    neg_jobs += [(None, 5_000_000 + i, "noise") for i in range(2400)]
    rng.shuffle(neg_jobs)
    pos_test = [(f, 6_000_000 + i, "pos") for i, f in enumerate(sorted(glob.glob(f"{M}/positive_test/*.wav")))]
    neg_test = [(f, 7_000_000 + i, "neg") for i, f in enumerate(sorted(glob.glob(f"{M}/negative_test/*.wav")))]
    print(f"jobs: pos {len(pos_jobs)} neg {len(neg_jobs)} pos_test {len(pos_test)} neg_test {len(neg_test)}", flush=True)

    # false-wake stream: 30 min of robot noise, the robot talking every ~20 s
    s = []
    while sum(len(a) for a in s) < 30 * 60 * 16000:
        n = NOISE[rng.integers(len(NOISE))]
        s.append(n[: 10 * 16000] if len(n) >= 10 * 16000 else n)
        if rng.random() < 0.5:
            s.append(ROBOT_SPEECH[rng.integers(len(ROBOT_SPEECH))])
    stream = np.concatenate(s)
    sf.write(f"{OUT}/robot_fp_stream.wav", np.clip(stream * 10, -1, 1), 16000)   # board level +20 dB, as the wake input
    print("fp stream %.1f min" % (len(stream) / 16000 / 60), flush=True)

    from openwakeword.utils import AudioFeatures
    F = AudioFeatures(device="gpu")
    with Pool(10) as pool:
        embed_jobs(pos_test, f"{OUT}/positive_test_robot.npy", F, pool)
        embed_jobs(neg_test, f"{OUT}/negative_test_robot.npy", F, pool)
        embed_jobs(pos_jobs, f"{OUT}/positive_robot.npy", F, pool)
        embed_jobs(neg_jobs, f"{OUT}/negative_robot.npy", F, pool)
    print("R4_BUILD_DONE")
