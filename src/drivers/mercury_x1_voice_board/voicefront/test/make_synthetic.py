#!/usr/bin/env python3
"""Synthetic voicefront input: talkers placed at angles on the 4-mic linear array, room noise, robot echo.

Writes raw S16_LE, 16 kHz, 6 channels in the USB layout (0-3 mics, 4 speaker reference, 5 zero).

  make_synthetic.py --talk hey.wav@60@1.0 --talk hey.wav@140@4.0 --echo talk1.wav@0.5 --len 8 -o in.raw
  --talk WAV@ANGLE@START   a talker (plane wave from ANGLE deg, 0 = mic-3 end, 90 = broadside)
  --echo WAV@START         played by the robot: clean on channel 4, through an echo path into the mics
"""
import argparse
import wave

import numpy as np
from scipy.signal import resample_poly

RATE, PITCH, C = 16000, 0.035, 343.0


def load(path):
    with wave.open(path) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float64).reshape(-1, ch)[:, 0] / 32768.0
    if sr != RATE:
        g = np.gcd(sr, RATE)
        x = resample_poly(x, RATE // g, sr // g)
    return x


def delayed(x, delay_s, n_out, start):
    """x delayed by a fractional delay (FFT phase shift), placed at sample `start` in an n_out buffer"""
    n = 1 << int(np.ceil(np.log2(len(x) + 64)))
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    y = np.fft.irfft(np.fft.rfft(x, n) * np.exp(-2j * np.pi * f * delay_s), n)
    out = np.zeros(n_out)
    seg = y[: max(0, min(n, n_out - start))]
    out[start : start + len(seg)] = seg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--talk", action="append", default=[])
    ap.add_argument("--echo", action="append", default=[])
    ap.add_argument("--talk-db", type=float, default=-26.0, help="talker rms at the mics, dBFS")
    ap.add_argument("--echo-db", type=float, default=-20.0, help="echo rms at the mics, dBFS")
    ap.add_argument("--noise-db", type=float, default=-55.0, help="diffuse-ish noise rms per mic, dBFS")
    ap.add_argument("--hum", action="store_true", help="add a 150-300 Hz hum like the robot room")
    ap.add_argument("--len", type=float, default=8.0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    n = int(a.len * RATE)
    rng = np.random.default_rng(1)
    xm = (np.arange(4) - 1.5) * PITCH
    mics = np.zeros((4, n))
    ref = np.zeros(n)

    for spec in a.talk:
        path, ang, start = spec.split("@")
        s = load(path)
        s *= 10 ** (a.talk_db / 20) / (np.sqrt(np.mean(s[np.abs(s) > 1e-3] ** 2)) + 1e-12)
        for m in range(4):
            tau = -xm[m] * np.cos(np.radians(float(ang))) / C
            mics[m] += delayed(s, tau, n, int(float(start) * RATE))

    for spec in a.echo:
        path, start = spec.split("@")
        s = load(path)
        s *= 0.3 / (np.max(np.abs(s)) + 1e-12)
        st = int(float(start) * RATE)
        ref[st : st + len(s)] += s[: max(0, n - st)]
    if a.echo:
        # echo path: direct sound from the speaker (end-fire, 25 cm) + a decaying diffuse tail per mic
        g = 10 ** (a.echo_db / 20) / (np.sqrt(np.mean(ref[np.abs(ref) > 1e-3] ** 2)) + 1e-12)
        for m in range(4):
            tau = 0.25 / C - xm[m] * np.cos(np.radians(170.0)) / C
            direct = delayed(ref, tau, n, 0)
            t = np.arange(int(0.15 * RATE)) / RATE
            ir = rng.standard_normal(len(t)) * np.exp(-t / 0.04) * 0.08
            mics[m] += g * (direct + np.convolve(ref, ir)[:n])

    for m in range(4):
        mics[m] += 10 ** (a.noise_db / 20) * rng.standard_normal(n)
    if a.hum:
        t = np.arange(n) / RATE
        hum = sum(np.sin(2 * np.pi * f * t + p) / k for k, (f, p) in enumerate([(150, 0.3), (200, 1.1), (300, 2.0)], 1))
        for m in range(4):
            mics[m] += 10 ** (-40 / 20) * hum

    out = np.zeros((n, 6))
    out[:, :4] = mics.T
    out[:, 4] = ref
    np.clip(np.round(out * 32768), -32768, 32767).astype("<i2").tofile(a.out)
    print(f"wrote {a.out}: {a.len:.1f} s, 6 ch")


if __name__ == "__main__":
    main()
