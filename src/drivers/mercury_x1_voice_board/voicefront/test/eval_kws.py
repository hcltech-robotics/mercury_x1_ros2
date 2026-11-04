#!/usr/bin/env python3
"""Wake-word evaluation on synthetic array input: detection rate, direction error, false wakes.

  eval_kws.py --pos DIR --neg DIR [--noise-db -50 --hum] [--echo WAV ...] -- [voicefront options]

Every positive clip (contains the keyword) and negative clip (does not) is placed at a random angle,
one after another with 1.5 s gaps, then voicefront runs on the result. A wake counts as a hit when it falls
inside a positive clip (+1.2 s); anything else is a false wake.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_synthetic import RATE, delayed, load, C, PITCH  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VF = os.path.join(HERE, "..", "build", "voicefront")


def build(clips, noise_db, hum, talk_db, echo, echo_db, seed):
    rng = np.random.default_rng(seed)
    xm = (np.arange(4) - 1.5) * PITCH
    items, t = [], 1.0
    for path, positive in clips:
        s = load(path)
        items.append((s, t, positive, float(rng.uniform(10, 170))))
        t += len(s) / RATE + 1.5
    n = int((t + 1.0) * RATE)
    mics, ref = np.zeros((4, n)), np.zeros(n)
    for s, start, _, ang in items:
        s = s * 10 ** (talk_db / 20) / (np.sqrt(np.mean(s[np.abs(s) > 1e-3] ** 2)) + 1e-12)
        for m in range(4):
            mics[m] += delayed(s, -xm[m] * np.cos(np.radians(ang)) / C, n, int(start * RATE))
    if echo:  # the robot keeps talking (looped clips) through an echo path
        e = np.concatenate([load(p) for p in echo])
        e = np.tile(e, n // len(e) + 1)[:n] * 0.3 / (np.max(np.abs(e)) + 1e-12)
        ref[:] = e
        g = 10 ** (echo_db / 20) / (np.sqrt(np.mean(e[np.abs(e) > 1e-3] ** 2)) + 1e-12)
        tt = np.arange(int(0.15 * RATE)) / RATE
        for m in range(4):
            ir = rng.standard_normal(len(tt)) * np.exp(-tt / 0.04) * 0.08
            mics[m] += g * (delayed(e, 0.25 / C - xm[m] * np.cos(np.radians(170.0)) / C, n, 0) + np.convolve(e, ir)[:n])
    for m in range(4):
        mics[m] += 10 ** (noise_db / 20) * rng.standard_normal(n)
    if hum:
        tt = np.arange(n) / RATE
        h = sum(np.sin(2 * np.pi * f * tt + p) / k for k, (f, p) in enumerate([(150, 0.3), (200, 1.1), (300, 2.0)], 1))
        mics += 10 ** (-40 / 20) * h
    out = np.zeros((n, 6))
    out[:, :4], out[:, 4] = mics.T, ref
    return np.clip(np.round(out * 32768), -32768, 32767).astype("<i2"), items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--noise-db", type=float, default=-50.0)
    ap.add_argument("--talk-db", type=float, default=-26.0)
    ap.add_argument("--hum", action="store_true")
    ap.add_argument("--echo", action="append", default=[])
    ap.add_argument("--echo-db", type=float, default=-26.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--raw", default="/tmp/eval_kws.raw")
    ap.add_argument("vf_args", nargs="*")
    a = ap.parse_args()

    clips = [(p, True) for p in sorted(glob.glob(os.path.join(a.pos, "*.wav")))]
    clips += [(p, False) for p in sorted(glob.glob(os.path.join(a.neg, "*.wav")))]
    rng = np.random.default_rng(a.seed)
    rng.shuffle(clips)
    if not os.path.exists(a.raw):
        pcm, items = build(clips, a.noise_db, a.hum, a.talk_db, a.echo, a.echo_db, a.seed)
        pcm.tofile(a.raw)
    else:  # reuse the cached input; the placement is recomputed with the same seed
        items, t, rng2 = [], 1.0, np.random.default_rng(a.seed)
        for path, positive in clips:
            s = load(path)
            items.append((s, t, positive, float(rng2.uniform(10, 170))))
            t += len(s) / RATE + 1.5

    r = subprocess.run([VF, "--status-s", "0", "--doa-hz", "0", "--no-output", *a.vf_args],
                       stdin=open(a.raw, "rb"), capture_output=True, text=True)
    wakes = []
    for line in r.stderr.splitlines():
        if line.startswith("{"):
            j = json.loads(line)
            if j.get("type") == "wake":
                wakes.append(j)
    hits, false, errs = set(), 0, []
    for w in wakes:
        t = w["t"]
        match = None
        for i, (s, start, pos, ang) in enumerate(items):
            if start <= t <= start + len(s) / RATE + 1.2:
                match = i
        if match is not None and items[match][2]:
            if match not in hits and w.get("angle") is not None:
                errs.append(abs(w["angle"] - items[match][3]))
            hits.add(match)
        else:
            false += 1
    npos = sum(1 for it in items if it[2])
    if os.environ.get("EVAL_MISSES"):
        pos_paths = [c[0] for c in clips]
        for i, it in enumerate(items):
            if it[2] and i not in hits:
                print("  missed", os.path.basename(pos_paths[i]), f"at {it[3]:.0f} deg")
    neg_s = sum(len(it[0]) / RATE for it in items if not it[2])
    print(f"hits {len(hits)}/{npos} ({100 * len(hits) / npos:.0f} %), false wakes {false} "
          f"(in {neg_s:.0f} s of other speech), angle error median {np.median(errs) if errs else float('nan'):.1f} "
          f"deg, 90th pct {np.percentile(errs, 90) if errs else float('nan'):.1f} deg")


if __name__ == "__main__":
    main()
