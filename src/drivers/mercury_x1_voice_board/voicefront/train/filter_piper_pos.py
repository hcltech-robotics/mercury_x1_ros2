"""Whisper (small.en, batched) filter for Piper-voice positives in positive_train: keep clips transcribed as
'hey/hay/pay <one word m...[ck]...r...>' (Mercury, McCurry, Macury, Makiri...); delete the rest. Drops l2arctic."""
import glob, os, re, torch, whisper, numpy as np
root = os.path.expanduser("~/oww_hey_mercury/hey_mercury_model/hey_mercury/positive_train")
fs = sorted(glob.glob(root + "/piper_*.wav"))
drop = [f for f in fs if "l2arctic" in f]
for f in drop: os.remove(f)
fs = [f for f in fs if "l2arctic" not in f]
print("l2arctic dropped:", len(drop), "to check:", len(fs), flush=True)
pat = re.compile(r"\b(hey|hay|pay|hi)\b[ ,!.]*m[a-z']*[ck][a-z']*r[a-z']*\b", re.I)
m = whisper.load_model("small.en", device="cuda")
opts = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=True)
stats = {}
B = 64
for i in range(0, len(fs), B):
    batch = fs[i:i + B]
    mels = torch.stack([whisper.log_mel_spectrogram(whisper.pad_or_trim(whisper.load_audio(f))) for f in batch]).to("cuda")
    for f, r in zip(batch, whisper.decode(m, mels, opts)):
        v = os.path.basename(f).split("_")[1]
        k = bool(pat.search(r.text))
        s = stats.setdefault(v, [0, 0]); s[0] += k; s[1] += 1
        if not k: os.remove(f)
    if i % (B * 20) == 0: print(i, flush=True)
for v, (k, n) in sorted(stats.items()): print(f"{v}: kept {k}/{n}")
print("total kept:", sum(k for k, n in stats.values()))
