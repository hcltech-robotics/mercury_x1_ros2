"""Generate a few TTS samples per spelling variant and transcribe with whisper to check pronunciation."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/oww_hey_mercury/piper-sample-generator"))
from generate_samples import generate_samples
import whisper, glob
variants = ["hey mercury", "hey, mercury", "hey mercury!", "hey murcury"]
base = os.path.expanduser("~/oww_hey_mercury/tts_check")
for i, v in enumerate(variants):
    d = f"{base}/v{i}"
    generate_samples(text=[v], max_samples=6, batch_size=6, noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[1.0], output_dir=d, max_speakers=600)
m = whisper.load_model("small.en")
for i, v in enumerate(variants):
    print(f"=== variant {i!r}: {v!r}")
    for w in sorted(glob.glob(f"{base}/v{i}/*.wav")):
        r = m.transcribe(w, language="en", fp16=True)
        print(f"   {os.path.basename(w)} -> {r['text'].strip()!r}")
