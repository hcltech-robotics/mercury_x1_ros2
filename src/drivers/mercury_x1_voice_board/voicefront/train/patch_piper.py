"""Patch piper_sample_generator to emit 16 kHz wavs (openWakeWord's augmentation assumes 16 kHz)."""
from pathlib import Path
p = Path.home() / "oww_hey_mercury/piper-sample-generator/piper_sample_generator/__main__.py"
s = p.read_text()
old1 = "            audio_numpy = audio.cpu().numpy()\n"
new1 = old1 + ("            if sample_rate != 16000:  # patched: resample to 16 kHz for openWakeWord\n"
               "                import torchaudio.functional as _F\n"
               "                audio_numpy = _F.resample(audio.detach().float(), sample_rate, 16000).cpu().numpy()\n")
old2 = "                    wav_file.setframerate(sample_rate)\n"
new2 = "                    wav_file.setframerate(16000)  # patched\n"
assert s.count(old1) == 1 and s.count(old2) == 1
p.write_text(s.replace(old1, new1).replace(old2, new2)); print("patched")
